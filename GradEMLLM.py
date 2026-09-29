import torch
from PIL import Image
from transformers import AutoProcessor, LlavaForConditionalGeneration
import pandas as pd
import numpy as np
from tqdm import tqdm
from collections import defaultdict
import torch.nn.functional as F
from captum.attr import visualization
import os
from util import reset_seeds
import math
from typing import Any, Dict, List, Optional, Tuple
import matplotlib.pyplot as plt
from transformers.models.llama.modeling_llama import apply_rotary_pos_emb, repeat_kv



class AttributionGenerator:
    """
    Attribution generator that supports:
      1) text-only causal LM (LLaMA family)
      2) HF LlavaForConditionalGeneration via AutoProcessor (VQA)

    Main APIs:
      - generate_text(prompt, ...)
      - generate_vqa(image, question, ...)

    Main change compared with your old version:
      - Reconstruct OFFICIAL q/k/v path:
            q_proj/k_proj/v_proj
            -> reshape to heads
            -> apply RoPE to q/k
            -> repeat_kv for GQA
            -> q @ k^T / sqrt(d)
      - Then use reconstructed qk score of the LAST query as lambda.
    """

    def __init__(
        self,
        model,
        tokenizer=None,
        processor=None,
        n_layers: int = 1,
        normalize: bool = True,
    ):
        self.model = model
        self.processor = processor
        self.tokenizer = tokenizer if tokenizer is not None else (
            processor.tokenizer if processor is not None else None
        )
        if self.tokenizer is None:
            raise ValueError("You must provide tokenizer (text-only) or processor (for LLaVA).")

        self.n_layers = n_layers
        self.normalize = normalize
        self.device = next(self.model.parameters()).device

        self.hooks: List[torch.utils.hooks.RemovableHandle] = []
        self.hooked_attn_modules = []

        # buffers: each idx stores ONLY the latest forward result for that layer
        self.attention_outputs = [[] for _ in range(self.n_layers)]   # self_attn output, keep graph
        self.attention_weights = [[] for _ in range(self.n_layers)]   # optional, detached
        self.projection_q = [[] for _ in range(self.n_layers)]        # q_proj out, detached
        self.projection_k = [[] for _ in range(self.n_layers)]        # k_proj out, detached
        self.projection_v = [[] for _ in range(self.n_layers)]        # v_proj out, detached
        self.attn_meta = [[] for _ in range(self.n_layers)]           # pre-hook kwargs

        layers = self._get_decoder_layers(self.model)
        self._register_hooks(layers[-self.n_layers:])

    # ---------------------------------------------------------
    # Model structure helpers
    # ---------------------------------------------------------
    @staticmethod
    def _get_decoder_layers(model):
        """
        Find decoder layers for:
          - LLaMA-like: model.model.layers
          - HF LLaVA:   model.language_model.model.layers
        """
        if hasattr(model, "model") and hasattr(model.model, "layers"):
            return model.model.layers

        if (
            hasattr(model, "language_model")
            and hasattr(model.language_model, "model")
            and hasattr(model.language_model.model, "layers")
        ):
            return model.language_model.model.layers

        if hasattr(model, "get_decoder"):
            dec = model.get_decoder()
            if hasattr(dec, "layers"):
                return dec.layers

        raise AttributeError("Cannot find decoder layers.")

    def _register_hooks(self, layers):
        self.hooked_attn_modules = []

        for layer_idx, layer in enumerate(layers):
            attn_module = layer.self_attn
            self.hooked_attn_modules.append(attn_module)

            # ---- pre-hook: capture kwargs entering self_attn ----
            def make_attn_prehook(idx):
                def hook(module, args, kwargs):
                    meta = {
                        "attention_mask": kwargs.get("attention_mask", None),
                        "position_embeddings": kwargs.get("position_embeddings", None),
                        "position_ids": kwargs.get("position_ids", None),
                    }
                    if len(self.attn_meta[idx]) > 0:
                        self.attn_meta[idx].pop()
                    self.attn_meta[idx].append(meta)
                return hook

            self.hooks.append(
                attn_module.register_forward_pre_hook(
                    make_attn_prehook(layer_idx),
                    with_kwargs=True,
                )
            )

            # ---- forward hook: capture self_attn output ----
            def make_attn_hook(idx):
                def hook(module, inp, out):
                    if isinstance(out, (tuple, list)):
                        attn_out = out[0]
                        attn_w = out[1] if len(out) > 1 else None
                    else:
                        attn_out = out
                        attn_w = None
                    pre_o_output = inp[0]

                    if len(self.attention_outputs[idx]) > 0:
                        self.attention_outputs[idx].pop()
                    self.attention_outputs[idx].append(pre_o_output)  # keep graph

                    if attn_w is not None and torch.is_tensor(attn_w):
                        if len(self.attention_weights[idx]) > 0:
                            self.attention_weights[idx].pop()
                        self.attention_weights[idx].append(attn_w.detach())
                return hook

            self.hooks.append(attn_module.o_proj.register_forward_hook(make_attn_hook(layer_idx)))

            # ---- q/k/v projection hooks ----
            def make_proj_hook(idx, proj_lst):
                def hook(module, inp, out):
                    if len(proj_lst[idx]) > 0:
                        proj_lst[idx].pop()
                    proj_lst[idx].append(out.detach())  # values only
                return hook

            self.hooks.append(attn_module.q_proj.register_forward_hook(
                make_proj_hook(layer_idx, self.projection_q)
            ))
            self.hooks.append(attn_module.k_proj.register_forward_hook(
                make_proj_hook(layer_idx, self.projection_k)
            ))
            self.hooks.append(attn_module.v_proj.register_forward_hook(
                make_proj_hook(layer_idx, self.projection_v)
            ))

    def _remove_hooks(self):
        for h in self.hooks:
            h.remove()
        self.hooks.clear()
        self.hooked_attn_modules.clear()

    def _reset_buffers(self):
        for i in range(self.n_layers):
            self.attention_outputs[i].clear()
            self.attention_weights[i].clear()
            self.projection_q[i].clear()
            self.projection_k[i].clear()
            self.projection_v[i].clear()
            self.attn_meta[i].clear()

    def _re_register_hooks(self):
        layers = self._get_decoder_layers(self.model)
        self._register_hooks(layers[-self.n_layers:])

    # ---------------------------------------------------------
    # Official q/k/v reconstruction
    # ---------------------------------------------------------
    @staticmethod
    def _minmax_norm(x: torch.Tensor) -> torch.Tensor:
        minv = x.min(dim=-1, keepdim=True).values
        maxv = x.max(dim=-1, keepdim=True).values
        return (x - minv) / (maxv - minv + 1e-8)
    
    def _masked_minmax_norm(self, x: torch.Tensor, valid_mask: torch.Tensor, eps: float = 1e-8):
        """
        x:          [B, H, S] or [B, H, K]
        valid_mask: same shape as x, bool tensor
        """
        x_for_min = torch.where(valid_mask, x, torch.full_like(x, float("inf")))
        x_min = x_for_min.min(dim=-1, keepdim=True).values

        x_for_max = torch.where(valid_mask, x, torch.full_like(x, float("-inf")))
        x_max = x_for_max.max(dim=-1, keepdim=True).values

        # fallback in degenerate cases
        bad = (~torch.isfinite(x_min)) | (~torch.isfinite(x_max))
        x_min = torch.where(bad, torch.zeros_like(x_min), x_min)
        x_max = torch.where(bad, torch.ones_like(x_max), x_max)

        out = (x - x_min) / (x_max - x_min + eps)
        out = torch.where(valid_mask, out, torch.zeros_like(out))
        return out

    @staticmethod
    def _proj_to_heads(x: torch.Tensor, head_dim: int) -> torch.Tensor:
        # x: [B, S, H*D] -> [B, H, S, D]
        bsz, seqlen, _ = x.shape
        return x.view(bsz, seqlen, -1, head_dim).transpose(1, 2).contiguous()

    @staticmethod
    def _heads_to_flat(x: torch.Tensor) -> torch.Tensor:
        # [B, H, S, D] -> [B, S, H*D]
        return x.transpose(1, 2).reshape(x.shape[0], x.shape[2], -1).contiguous()

    def _get_layer_attn_module(self, layer_idx: int):
        return self.hooked_attn_modules[layer_idx]

    def _reconstruct_qkv(self, layer_idx: int) -> Dict[str, torch.Tensor]:
        """
        Reconstruct OFFICIAL q/k/v used by HF LLaMA attention:
          q_proj/k_proj/v_proj -> reshape -> RoPE -> repeat_kv (for k,v)
        """
        if len(self.attn_meta[layer_idx]) == 0:
            raise RuntimeError("attn_meta is empty. Did the forward hooks run?")

        meta = self.attn_meta[layer_idx][0]
        position_embeddings = meta["position_embeddings"]
        if position_embeddings is None:
            raise RuntimeError(
                "position_embeddings not found in self_attn kwargs. "
                "This code expects recent HF LLaMA/LLaVA implementations."
            )

        q_flat = self.projection_q[layer_idx][0]  # [B, S, H*D]
        k_flat = self.projection_k[layer_idx][0]  # [B, S, H_kv*D]
        v_flat = self.projection_v[layer_idx][0]  # [B, S, H_kv*D]

        attn_module = self._get_layer_attn_module(layer_idx)
        head_dim = attn_module.head_dim

        q = self._proj_to_heads(q_flat, head_dim)  # [B, H,   S, D]
        k = self._proj_to_heads(k_flat, head_dim)  # [B, Hkv, S, D]
        v = self._proj_to_heads(v_flat, head_dim)  # [B, Hkv, S, D]

        cos, sin = position_embeddings
        q, k = apply_rotary_pos_emb(q, k, cos, sin)

        k_rep = repeat_kv(k, attn_module.num_key_value_groups)  # [B, H, S, D]
        v_rep = repeat_kv(v, attn_module.num_key_value_groups)  # [B, H, S, D]

        return {
            "q": q,
            "k": k,
            "v": v,
            "k_rep": k_rep,
            "v_rep": v_rep,
            "attention_mask": meta["attention_mask"],
            "scaling": attn_module.scaling,
        }

    def _get_qk_scores_last_query(
        self,
        layer_idx: int,
        return_probs: bool = False,
    ) -> torch.Tensor:
        """
        Compute official last-query score:
            logits = q_last @ k^T / sqrt(d) + attention_mask
        Then aggregate over heads by mean.

        return_probs=False:
            if normalize=True  -> min-max normalized mean logits over heads, [B, S]
            else               -> softmax(mean logits), [B, S]

        return_probs=True:
            return mean attention probabilities over heads, [B, S]
        """
        rec = self._reconstruct_qkv(layer_idx)

        q_last = rec["q"][:, :, -1:, :]                  # [B, H, 1, D]
        k_rep = rec["k_rep"]                            # [B, H, S, D]
        logits = torch.matmul(q_last, k_rep.transpose(2, 3)) * rec["scaling"]  # [B, H, 1, S]

        attn_mask = rec["attention_mask"]
        if attn_mask is not None:
            logits = logits + attn_mask[:, :, -1:, : logits.shape[-1]]

        logits = logits.squeeze(-2)  # [B, H, S]

        if return_probs:
            probs = torch.softmax(logits, dim=-1, dtype=torch.float32).to(q_last.dtype)
            return probs.mean(dim=1)  # [B, S]

        logits_mean = logits.mean(dim=1)  # [B, S]
        if self.normalize:
            return self._minmax_norm(logits_mean)
        else:
            return torch.softmax(logits_mean, dim=-1)

    def get_last_qk_logits(self, layer_idx: int) -> torch.Tensor:
        """
        For paper / debugging:
        returns per-head LAST-query qk logits after RoPE + GQA + mask, shape [B, H, S]
        """
        rec = self._reconstruct_qkv(layer_idx)
        q_last = rec["q"][:, :, -1:, :]
        logits = torch.matmul(q_last, rec["k_rep"].transpose(2, 3)) * rec["scaling"]
        attn_mask = rec["attention_mask"]
        if attn_mask is not None:
            logits = logits + attn_mask[:, :, -1:, : logits.shape[-1]]
        return logits.squeeze(-2)

    def get_last_qk_attention(self, layer_idx: int) -> torch.Tensor:
        """
        For paper / debugging:
        returns per-head LAST-query attention probabilities, shape [B, H, S]
        """
        logits = self.get_last_qk_logits(layer_idx)
        return torch.softmax(logits, dim=-1, dtype=torch.float32)

    def _get_repeated_v_flat(self, layer_idx: int) -> torch.Tensor:
        """
        Expand GQA value heads to full attention heads and flatten back to hidden size:
            [B, Hkv, S, D] --repeat_kv--> [B, H, S, D] --flatten--> [B, S, H*D]
        This should match grad_last shape from self_attn output more naturally than ad-hoc repeating.
        """
        rec = self._reconstruct_qkv(layer_idx)
        return self._heads_to_flat(rec["v_rep"])  # [B, S, hidden_size]

    @staticmethod
    def _grad_eclip_from_lambda(
        v: torch.Tensor,         # [B, S, hidden_size]
        grad_last: torch.Tensor, # [B, hidden_size]
        lam: torch.Tensor,       # [B, S]
    ) -> torch.Tensor:
        if v.shape[-1] != grad_last.shape[-1]:
            raise RuntimeError(
                f"Shape mismatch after GQA expansion: v.shape={tuple(v.shape)}, "
                f"grad_last.shape={tuple(grad_last.shape)}"
            )
        emap = grad_last.unsqueeze(1) * v * lam.unsqueeze(-1)  # [B, S, hidden_size]
        emap = F.relu(emap.sum(-1))                            # [B, S]
        return emap

    # ---------------------------------------------------------
    # Image token span helpers
    # ---------------------------------------------------------
    def _get_image_token_id(self) -> Optional[int]:
        image_token_id = getattr(getattr(self.model, "config", None), "image_token_index", None)
        if image_token_id is not None:
            return int(image_token_id)

        image_token_id = getattr(self.tokenizer, "image_token_id", None)
        if image_token_id is not None:
            return int(image_token_id)

        try:
            tid = self.tokenizer.convert_tokens_to_ids("<image>")
            return int(tid) if tid is not None else None
        except Exception:
            return None

    def infer_image_positions(self, input_ids: torch.Tensor) -> Optional[Dict[str, Any]]:
        image_token_id = self._get_image_token_id()
        if image_token_id is None:
            return None

        ids = input_ids[0]
        pos = (ids == image_token_id).nonzero(as_tuple=False).squeeze(-1)
        if pos.numel() == 0:
            return None

        spans = []
        start = pos[0].item()
        prev = start
        for p in pos[1:].tolist():
            if p == prev + 1:
                prev = p
            else:
                spans.append((start, prev + 1))
                start = p
                prev = p
        spans.append((start, prev + 1))

        return {
            "positions": pos,
            "spans": spans,
            "K": int(pos.numel()),
            "image_token_id": int(image_token_id),
        }

    def infer_patch_grid_hw(self, inputs: Dict[str, torch.Tensor], K: int) -> Tuple[int, int]:
        if "pixel_values" not in inputs:
            s = int(round(math.sqrt(K)))
            return (s, s) if s * s == K else (1, K)

        pv = inputs["pixel_values"]  # [B, 3, Hpx, Wpx]
        Hpx, Wpx = pv.shape[-2], pv.shape[-1]

        patch = getattr(self.processor, "patch_size", None) if self.processor is not None else None
        if patch is None and self.processor is not None and hasattr(self.processor, "image_processor"):
            patch = getattr(self.processor.image_processor, "patch_size", None)
        if patch is None:
            patch = 14

        Hp, Wp = Hpx // patch, Wpx // patch
        if Hp * Wp == K:
            return Hp, Wp

        s = int(round(math.sqrt(K)))
        if s * s == K:
            return s, s
        return 1, K

    # ---------------------------------------------------------
    # Build VQA inputs
    # ---------------------------------------------------------
    def _build_vqa_inputs(self, image: Image.Image, question: str) -> Dict[str, torch.Tensor]:
        if self.processor is None:
            raise ValueError("processor is required for generate_vqa (LLaVA).")

        conversation = [
            {
                "role": "user",
                "content": [{"type": "image"}, {"type": "text", "text": question}],
            }
        ]

        if hasattr(self.processor, "apply_chat_template"):
            prompt = self.processor.apply_chat_template(conversation, add_generation_prompt=True)
        else:
            prompt = self.tokenizer.apply_chat_template(
                conversation,
                add_generation_prompt=True,
                tokenize=False,
            )

        inputs = self.processor(images=image, text=prompt, return_tensors="pt")

        out = {}
        for k, v in inputs.items():
            if torch.is_tensor(v):
                out[k] = v.to(self.device)
        return out

    # ---------------------------------------------------------
    # Text-only generation + replay for grads
    # ---------------------------------------------------------
    def generate_text(
        self,
        input_text: str,
        max_new_tokens: int = 300,
        temperature: float = 0.7,
        do_sample: bool = True,
        top_p: float = 1.0,
        use_cache_generate: bool = True,
    ):
        self.model.eval()

        input_ids = self.tokenizer.encode(
            input_text,
            return_tensors="pt",
            add_special_tokens=False,
        ).to(self.device)
        attn_mask = torch.ones_like(input_ids)
        prompt_len = input_ids.shape[1]

        with torch.no_grad():
            gen = self.model.generate(
                input_ids=input_ids,
                attention_mask=attn_mask,
                do_sample=do_sample,
                temperature=temperature,
                top_p=top_p,
                max_new_tokens=max_new_tokens,
                use_cache=use_cache_generate,
                pad_token_id=self.tokenizer.eos_token_id,
                return_dict_in_generate=True,
            )

        seq = gen.sequences
        new_len = seq.shape[1] - prompt_len
        emap_steps = []

        for t in range(new_len):
            self._reset_buffers()

            prefix = seq[:, :prompt_len + t]
            target = seq[:, prompt_len + t]

            prefix_mask = torch.cat(
                [
                    attn_mask,
                    torch.ones(
                        (attn_mask.shape[0], t),
                        device=self.device,
                        dtype=attn_mask.dtype,
                    ),
                ],
                dim=1,
            )

            outputs = self.model(
                input_ids=prefix,
                attention_mask=prefix_mask,
                use_cache=False,
            )
            logits = outputs.logits[:, -1, :]
            target_logit = logits.gather(-1, target.unsqueeze(-1)).squeeze(-1)

            attn_outputs = [x[0] for x in self.attention_outputs]
            grads = torch.autograd.grad(
                target_logit.sum(),
                attn_outputs,
                retain_graph=False,
            )

            per_layer = []
            for layer_idx in range(self.n_layers):
                lam = self._get_qk_scores_last_query(layer_idx, return_probs=False)  # [B, S]
                v = self._get_repeated_v_flat(layer_idx)                              # [B, S, hidden]
                grad_last = grads[layer_idx][:, -1, :].detach()                      # [B, hidden]
                per_layer.append(self._grad_eclip_from_lambda(v, grad_last, lam))    # [B, S]

            combined = torch.stack(per_layer, dim=0).sum(dim=0)  # [B, S]
            emap_steps.append(combined[0, :prompt_len].detach().cpu().float())

            if target.item() == self.tokenizer.eos_token_id:
                break

        emap_full = (
            torch.stack(emap_steps, dim=0)
            if len(emap_steps) > 0
            else torch.zeros((0, prompt_len))
        )

        self._reset_buffers()
        return emap_full, seq

    # ---------------------------------------------------------
    # LLaVA VQA generation + replay for grads
    # ---------------------------------------------------------
    def generate_vqa(
        self,
        image: Image.Image,
        question: str,
        max_new_tokens: int = 64,
        temperature: float = 0.7,
        do_sample: bool = True,
        top_p: float = 1.0,
        use_cache_generate: bool = True,
    ):
        self.model.eval()

        inputs = self._build_vqa_inputs(image, question)
        input_ids = inputs["input_ids"]
        attn_mask = inputs.get("attention_mask", torch.ones_like(input_ids))
        prompt_len = input_ids.shape[1]

        img_info = self.infer_image_positions(input_ids)
        meta = {"img_info": img_info}

        Hp = Wp = None
        if img_info is not None:
            Hp, Wp = self.infer_patch_grid_hw(inputs, img_info["K"])
            meta.update({
                "K": img_info["K"],
                "grid_hw": (Hp, Wp),
                "spans": img_info["spans"],
            })

        with torch.no_grad():
            gen = self.model.generate(
                **inputs,
                do_sample=do_sample,
                temperature=temperature,
                top_p=top_p,
                max_new_tokens=max_new_tokens,
                use_cache=use_cache_generate,
                pad_token_id=self.tokenizer.eos_token_id,
                return_dict_in_generate=True,
            )

        seq = gen.sequences
        new_len = seq.shape[1] - prompt_len
        emap_steps = []

        static_modal_inputs = {
            k: v for k, v in inputs.items()
            if k not in ["input_ids", "attention_mask"]
        }

        for t in range(new_len):
            self._reset_buffers()

            prefix = seq[:, :prompt_len + t]
            target = seq[:, prompt_len + t]

            prefix_mask = torch.cat(
                [
                    attn_mask,
                    torch.ones(
                        (attn_mask.shape[0], t),
                        device=self.device,
                        dtype=attn_mask.dtype,
                    ),
                ],
                dim=1,
            )

            outputs = self.model(
                input_ids=prefix,
                attention_mask=prefix_mask,
                use_cache=False,
                **static_modal_inputs,
            )
            logits = outputs.logits[:, -1, :]
            target_logit = logits.gather(-1, target.unsqueeze(-1)).squeeze(-1)

            attn_outputs = [x[0] for x in self.attention_outputs]
            grads = torch.autograd.grad(
                target_logit.sum(),
                attn_outputs,
                retain_graph=False,
            )

            per_layer = []
            for layer_idx in range(self.n_layers):
                # recon_mean = self._get_reconstructed_last_attention_per_head(layer_idx).mean(dim=1)   # [B, S]
                # official_mean = self._get_official_last_attention(layer_idx, src_len=seq.shape[1]).mean(dim=1)  # [B, S]

                # print(recon_mean.shape)
                # print(official_mean.shape)
                # print((recon_mean - official_mean).abs().max())
                lam = self._get_qk_scores_last_query(layer_idx, return_probs=False)  # [B, S]
                # print(lam.shape)
                # original_attn_weights = self._get_official_last_attention(layer_idx)
                # print(original_attn_weights.shape)
                v = self._get_repeated_v_flat(layer_idx)                              # [B, S, hidden]
                grad_last = grads[layer_idx][:, -1, :].detach()                      # [B, hidden]
                per_layer.append(self._grad_eclip_from_lambda(v, grad_last, lam))    # [B, S]

            combined = torch.stack(per_layer, dim=0).sum(dim=0)  # [B, S]
            emap_steps.append(combined[0, :prompt_len].detach().cpu().float())

            if target.item() == self.tokenizer.eos_token_id:
                break

        emap_full = (
            torch.stack(emap_steps, dim=0)
            if len(emap_steps) > 0
            else torch.zeros((0, prompt_len))
        )

        img_grid = None
        if img_info is not None and Hp is not None and Wp is not None:
            pos = img_info["positions"].cpu()
            img_part = emap_full.index_select(dim=1, index=pos)  # [T, K]
            if Hp * Wp == img_part.shape[1]:
                img_grid = img_part.reshape(img_part.shape[0], Hp, Wp)
            else:
                img_grid = img_part

        decoded = self.tokenizer.decode(seq[0], skip_special_tokens=True)

        self._reset_buffers()
        return emap_full, img_grid, seq, decoded, meta, input_ids
    
    def _get_official_last_attention(self, layer_idx: int, src_len: Optional[int] = None) -> torch.Tensor:
        """
        Get official attention weights of the LAST query from hook-captured out[1].

        Returns:
            [B, H, S]  if official attn is [B, H, T, S]
            [B, 1, S]  if official attn is [B, T, S] (head dim missing)
        """
        if len(self.attention_weights[layer_idx]) == 0:
            raise RuntimeError(
                "attention_weights is empty. Make sure forward() is called with "
                "output_attentions=True and model uses eager attention."
            )

        attn_w = self.attention_weights[layer_idx][0]  # detached

        if attn_w.dim() == 4:
            # [B, H, T, S]
            out = attn_w[:, :, -1, :]
        elif attn_w.dim() == 3:
            # [B, T, S]
            out = attn_w[:, -1, :].unsqueeze(1)  # -> [B, 1, S]
        else:
            raise RuntimeError(f"Unexpected official attn shape: {tuple(attn_w.shape)}")

        if src_len is not None:
            out = out[..., :src_len]

        return out


    def _get_reconstructed_last_attention_per_head(self, layer_idx: int) -> torch.Tensor:
        """
        Reconstructed official last-query attention weights per head.

        Returns:
            [B, H, S]
        """
        logits = self.get_last_qk_logits(layer_idx)  # [B, H, S], already includes scaling + mask
        return torch.softmax(logits, dim=-1, dtype=torch.float32).to(logits.dtype)


    def compare_last_attention(
        self,
        layer_idx: int,
        verbose: bool = True,
    ) -> Dict[str, float]:
        """
        Compare reconstructed attention with official attention for ONE layer.

        Returns a dict with max/mean abs diff and relative diff.
        """
        recon = self._get_reconstructed_last_attention_per_head(layer_idx)  # [B, H, S]
        official = self._get_official_last_attention(layer_idx, src_len=recon.shape[-1])  # [B, H, S] or [B,1,S]

        # if official has no head dim, try broadcasting
        if official.shape[1] == 1 and recon.shape[1] > 1:
            official = official.expand(-1, recon.shape[1], -1)

        if official.shape != recon.shape:
            raise RuntimeError(
                f"Shape mismatch: official={tuple(official.shape)}, recon={tuple(recon.shape)}"
            )

        abs_diff = (official - recon).abs()
        denom = official.abs().clamp_min(1e-12)
        rel_diff = abs_diff / denom

        metrics = {
            "max_abs_diff": abs_diff.max().item(),
            "mean_abs_diff": abs_diff.mean().item(),
            "max_rel_diff": rel_diff.max().item(),
            "mean_rel_diff": rel_diff.mean().item(),
        }

        if verbose:
            print(f"[Layer {layer_idx}]")
            for k, v in metrics.items():
                print(f"  {k}: {v:.8e}")

        return metrics


    def compare_all_layers_last_attention(self, verbose: bool = True) -> List[Dict[str, float]]:
        """
        Compare all hooked layers after one forward pass.
        """
        results = []
        for layer_idx in range(self.n_layers):
            metrics = self.compare_last_attention(layer_idx, verbose=verbose)
            metrics["layer_idx"] = layer_idx
            results.append(metrics)
        return results
    

    def forward_vqa_prompt_only(self, image: Image.Image, question: str):
        """
        只对 prompt 做一次前向，不生成答案。
        目的：拿到每层的 q_proj / k_proj，用于 grounding 分析。
        """
        self.model.eval()

        inputs = self._build_vqa_inputs(image, question)
        input_ids = inputs["input_ids"]
        attention_mask = inputs.get("attention_mask", torch.ones_like(input_ids))
        pixel_values = inputs["pixel_values"]

        # 关键：只清空，不要 rebind
        self._reset_buffers()

        with torch.no_grad():
            _ = self.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                pixel_values=pixel_values,
                use_cache=False,
                output_attentions=False,
                return_dict=True,
            )

        img_info = self.infer_image_positions(input_ids)
        Hp, Wp = self.infer_patch_grid_hw(inputs, img_info["K"])

        meta = {
            "input_ids": input_ids.detach().cpu(),
            "attention_mask": attention_mask.detach().cpu(),
            "image_positions": img_info["positions"].detach().cpu(),
            "spans": img_info["spans"],
            "K": img_info["K"],
            "grid_hw": (Hp, Wp),
        }
        return meta

    def print_prompt_tokens(self, input_ids, max_print=None):
        """
        input_ids: [1, S]
        """
        ids = input_ids[0].tolist()
        toks = self.tokenizer.convert_ids_to_tokens(ids)

        if max_print is None:
            max_print = len(toks)

        for i, tok in enumerate(toks[:max_print]):
            print(f"{i:4d}: {tok}")

    def _find_subsequence(self, full_ids, sub_ids):
        """
        full_ids: list[int]
        sub_ids: list[int]
        return: list[int] 起始位置列表
        """
        starts = []
        n, m = len(full_ids), len(sub_ids)
        if m == 0 or m > n:
            return starts

        for i in range(n - m + 1):
            if full_ids[i:i+m] == sub_ids:
                starts.append(i)
        return starts

    def find_word_token_indices(self, input_ids, word, verbose=True):
        """
        在完整 prompt 的 input_ids 中，尽量自动找到某个词对应的 token indices。
        返回一个 list[int]，比如 [590] 或 [590, 591]
        """
        full_ids = input_ids[0].tolist()

        # 尝试几种常见形式
        variants = [
            word,
            " " + word,
            word.lower(),
            " " + word.lower(),
            word.capitalize(),
            " " + word.capitalize(),
            word.upper(),
            " " + word.upper(),
        ]

        matches = []
        for v in variants:
            sub_ids = self.tokenizer.encode(v, add_special_tokens=False)
            starts = self._find_subsequence(full_ids, sub_ids)
            for s in starts:
                matches.append((s, s + len(sub_ids), v, sub_ids))

        # 去重
        uniq = []
        seen = set()
        for item in matches:
            key = (item[0], item[1])
            if key not in seen:
                uniq.append(item)
                seen.add(key)

        if len(uniq) == 0:
            if verbose:
                print(f"[WARN] cannot find token span for word={word}")
                self.print_prompt_tokens(input_ids)
            return None

        # 选最短匹配，通常更合理
        uniq = sorted(uniq, key=lambda x: (x[1] - x[0], x[0]))
        s, e, variant, sub_ids = uniq[0]

        if verbose:
            toks = self.tokenizer.convert_ids_to_tokens(full_ids[s:e])
            print(f"[INFO] word={word!r}, matched variant={variant!r}, token span=({s},{e}), tokens={toks}")

        return list(range(s, e))
    

    def _get_qk_scores_for_query(
        self,
        layer_idx: int,
        query_idx: int,
        selected_positions: Optional[torch.Tensor] = None,
        return_probs: bool = False,
        head_reduce: str = "mean",          # "mean" | "max" | "none"
        normalize: Optional[bool] = None,   # None -> use self.normalize
        normalize_scope: str = "selected",  # "selected" | "visible_all"
    ) -> torch.Tensor:
        """
        Generalized grounding score for an arbitrary query token.

        Args:
            layer_idx: hooked layer index in [0, self.n_layers)
            query_idx: token index in the current prompt/prefix
            selected_positions: optional 1D LongTensor of source positions to keep
                                (e.g. image_positions). If None, use all source tokens.
            return_probs:
                False -> return qk logits-based score
                        if normalize=True  : min-max normalized logits
                        if normalize=False : softmax probs over source tokens
                True  -> return attention probabilities over source tokens
            head_reduce:
                "mean" -> average over heads
                "max"  -> max over heads
                "none" -> keep head dimension
            normalize:
                None -> use self.normalize
            normalize_scope:
                "selected" -> if selected_positions is not None, normalize only over selected tokens
                "all"      -> normalize over all tokens first, then slice selected_positions
        Returns:
            [B, K] if head_reduce != "none"
            [B, H, K] if head_reduce == "none"
            where K = len(selected_positions) if selected_positions is not None else seq_len
        """
        if normalize is None:
            normalize = self.normalize

        rec = self._reconstruct_qkv(layer_idx)

        # q_t: [B, H, 1, D]
        q_t = rec["q"][:, :, query_idx:query_idx + 1, :]  # [B, H, 1, D]
        k_rep = rec["k_rep"]                              # [B, H, S, D]

        logits = torch.matmul(q_t, k_rep.transpose(2, 3)) * rec["scaling"]  # [B, H, 1, S]

        attn_mask = rec["attention_mask"]
        # print(attn_mask.shape)
        # print(attn_mask)
        if attn_mask is not None:
            # use the mask row corresponding to this query token
            logits = logits + attn_mask[:, :, query_idx:query_idx + 1, : logits.shape[-1]]

        logits = logits.squeeze(-2)  # [B, H, S]
        # print(logits.shape)
        # print(logits.min())
        # print(logits.max())
        # print(logits)

        # selected positions (e.g. image tokens only)

        if attn_mask is not None:
            row_mask = attn_mask[:, :, query_idx:query_idx + 1, :logits.shape[-1]].squeeze(-2)  # [B,1,S] or [B,H,S]
            visible_mask = (row_mask == 0)
            if visible_mask.shape[1] == 1 and logits.shape[1] > 1:
                visible_mask = visible_mask.expand(-1, logits.shape[1], -1)
        else:
            visible_mask = torch.ones_like(logits, dtype=torch.bool)    

        if selected_positions is not None:
            selected_positions = selected_positions.to(logits.device)

        if return_probs:
            probs = torch.softmax(logits, dim=-1, dtype=torch.float32).to(logits.dtype)  # [B, H, S]
            x = probs
        else:
            if normalize:
                if selected_positions is not None and normalize_scope == "selected":
                    x = logits.index_select(dim=-1, index=selected_positions)      # [B,H,K]
                    vmask = visible_mask.index_select(dim=-1, index=selected_positions) # [B,H,K]
                    x = self._masked_minmax_norm(x, vmask)
                elif normalize_scope == 'visible_all':
                    x = self._masked_minmax_norm(logits, visible_mask)  
                else:
                    raise ValueError("normalize_scope must be 'selected' or 'visible_all'")

            else:
                x = torch.softmax(logits, dim=-1, dtype=torch.float32).to(logits.dtype)

        # slice after normalization if needed
        if selected_positions is not None:
            if not (normalize and normalize_scope == "selected"):
                x = x.index_select(dim=-1, index=selected_positions)  # [B, H, K]

        # head reduction
        if head_reduce == "mean":
            x = x.mean(dim=1)          # [B, K] or [B, S]
        elif head_reduce == "max":
            x = x.max(dim=1).values
        elif head_reduce == "none":
            pass                       # [B, H, K] or [B, H, S]
        else:
            raise ValueError("head_reduce must be mean/max/none")

        return x
    

    def get_word_grounding_map(
        self,
        word_token_indices: List[int],
        image_positions: torch.Tensor,
        grid_hw: Tuple[int, int],
        layer_idx: int,
        head_reduce: str = "mean",
        subtoken_reduce: str = "mean",      # "mean" | "max"
        normalize: Optional[bool] = None,
        normalize_scope: str = "selected",  # "selected" | "all"
        return_probs: bool = False,
    ):
        """
        Compute grounding map of a word (possibly multiple subtokens) to image tokens.

        Returns:
            grid: [Hp, Wp]
            flat: [K]
        """
        if word_token_indices is None or len(word_token_indices) == 0:
            raise ValueError("word_token_indices is empty")

        maps = []
        for q_idx in word_token_indices:
            sim = self._get_qk_scores_for_query(
                layer_idx=layer_idx,
                query_idx=q_idx,
                selected_positions=image_positions,
                return_probs=return_probs,
                head_reduce=head_reduce,
                normalize=normalize,
                normalize_scope=normalize_scope,
            )  # [B, K] or [B, H, K]

            if sim.dim() == 3:
                # if head_reduce="none", collapse heads here for visualization
                # sim = sim.mean(dim=1)
                pass

            maps.append(sim)  # each [B, K]

        maps = torch.stack(maps, dim=0)  # [n_subtokens, B, K]

        if subtoken_reduce == "mean":
            sim = maps.mean(dim=0)              # [B, K]
        elif subtoken_reduce == "max":
            sim = maps.max(dim=0).values
        else:
            raise ValueError("subtoken_reduce must be mean or max")

        Hp, Wp = grid_hw
        flat = sim[0].detach().cpu()
        if sim.dim() == 3:
            grid = flat.reshape(flat.shape[0], Hp, Wp)
        else:
            grid = flat.reshape(Hp, Wp)

        return grid, flat
    

    def get_layerwise_grounding_maps(
        self,
        image: Image.Image,
        question: str,
        target_words=("car", "color"),
        head_reduce: str = "mean",
        subtoken_reduce: str = "mean",
        normalize: Optional[bool] = None,
        normalize_scope: str = "selected",
        return_probs: bool = False,
        print_tokens: bool = True,
    ):
        """
        Run prompt-only forward once, then compute layer-wise grounding maps for target words.

        Returns:
            {
                "meta": meta,
                "tokens": tokens,
                "layer_ids": layer_ids,
                "word_token_indices": {word: [idxs]},
                "maps": {word: tensor[L, Hp, Wp] or None},
            }
        """
        meta = self.forward_vqa_prompt_only(image, question)

        input_ids = meta["input_ids"]
        image_positions = meta["image_positions"]
        grid_hw = meta["grid_hw"]

        if print_tokens:
            self.print_prompt_tokens(input_ids)

        tokens = self.tokenizer.convert_ids_to_tokens(input_ids[0].tolist())

        word_token_indices = {}
        for w in target_words:
            idxs = self.find_word_token_indices(input_ids, w, verbose=True)
            word_token_indices[w] = idxs

        total_layers = len(self._get_decoder_layers(self.model))
        layer_ids = list(range(total_layers - self.n_layers, total_layers))

        maps = {}
        for w in target_words:
            idxs = word_token_indices[w]
            if idxs is None:
                maps[w] = None
                continue

            per_layer = []
            for li in range(self.n_layers):
                grid, flat = self.get_word_grounding_map(
                    word_token_indices=idxs,
                    image_positions=image_positions,
                    grid_hw=grid_hw,
                    layer_idx=li,
                    head_reduce=head_reduce,
                    subtoken_reduce=subtoken_reduce,
                    normalize=normalize,
                    normalize_scope=normalize_scope,
                    return_probs=return_probs,
                )
                per_layer.append(grid)

            maps[w] = torch.stack(per_layer, dim=0)  # [L, Hp, Wp]

        return {
            "meta": meta,
            "tokens": tokens,
            "layer_ids": layer_ids,
            "word_token_indices": word_token_indices,
            "maps": maps,
        }


class AttributionGeneratorEvolution(AttributionGenerator):
    """
    Extension of AttributionGenerator for token-evolution analysis.

    New objective:
        For a chosen prompt token p (e.g. 'car'), analyze layer-wise update:
            1) delta_norm: || z_out^k[p] - z_in^k[p] ||^2
            2) attn_norm : || o^k[p] ||^2

    Then run Grad-ELLM-style attribution on this scalar objective.
    """

    def __init__(self, *args, **kwargs):
        self.layer_inputs = None
        self.layer_outputs = None
        super().__init__(*args, **kwargs)

    # ---------------------------------------------------------
    # hook whole decoder layer input / output
    # ---------------------------------------------------------
    def _register_hooks(self, layers):
        self.layer_inputs = [[] for _ in range(self.n_layers)]
        self.layer_outputs = [[] for _ in range(self.n_layers)]

        # keep all original hooks from parent
        super()._register_hooks(layers)

        for layer_idx, layer in enumerate(layers):
            def make_layer_in_hook(idx):
                def hook(module, args, kwargs):
                    hidden_states = kwargs.get("hidden_states", None)
                    if hidden_states is None and len(args) > 0:
                        hidden_states = args[0]
                    if hidden_states is None:
                        return

                    if len(self.layer_inputs[idx]) > 0:
                        self.layer_inputs[idx].pop()
                    self.layer_inputs[idx].append(hidden_states)   # keep graph
                return hook

            def make_layer_out_hook(idx):
                def hook(module, inp, out):
                    hidden_out = out[0] if isinstance(out, (tuple, list)) else out
                    if len(self.layer_outputs[idx]) > 0:
                        self.layer_outputs[idx].pop()
                    self.layer_outputs[idx].append(hidden_out)     # keep graph
                return hook

            self.hooks.append(
                layer.register_forward_pre_hook(
                    make_layer_in_hook(layer_idx),
                    with_kwargs=True,
                )
            )
            self.hooks.append(
                layer.register_forward_hook(
                    make_layer_out_hook(layer_idx)
                )
            )

    def _reset_buffers(self):
        super()._reset_buffers()
        if self.layer_inputs is not None:
            for i in range(self.n_layers):
                self.layer_inputs[i].clear()
        if self.layer_outputs is not None:
            for i in range(self.n_layers):
                self.layer_outputs[i].clear()

    # ---------------------------------------------------------
    # helpers
    # ---------------------------------------------------------
    def _resolve_target_positions(
        self,
        input_ids: torch.Tensor,
        target_word: Optional[str] = None,
        target_token_indices: Optional[List[int]] = None,
        verbose: bool = True,
    ) -> List[int]:
        """
        Resolve which prompt token(s) will be used as query positions.
        """
        if target_token_indices is not None:
            return list(target_token_indices)

        if target_word is None:
            raise ValueError("Either target_word or target_token_indices must be provided.")

        pos = self.find_word_token_indices(input_ids, target_word, verbose=verbose)
        if pos is None or len(pos) == 0:
            raise ValueError(f"Cannot find token span for target_word={target_word!r}")
        return pos

    def _build_source_positions(
        self,
        input_ids: torch.Tensor,
        source_mode: str,
        img_info: Optional[Dict[str, Any]] = None,
    ) -> Optional[torch.Tensor]:
        """
        source_mode:
            - 'image': only image tokens
            - 'text' : only text tokens
            - 'all'  : all visible prefix tokens
        """
        S = input_ids.shape[1]
        all_pos = torch.arange(S, device=input_ids.device)

        if source_mode == "all":
            return None

        if source_mode == "image":
            if img_info is None:
                raise ValueError("source_mode='image' but no image tokens found.")
            return img_info["positions"].to(input_ids.device)

        if source_mode == "text":
            if img_info is None:
                return all_pos
            mask = torch.ones(S, dtype=torch.bool, device=input_ids.device)
            mask[img_info["positions"].to(input_ids.device)] = False
            return all_pos[mask]

        raise ValueError("source_mode must be 'image' | 'text' | 'all'")

    def _get_layer_objective(
        self,
        layer_idx: int,
        token_idx: int,
        objective_mode: str = "attn_norm",
    ) -> torch.Tensor:
        """
        Returns scalar objective for one chosen token at one chosen layer.

        objective_mode:
            - 'attn_norm' : || o_k[p] ||^2
            - 'delta_norm': || z_out_k[p] - z_in_k[p] ||^2
        """
        if objective_mode == "attn_norm":
            attn_out = self.attention_outputs[layer_idx][0]   # [B, S, H]
            vec = attn_out[:, token_idx, :]                   # [B, H]
            obj = (vec ** 2).sum(dim=-1)                     # [B]
            return obj

        if objective_mode == "delta_norm":
            z_in = self.layer_inputs[layer_idx][0]           # [B, S, H]
            z_out = self.layer_outputs[layer_idx][0]         # [B, S, H]
            delta = z_out[:, token_idx, :] - z_in[:, token_idx, :]
            obj = (delta ** 2).sum(dim=-1)                   # [B]
            return obj

        raise ValueError("objective_mode must be 'attn_norm' or 'delta_norm'")

    def _compute_one_evolution_map(
        self,
        layer_idx: int,
        query_idx: int,
        selected_positions: Optional[torch.Tensor],
        objective_mode: str = "attn_norm",
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Compute one Grad-ELLM-style attribution map for:
            chosen layer_idx + chosen query token query_idx
        """
        objective = self._get_layer_objective(
            layer_idx=layer_idx,
            token_idx=query_idx,
            objective_mode=objective_mode,
        )   # [B]

        attn_out = self.attention_outputs[layer_idx][0]      # [B, S, H]
        grad = torch.autograd.grad(
            objective.sum(),
            attn_out,
            retain_graph=True,
            create_graph=False,
        )[0]                                                 # [B, S, H]

        grad_query = grad[:, query_idx, :].detach()          # [B, H]

        lam = self._get_qk_scores_for_query(
            layer_idx=layer_idx,
            query_idx=query_idx,
            selected_positions=selected_positions,
            return_probs=False,
            head_reduce="mean",
            normalize=None,
            normalize_scope="selected" if selected_positions is not None else "visible_all",
        )                                                    # [B, K] or [B, S]

        v = self._get_repeated_v_flat(layer_idx)             # [B, S, H]
        if selected_positions is not None:
            selected_positions = selected_positions.to(v.device)
            v = v.index_select(dim=1, index=selected_positions)   # [B, K, H]

        emap = self._grad_eclip_from_lambda(v, grad_query, lam)   # [B, K] or [B, S]
        return emap, objective.detach(), lam.detach()

    # ---------------------------------------------------------
    # main API: prompt-only VQA evolution
    # ---------------------------------------------------------
    def analyze_token_evolution_vqa(
        self,
        image: Image.Image,
        question: str,
        target_word: Optional[str] = None,
        target_token_indices: Optional[List[int]] = None,
        objective_mode: str = "attn_norm",
        source_mode: str = "image",
        token_reduce: str = "mean",   # how to merge multi-subtoken target span
        verbose: bool = True,
    ) -> Dict[str, Any]:
        """
        Prompt-only analysis for a chosen token (e.g. 'car').

        Returns per-layer attribution maps that explain:
            "which source tokens cause the largest update of this target token"
        """
        self.model.eval()

        inputs = self._build_vqa_inputs(image, question)
        input_ids = inputs["input_ids"]
        attention_mask = inputs.get("attention_mask", torch.ones_like(input_ids))

        self._reset_buffers()

        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            pixel_values=inputs["pixel_values"],
            use_cache=False,
            output_attentions=False,
            return_dict=True,
        )

        # just to keep reference; not directly used
        _ = outputs.logits if hasattr(outputs, "logits") else None

        img_info = self.infer_image_positions(input_ids)
        Hp = Wp = None
        if img_info is not None:
            Hp, Wp = self.infer_patch_grid_hw(inputs, img_info["K"])

        target_positions = self._resolve_target_positions(
            input_ids=input_ids,
            target_word=target_word,
            target_token_indices=target_token_indices,
            verbose=verbose,
        )

        selected_positions = self._build_source_positions(
            input_ids=input_ids,
            source_mode=source_mode,
            img_info=img_info,
        )

        per_layer_maps = []
        per_layer_objectives = []
        per_layer_lams = []

        for layer_idx in range(self.n_layers):
            q_maps = []
            q_objs = []
            q_lams = []

            for qidx in target_positions:
                emap, obj, lam = self._compute_one_evolution_map(
                    layer_idx=layer_idx,
                    query_idx=qidx,
                    selected_positions=selected_positions,
                    objective_mode=objective_mode,
                )
                q_maps.append(emap[0].detach().cpu().float())   # [K] or [S]
                q_objs.append(obj[0].detach().cpu().float())    # scalar
                q_lams.append(lam[0].detach().cpu().float())    # [K] or [S]

            q_maps = torch.stack(q_maps, dim=0)                 # [Q, K]
            q_objs = torch.stack(q_objs, dim=0)                 # [Q]
            q_lams = torch.stack(q_lams, dim=0)                 # [Q, K]

            if token_reduce == "mean":
                layer_map = q_maps.mean(dim=0)
                layer_obj = q_objs.mean(dim=0)
                layer_lam = q_lams.mean(dim=0)
            elif token_reduce == "sum":
                layer_map = q_maps.sum(dim=0)
                layer_obj = q_objs.sum(dim=0)
                layer_lam = q_lams.sum(dim=0)
            elif token_reduce == "max":
                layer_map = q_maps.max(dim=0).values
                layer_obj = q_objs.max(dim=0).values
                layer_lam = q_lams.max(dim=0).values
            else:
                raise ValueError("token_reduce must be 'mean' | 'sum' | 'max'")

            per_layer_maps.append(layer_map)
            per_layer_objectives.append(layer_obj)
            per_layer_lams.append(layer_lam)

        per_layer_maps = torch.stack(per_layer_maps, dim=0)             # [L, K] or [L, S]
        per_layer_objectives = torch.stack(per_layer_objectives, dim=0) # [L]
        per_layer_lams = torch.stack(per_layer_lams, dim=0)             # [L, K] or [L, S]

        result = {
            "input_ids": input_ids.detach().cpu(),
            "tokens": self.tokenizer.convert_ids_to_tokens(input_ids[0].tolist()),
            "target_positions": target_positions,
            "objective_mode": objective_mode,
            "source_mode": source_mode,
            "token_reduce": token_reduce,
            "per_layer_maps": per_layer_maps,               # [L, K] or [L, S]
            "per_layer_objectives": per_layer_objectives,   # [L]
            "per_layer_lams": per_layer_lams,               # [L, K] or [L, S]
            "img_info": None if img_info is None else {
                "positions": img_info["positions"].detach().cpu(),
                "spans": img_info["spans"],
                "K": img_info["K"],
            },
            "grid_hw": (Hp, Wp) if Hp is not None else None,
        }

        if source_mode == "image":
            if img_info is None:
                raise RuntimeError("Expected image tokens but img_info is None.")
            K = img_info["K"]
            if Hp is None or Wp is None or Hp * Wp != K:
                raise RuntimeError(f"Invalid grid_hw={(Hp, Wp)} for K={K}")
            result["per_layer_image_maps"] = per_layer_maps.reshape(self.n_layers, Hp, Wp)

        self._reset_buffers()
        return result

    def _find_word_token_indices_in_range(
        self,
        input_ids: torch.Tensor,
        word: str,
        start_pos: int,
        end_pos: Optional[int] = None,
        verbose: bool = True,
    ) -> Optional[List[int]]:
        """
        在 input_ids 的 [start_pos, end_pos) 范围内找 word 对应的 token span。
        用于只在 generated answer 部分匹配，而不是误匹配到 prompt 里同名词。
        """
        full_ids = input_ids[0].tolist()
        n = len(full_ids)
        if end_pos is None:
            end_pos = n

        variants = [
            word,
            " " + word,
            word.lower(),
            " " + word.lower(),
            word.capitalize(),
            " " + word.capitalize(),
            word.upper(),
            " " + word.upper(),
        ]

        matches = []
        for v in variants:
            sub_ids = self.tokenizer.encode(v, add_special_tokens=False)
            m = len(sub_ids)
            if m == 0:
                continue

            for s in range(start_pos, end_pos - m + 1):
                if full_ids[s:s + m] == sub_ids:
                    matches.append((s, s + m, v, sub_ids))

        uniq = []
        seen = set()
        for item in matches:
            key = (item[0], item[1])
            if key not in seen:
                uniq.append(item)
                seen.add(key)

        if len(uniq) == 0:
            if verbose:
                print(
                    f"[WARN] cannot find generated word={word!r} in range "
                    f"[{start_pos}, {end_pos})"
                )
                self.print_prompt_tokens(input_ids)
            return None

        uniq = sorted(uniq, key=lambda x: (x[1] - x[0], x[0]))
        s, e, variant, _ = uniq[0]

        if verbose:
            toks = self.tokenizer.convert_ids_to_tokens(full_ids[s:e])
            print(
                f"[INFO] generated word={word!r}, matched variant={variant!r}, "
                f"token span=({s},{e}), tokens={toks}"
            )

        return list(range(s, e))


    def _exclude_positions(
        self,
        selected_positions: Optional[torch.Tensor],
        total_len: int,
        exclude_positions: List[int],
        device: torch.device,
    ) -> Optional[torch.Tensor]:
        """
        从 selected_positions 中排除一些位置；如果 selected_positions=None，
        则表示原本是 all，这里返回 'all except exclude_positions'。
        """
        exclude_set = set(int(x) for x in exclude_positions)

        if selected_positions is None:
            keep = [i for i in range(total_len) if i not in exclude_set]
            return torch.tensor(keep, device=device, dtype=torch.long)

        keep = [int(x) for x in selected_positions.tolist() if int(x) not in exclude_set]
        return torch.tensor(keep, device=device, dtype=torch.long)


    def _prepare_vqa_generation_and_seq(
        self,
        image: Image.Image,
        question: str,
        seq: Optional[torch.Tensor] = None,
        max_new_tokens: int = 64,
        temperature: float = 0.7,
        do_sample: bool = True,
        top_p: float = 1.0,
        use_cache_generate: bool = True,
    ):
        """
        统一准备：
        - prompt inputs
        - generated seq
        - prompt_len
        - static_modal_inputs
        """
        inputs = self._build_vqa_inputs(image, question)
        input_ids = inputs["input_ids"]
        attn_mask = inputs.get("attention_mask", torch.ones_like(input_ids))
        prompt_len = input_ids.shape[1]

        if seq is None:
            with torch.no_grad():
                gen = self.model.generate(
                    **inputs,
                    do_sample=do_sample,
                    temperature=temperature,
                    top_p=top_p,
                    max_new_tokens=max_new_tokens,
                    use_cache=use_cache_generate,
                    pad_token_id=self.tokenizer.eos_token_id,
                    return_dict_in_generate=True,
                )
            seq = gen.sequences.to(self.device)
        else:
            seq = seq.to(self.device)

        static_modal_inputs = {
            k: v for k, v in inputs.items()
            if k not in ["input_ids", "attention_mask"]
        }

        return inputs, seq, prompt_len, attn_mask, static_modal_inputs


    def analyze_generated_token_evolution_vqa(
        self,
        image: Image.Image,
        question: str,
        answer_word: Optional[str] = None,
        generated_token_offset: Optional[int] = None,
        target_token_indices: Optional[List[int]] = None,   # absolute indices in replay prefix
        seq: Optional[torch.Tensor] = None,                 # if already generated, pass it to avoid regenerate
        replay_upto_offset: Optional[int] = None,           # generated side offset, inclusive
        objective_mode: str = "attn_norm",
        source_mode: str = "image",                         # "image" | "text" | "all"
        token_reduce: str = "mean",
        exclude_self_from_sources: bool = True,
        verbose: bool = True,
        max_new_tokens: int = 64,
        temperature: float = 0.7,
        do_sample: bool = True,
        top_p: float = 1.0,
        use_cache_generate: bool = True,
    ) -> Dict[str, Any]:
        """
        Analyze representation evolution of a generated answer token AFTER it has been fed back.

        例子：
        - answer_word="yellow"
        - generated_token_offset=0   表示第一个生成 token
        - 如果模型答案是单词 yellow，且下一步生成 eos，
            那这里会构造 prefix = [prompt + yellow]
            然后分析 yellow 在各层的 update 是由哪些 source 引起的。

        关键点：
        - 这里不是解释“为什么选 yellow”
        - 而是解释“yellow 被喂回模型后，它自己的 embedding 在各层如何被更新”
        """
        self.model.eval()

        inputs, seq, prompt_len, attn_mask, static_modal_inputs = self._prepare_vqa_generation_and_seq(
            image=image,
            question=question,
            seq=seq,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            do_sample=do_sample,
            top_p=top_p,
            use_cache_generate=use_cache_generate,
        )

        input_ids = inputs["input_ids"]
        full_generated_len = seq.shape[1] - prompt_len
        if full_generated_len <= 0:
            raise RuntimeError("No generated tokens found.")

        # ---------------------------------------------------------
        # 1) 先确定 target token 在 generated 部分的位置
        # ---------------------------------------------------------
        if target_token_indices is not None:
            abs_target_positions = list(target_token_indices)
            if len(abs_target_positions) == 0:
                raise ValueError("target_token_indices is empty.")
            inferred_last_offset = max(abs_target_positions) - prompt_len

        elif generated_token_offset is not None:
            if generated_token_offset < 0 or generated_token_offset >= full_generated_len:
                raise ValueError(
                    f"generated_token_offset={generated_token_offset} out of range, "
                    f"generated length={full_generated_len}"
                )
            abs_target_positions = [prompt_len + generated_token_offset]
            inferred_last_offset = generated_token_offset

        elif answer_word is not None:
            # 先在完整 seq 里，只在 generated 部分搜索
            full_seq_cpu = seq.detach().cpu()
            pos = self._find_word_token_indices_in_range(
                input_ids=full_seq_cpu,
                word=answer_word,
                start_pos=prompt_len,
                end_pos=seq.shape[1],
                verbose=verbose,
            )
            if pos is None or len(pos) == 0:
                raise ValueError(f"Cannot find answer_word={answer_word!r} in generated segment.")
            abs_target_positions = pos
            inferred_last_offset = max(abs_target_positions) - prompt_len

        else:
            raise ValueError(
                "You must provide one of: answer_word / generated_token_offset / target_token_indices"
            )

        # ---------------------------------------------------------
        # 2) 决定 replay prefix 到哪里为止
        #    默认至少要包含目标 answer token 本身
        # ---------------------------------------------------------
        if replay_upto_offset is None:
            replay_upto_offset = inferred_last_offset

        if replay_upto_offset < inferred_last_offset:
            raise ValueError(
                f"replay_upto_offset={replay_upto_offset} must be >= target last offset={inferred_last_offset}"
            )

        if replay_upto_offset >= full_generated_len:
            replay_upto_offset = full_generated_len - 1

        prefix_len = prompt_len + replay_upto_offset + 1
        prefix = seq[:, :prefix_len]

        prefix_mask = torch.cat(
            [
                attn_mask,
                torch.ones(
                    (attn_mask.shape[0], prefix_len - prompt_len),
                    device=self.device,
                    dtype=attn_mask.dtype,
                ),
            ],
            dim=1,
        )

        # 现在 target 必须落在当前 prefix 内
        abs_target_positions = [p for p in abs_target_positions if p < prefix_len]
        if len(abs_target_positions) == 0:
            raise RuntimeError("Target answer token is not included in the replay prefix.")

        # ---------------------------------------------------------
        # 3) 对 replay prefix 做一次 forward，拿 hooks
        # ---------------------------------------------------------
        self._reset_buffers()

        outputs = self.model(
            input_ids=prefix,
            attention_mask=prefix_mask,
            use_cache=False,
            output_attentions=False,
            return_dict=True,
            **static_modal_inputs,
        )

        _ = outputs.logits if hasattr(outputs, "logits") else None

        # image positions / grid
        img_info = self.infer_image_positions(prefix)
        Hp = Wp = None
        if img_info is not None:
            Hp, Wp = self.infer_patch_grid_hw(inputs, img_info["K"])

        selected_positions = self._build_source_positions(
            input_ids=prefix,
            source_mode=source_mode,
            img_info=img_info,
        )

        if exclude_self_from_sources:
            selected_positions = self._exclude_positions(
                selected_positions=selected_positions,
                total_len=prefix.shape[1],
                exclude_positions=abs_target_positions,
                device=prefix.device,
            )

        # ---------------------------------------------------------
        # 4) 复用你现有的 evolution map 逻辑
        # ---------------------------------------------------------
        per_layer_maps = []
        per_layer_objectives = []
        per_layer_lams = []

        for layer_idx in range(self.n_layers):
            q_maps = []
            q_objs = []
            q_lams = []

            for qidx in abs_target_positions:
                emap, obj, lam = self._compute_one_evolution_map(
                    layer_idx=layer_idx,
                    query_idx=qidx,
                    selected_positions=selected_positions,
                    objective_mode=objective_mode,
                )
                q_maps.append(emap[0].detach().cpu().float())
                q_objs.append(obj[0].detach().cpu().float())
                q_lams.append(lam[0].detach().cpu().float())

            q_maps = torch.stack(q_maps, dim=0)
            q_objs = torch.stack(q_objs, dim=0)
            q_lams = torch.stack(q_lams, dim=0)

            if token_reduce == "mean":
                layer_map = q_maps.mean(dim=0)
                layer_obj = q_objs.mean(dim=0)
                layer_lam = q_lams.mean(dim=0)
            elif token_reduce == "sum":
                layer_map = q_maps.sum(dim=0)
                layer_obj = q_objs.sum(dim=0)
                layer_lam = q_lams.sum(dim=0)
            elif token_reduce == "max":
                layer_map = q_maps.max(dim=0).values
                layer_obj = q_objs.max(dim=0).values
                layer_lam = q_lams.max(dim=0).values
            else:
                raise ValueError("token_reduce must be 'mean' | 'sum' | 'max'")

            per_layer_maps.append(layer_map)
            per_layer_objectives.append(layer_obj)
            per_layer_lams.append(layer_lam)

        per_layer_maps = torch.stack(per_layer_maps, dim=0)             # [L, K] or [L, S]
        per_layer_objectives = torch.stack(per_layer_objectives, dim=0) # [L]
        per_layer_lams = torch.stack(per_layer_lams, dim=0)             # [L, K] or [L, S]

        result = {
            "input_ids": input_ids.detach().cpu(),
            "tokens": self.tokenizer.convert_ids_to_tokens(input_ids[0].tolist()),
            "seq": seq.detach().cpu(),
            "full_tokens": self.tokenizer.convert_ids_to_tokens(seq[0].detach().cpu().tolist()),
            "prompt_len": prompt_len,
            "generated_len": full_generated_len,
            "prefix_ids": prefix.detach().cpu(),
            "prefix_tokens": self.tokenizer.convert_ids_to_tokens(prefix[0].detach().cpu().tolist()),
            "target_positions": abs_target_positions,        # absolute positions in replay prefix
            "target_generated_offsets": [p - prompt_len for p in abs_target_positions],
            "replay_upto_offset": replay_upto_offset,
            "objective_mode": objective_mode,
            "source_mode": source_mode,
            "token_reduce": token_reduce,
            "exclude_self_from_sources": exclude_self_from_sources,
            "per_layer_maps": per_layer_maps,
            "per_layer_objectives": per_layer_objectives,
            "per_layer_lams": per_layer_lams,
            "img_info": None if img_info is None else {
                "positions": img_info["positions"].detach().cpu(),
                "spans": img_info["spans"],
                "K": img_info["K"],
            },
            "grid_hw": (Hp, Wp) if Hp is not None else None,
            "decoded_full": self.tokenizer.decode(seq[0], skip_special_tokens=True),
            "decoded_prefix": self.tokenizer.decode(prefix[0], skip_special_tokens=True),
        }

        if source_mode == "image":
            if img_info is None:
                raise RuntimeError("Expected image tokens but img_info is None.")
            K = img_info["K"]
            if Hp is None or Wp is None or Hp * Wp != K:
                raise RuntimeError(f"Invalid grid_hw={(Hp, Wp)} for K={K}")
            result["per_layer_image_maps"] = per_layer_maps.reshape(self.n_layers, Hp, Wp)

        self._reset_buffers()
        return result