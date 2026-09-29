import argparse
import math
import os

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torch import nn
from torchvision.models import resnet50
import torchvision.transforms as T


device = "cuda:0" if torch.cuda.is_available() else "cpu"


CLASSES = [
    "N/A", "person", "bicycle", "car", "motorcycle", "airplane", "bus",
    "train", "truck", "boat", "traffic light", "fire hydrant", "N/A",
    "stop sign", "parking meter", "bench", "bird", "cat", "dog", "horse",
    "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "N/A",
    "backpack", "umbrella", "N/A", "N/A", "handbag", "tie", "suitcase",
    "frisbee", "skis", "snowboard", "sports ball", "kite", "baseball bat",
    "baseball glove", "skateboard", "surfboard", "tennis racket", "bottle",
    "N/A", "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana",
    "apple", "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza",
    "donut", "cake", "chair", "couch", "potted plant", "bed", "N/A",
    "dining table", "N/A", "N/A", "toilet", "N/A", "tv", "laptop", "mouse",
    "remote", "keyboard", "cell phone", "microwave", "oven", "toaster",
    "sink", "refrigerator", "N/A", "book", "clock", "vase", "scissors",
    "teddy bear", "hair drier", "toothbrush",
]


class DecoderLayerForECLIP(nn.TransformerDecoderLayer):
    """Decoder layer that keeps the tensors needed for object-query ECLIP."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.eclip_record = {}

    def _project_self_attn_qkv(self, tgt):
        embed_dim = self.self_attn.embed_dim
        num_heads = self.self_attn.num_heads
        head_dim = embed_dim // num_heads

        weight = self.self_attn.in_proj_weight
        bias = self.self_attn.in_proj_bias
        q_w, k_w, v_w = weight.chunk(3, dim=0)
        q_b, k_b, v_b = bias.chunk(3, dim=0) if bias is not None else (None, None, None)

        q = F.linear(tgt, q_w, q_b)
        k = F.linear(tgt, k_w, k_b)
        v = F.linear(tgt, v_w, v_b)

        def to_heads(x):
            # [num_queries, batch, hidden] -> [batch, heads, num_queries, head_dim]
            nq, batch, _ = x.shape
            return x.permute(1, 0, 2).contiguous().view(batch, nq, num_heads, head_dim).transpose(1, 2)

        return to_heads(q), to_heads(k), to_heads(v)

    def forward(
        self,
        tgt,
        memory,
        tgt_mask=None,
        memory_mask=None,
        tgt_key_padding_mask=None,
        memory_key_padding_mask=None,
        tgt_is_causal=False,
        memory_is_causal=False,
    ):
        q_heads, k_heads, v_heads = self._project_self_attn_qkv(tgt)

        tgt2, attn_weights = self.self_attn(
            tgt,
            tgt,
            tgt,
            attn_mask=tgt_mask,
            key_padding_mask=tgt_key_padding_mask,
            need_weights=True,
            average_attn_weights=False,
        )

        # tgt2 keeps graph, so gradients from the classification logit can flow here.
        self.eclip_record = {
            "q": q_heads.detach(),
            "k": k_heads.detach(),
            "v": v_heads.detach(),
            "attn": attn_weights.detach(),
            "attn_out": tgt2,
        }

        tgt = tgt + self.dropout1(tgt2)
        tgt = self.norm1(tgt)

        tgt2 = self.multihead_attn(
            tgt,
            memory,
            memory,
            attn_mask=memory_mask,
            key_padding_mask=memory_key_padding_mask,
            need_weights=False,
        )[0]
        tgt = tgt + self.dropout2(tgt2)
        tgt = self.norm2(tgt)

        tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt))))
        tgt = tgt + self.dropout3(tgt2)
        tgt = self.norm3(tgt)
        return tgt


class DETRdemo(nn.Module):
    def __init__(
        self,
        num_classes,
        hidden_dim=256,
        nheads=8,
        num_encoder_layers=6,
        num_decoder_layers=6,
    ):
        super().__init__()
        self.backbone = resnet50()
        del self.backbone.fc

        self.conv = nn.Conv2d(2048, hidden_dim, 1)
        self.transformer = nn.Transformer(
            hidden_dim,
            nheads,
            num_encoder_layers,
            num_decoder_layers,
        )
        decoder_layer = DecoderLayerForECLIP(
            d_model=hidden_dim,
            nhead=nheads,
            dim_feedforward=2048,
            dropout=0.1,
            activation="relu",
        )
        self.transformer.decoder = nn.TransformerDecoder(decoder_layer, num_layers=num_decoder_layers)

        self.linear_class = nn.Linear(hidden_dim, num_classes + 1)
        self.linear_bbox = nn.Linear(hidden_dim, 4)
        self.query_pos = nn.Parameter(torch.rand(100, hidden_dim))
        self.row_embed = nn.Parameter(torch.rand(50, hidden_dim // 2))
        self.col_embed = nn.Parameter(torch.rand(50, hidden_dim // 2))

    def forward(self, inputs):
        x = self.backbone.conv1(inputs)
        x = self.backbone.bn1(x)
        x = self.backbone.relu(x)
        x = self.backbone.maxpool(x)
        x = self.backbone.layer1(x)
        x = self.backbone.layer2(x)
        x = self.backbone.layer3(x)
        x = self.backbone.layer4(x)

        h = self.conv(x)
        H, W = h.shape[-2:]
        pos = torch.cat(
            [
                self.col_embed[:W].unsqueeze(0).repeat(H, 1, 1),
                self.row_embed[:H].unsqueeze(1).repeat(1, W, 1),
            ],
            dim=-1,
        ).flatten(0, 1).unsqueeze(1)

        src = pos + 0.1 * h.flatten(2).permute(2, 0, 1)
        memory = self.transformer.encoder(src)
        query_embed = self.query_pos.unsqueeze(1)
        hs = self.transformer.decoder(query_embed, memory).transpose(0, 1)

        records = [layer.eclip_record for layer in self.transformer.decoder.layers]
        return {
            "pred_logits": self.linear_class(hs),
            "pred_boxes": self.linear_bbox(hs).sigmoid(),
            "decoder_eclip_records": records,
        }


transform = T.Compose(
    [
        T.Resize(800),
        T.ToTensor(),
        T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ]
)


def build_model():
    model = DETRdemo(num_classes=91)
    state_dict = torch.hub.load_state_dict_from_url(
        url="https://dl.fbaipublicfiles.com/detr/detr_demo-da2a99e9.pth",
        map_location="cpu",
        check_hash=True,
    )
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    return model


def box_cxcywh_to_xyxy(x):
    x_c, y_c, w, h = x.unbind(1)
    return torch.stack(
        [x_c - 0.5 * w, y_c - 0.5 * h, x_c + 0.5 * w, y_c + 0.5 * h],
        dim=1,
    )


def rescale_bboxes(out_bbox, size):
    img_w, img_h = size
    boxes = box_cxcywh_to_xyxy(out_bbox)
    scale = torch.tensor([img_w, img_h, img_w, img_h], dtype=torch.float32, device=out_bbox.device)
    return boxes * scale


def detect_with_graph(image, model):
    img = transform(image).unsqueeze(0).to(device)
    return model(img)


def select_target_query(outputs, target_class=None, target_query=None, threshold=0.5):
    logits = outputs["pred_logits"][0]
    probas = logits.softmax(-1)[:, :-1]
    conf, labels = probas.max(dim=1)

    if target_query is not None:
        q = int(target_query)
        return q, int(labels[q]), float(conf[q].detach().cpu())

    keep = conf > threshold
    if target_class is not None:
        cls_idx = CLASSES.index(target_class)
        candidates = torch.where(keep & (labels == cls_idx))[0]
    else:
        candidates = torch.where(keep)[0]

    if candidates.numel() == 0:
        raise RuntimeError("No target detection found. Lower --threshold or choose another --target-class.")

    best = candidates[conf[candidates].argmax()]
    return int(best), int(labels[best]), float(conf[best].detach().cpu())


def compute_eclip_object_influence(outputs, target_query, target_class_idx, normalize=True):
    target_logit = outputs["pred_logits"][0, target_query, target_class_idx]
    records = outputs["decoder_eclip_records"]

    per_layer = []
    for rec in records:
        attn = rec["attn"][0]       # [heads, target_query, source_query]
        v = rec["v"][0]             # [heads, source_query, head_dim]
        attn_out = rec["attn_out"]  # [source_query, batch, hidden]

        grad = torch.autograd.grad(
            target_logit,
            attn_out,
            retain_graph=True,
            allow_unused=False,
        )[0]

        grad_t = grad[target_query, 0]  # [hidden]
        num_heads, num_queries, head_dim = v.shape
        grad_t = grad_t.view(num_heads, head_dim)
        lam = attn[:, target_query, :]  # [heads, source_query]

        contrib = v * grad_t[:, None, :] * lam[:, :, None]
        contrib = F.relu(contrib.sum(dim=-1)).mean(dim=0)  # [source_query]
        contrib[target_query] = 0

        if normalize:
            contrib = contrib / (contrib.max() + 1e-8)

        per_layer.append(contrib)

    per_layer = torch.stack(per_layer, dim=0)
    total = per_layer.mean(dim=0)
    return per_layer.detach(), total.detach()


def get_detections(outputs, image_size, threshold):
    probas = outputs["pred_logits"].softmax(-1)[0, :, :-1]
    conf, labels = probas.max(dim=1)
    keep = conf > threshold
    query_ids = torch.arange(probas.shape[0], device=probas.device)[keep]
    boxes = rescale_bboxes(outputs["pred_boxes"][0, keep], image_size)
    return {
        "query_ids": query_ids.detach().cpu(),
        "boxes": boxes.detach().cpu(),
        "labels": labels[keep].detach().cpu(),
        "scores": conf[keep].detach().cpu(),
    }


def draw_visualization(image_path, detections, influence, target_query, target_label, output_path):
    img = cv2.imread(image_path)
    if img is None:
        raise RuntimeError(f"Could not read image: {image_path}")

    influence = influence.detach().cpu()
    target_color = (0, 0, 255)
    source_color = (255, 180, 0)

    order = torch.argsort(influence[detections["query_ids"]], descending=True)
    for rank_idx in order.tolist():
        q = int(detections["query_ids"][rank_idx])
        box = detections["boxes"][rank_idx].numpy().astype(int)
        label = int(detections["labels"][rank_idx])
        score = float(detections["scores"][rank_idx])
        val = float(influence[q])

        x1, y1, x2, y2 = box.tolist()
        color = target_color if q == target_query else source_color
        thickness = 3 if q == target_query else 2
        cv2.rectangle(img, (x1, y1), (x2, y2), color, thickness)

        if q == target_query:
            text = f"TARGET q{q} {CLASSES[label]} {score:.2f}"
        else:
            text = f"q{q} {CLASSES[label]} infl={val:.3f}"
        cv2.putText(img, text, (x1, max(20, y1 - 7)), cv2.FONT_HERSHEY_SIMPLEX, 0.65, color, 2)

    title = f"ECLIP object influence on target q{target_query} ({CLASSES[target_label]})"
    cv2.putText(img, title, (12, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    cv2.imwrite(output_path, img)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", required=True)
    parser.add_argument("--target-class", default=None, choices=[c for c in CLASSES if c != "N/A"])
    parser.add_argument("--target-query", type=int, default=None)
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--output", default="outputs/detr_eclip_object_influence_vis.jpg")
    args = parser.parse_args()

    model = build_model()
    image = Image.open(args.image).convert("RGB")
    outputs = detect_with_graph(image, model)

    target_query, target_label, target_score = select_target_query(
        outputs,
        target_class=args.target_class,
        target_query=args.target_query,
        threshold=args.threshold,
    )
    per_layer, influence = compute_eclip_object_influence(
        outputs,
        target_query=target_query,
        target_class_idx=target_label,
    )
    detections = get_detections(outputs, image.size, args.threshold)
    draw_visualization(args.image, detections, influence, target_query, target_label, args.output)

    print(f"target query: {target_query}")
    print(f"target class: {CLASSES[target_label]}")
    print(f"target score: {target_score:.4f}")
    print(f"visualization saved to: {args.output}")

    top = torch.argsort(influence, descending=True)[:10]
    print("top source queries:")
    for q in top.tolist():
        print(f"  q{q}: {float(influence[q]):.5f}")

    print("per-layer target-source influence shape:", tuple(per_layer.shape))


if __name__ == "__main__":
    main()
