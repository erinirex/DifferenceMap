import ast
import os

import cv2
import numpy as np
import pandas as pd
import torch
from PIL import Image
from mmdet.datasets.api_wrappers import COCO
from torch import nn
from torchvision.models import resnet50
import torchvision.transforms as T


device = "cuda:0" if torch.cuda.is_available() else "cpu"

COCO_ANN_FILE = "../../data/coco/annotations/instances_train2017.json"
INPUT_CSV = "detr_context_influence_all_info_1000img.csv"
OUTPUT_CSV = "detr_context_influence_all_info_1000img_decoder_attn.csv"
IMAGE_ROOT = "../../data"

DET_THRESHOLD = 0.7
MATCH_IOU_THRESHOLD = 0.4
MAX_VALID_IMAGES_PER_PAIR = 1000


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


class DecoderLayerWithSelfAttn(nn.TransformerDecoderLayer):
    """Transformer decoder layer that stores object-query self-attention."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.self_attn_weights = None

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
        tgt2, attn_weights = self.self_attn(
            tgt,
            tgt,
            tgt,
            attn_mask=tgt_mask,
            key_padding_mask=tgt_key_padding_mask,
            need_weights=True,
            average_attn_weights=False,
        )
        self.self_attn_weights = attn_weights.detach()

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

        decoder_layer = DecoderLayerWithSelfAttn(
            d_model=hidden_dim,
            nhead=nheads,
            dim_feedforward=2048,
            dropout=0.1,
            activation="relu",
        )
        self.transformer.decoder = nn.TransformerDecoder(
            decoder_layer,
            num_layers=num_decoder_layers,
        )

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

        encoder_input = pos + 0.1 * h.flatten(2).permute(2, 0, 1)
        encoder_output = self.transformer.encoder(encoder_input)

        query_embed = self.query_pos.unsqueeze(1)
        decoder_output = self.transformer.decoder(query_embed, encoder_output)

        decoder_self_attns = []
        for layer in self.transformer.decoder.layers:
            attn = layer.self_attn_weights
            if attn is not None:
                decoder_self_attns.append(attn[0])

        h = decoder_output.transpose(0, 1)
        return {
            "pred_logits": self.linear_class(h),
            "pred_boxes": self.linear_bbox(h).sigmoid(),
            "decoder_self_attns": decoder_self_attns,
        }


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


transform = T.Compose(
    [
        T.Resize(800),
        T.ToTensor(),
        T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ]
)


def box_cxcywh_to_xyxy(x):
    x_c, y_c, w, h = x.unbind(1)
    return torch.stack(
        [
            x_c - 0.5 * w,
            y_c - 0.5 * h,
            x_c + 0.5 * w,
            y_c + 0.5 * h,
        ],
        dim=1,
    )


def rescale_bboxes(out_bbox, size):
    img_w, img_h = size
    boxes = box_cxcywh_to_xyxy(out_bbox)
    scale = torch.tensor([img_w, img_h, img_w, img_h], dtype=torch.float32, device=device)
    return boxes * scale


def bbox_iou(box1, box2):
    x1 = torch.max(box1[0], box2[0])
    y1 = torch.max(box1[1], box2[1])
    x2 = torch.min(box1[2], box2[2])
    y2 = torch.min(box1[3], box2[3])
    inter = torch.clamp(x2 - x1, min=0) * torch.clamp(y2 - y1, min=0)
    area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
    return inter / (area1 + area2 - inter + 1e-6)


def compute_context_influence(decoder_self_attns, layer_reduce="sum"):
    """Return object-query influence matrix with shape [source_query, target_query]."""
    layer_scores = []
    for attn in decoder_self_attns:
        # attn: [num_heads, target_query, source_query]
        score = attn.mean(dim=0).transpose(0, 1)
        layer_scores.append(score)

    per_layer = torch.stack(layer_scores, dim=0)
    if layer_reduce == "mean":
        return per_layer.mean(dim=0), per_layer
    if layer_reduce == "sum":
        return per_layer.sum(dim=0), per_layer
    return per_layer, per_layer


@torch.no_grad()
def detect(im, model, keep_classes=None, threshold=DET_THRESHOLD):
    if im.mode == "L":
        return None

    img = transform(im).unsqueeze(0).to(device)
    if img.shape[-2] > 1600 or img.shape[-1] > 1600:
        return None

    outputs = model(img)
    probas = outputs["pred_logits"].softmax(-1)[0, :, :-1]
    conf, labels = probas.max(dim=1)

    keep = conf > threshold
    if keep_classes is not None:
        keep_tensor = torch.tensor(keep_classes, device=labels.device)
        keep = keep & torch.isin(labels, keep_tensor)

    query_ids = torch.arange(probas.shape[0], device=device)[keep]
    boxes = rescale_bboxes(outputs["pred_boxes"][0, keep], im.size)
    influence, per_layer_influence = compute_context_influence(outputs["decoder_self_attns"])

    return {
        "scores": conf[keep].detach().cpu(),
        "labels": labels[keep].detach().cpu(),
        "boxes": boxes.detach().cpu(),
        "query_ids": query_ids.detach().cpu(),
        "influence": influence.detach().cpu(),
        "per_layer_influence": per_layer_influence.detach().cpu(),
    }


def get_gt_boxes(coco, img_path, category_names):
    filename = os.path.basename(img_path)
    img_id = int(os.path.splitext(filename)[0])

    all_categories = coco.loadCats(coco.getCatIds())
    name_to_id = {cat["name"]: cat["id"] for cat in all_categories}
    desired_ids = {
        name_to_id[name]
        for name in category_names
        if name in name_to_id
    }

    ann_ids = coco.getAnnIds(imgIds=[img_id])
    anns = coco.loadAnns(ann_ids)

    gt_boxes = []
    gt_names = []
    id_to_name = {v: k for k, v in name_to_id.items()}
    for ann in anns:
        cat_id = ann["category_id"]
        if cat_id not in desired_ids:
            continue
        x, y, w, h = ann["bbox"]
        gt_boxes.append(torch.tensor([x, y, x + w, y + h], dtype=torch.float32))
        gt_names.append(id_to_name[cat_id])

    if not gt_boxes:
        return torch.empty((0, 4)), []
    return torch.stack(gt_boxes), gt_names


def match_detections_to_gt(detections, gt_boxes, gt_names):
    matches = []
    used_det = set()

    for gt_idx, (gt_box, gt_name) in enumerate(zip(gt_boxes, gt_names)):
        best_det_idx = None
        best_iou = 0.0

        for det_idx, (det_box, det_label) in enumerate(
            zip(detections["boxes"], detections["labels"])
        ):
            if det_idx in used_det:
                continue
            if CLASSES[int(det_label)] != gt_name:
                continue

            iou = float(bbox_iou(gt_box, det_box))
            if iou > best_iou and iou >= MATCH_IOU_THRESHOLD:
                best_iou = iou
                best_det_idx = det_idx

        if best_det_idx is not None:
            used_det.add(best_det_idx)
            matches.append(
                {
                    "gt_idx": gt_idx,
                    "det_idx": best_det_idx,
                    "class_name": gt_name,
                    "iou": best_iou,
                }
            )

    return matches


def box_center(box):
    x1, y1, x2, y2 = box.tolist()
    return np.array([(x1 + x2) / 2.0, (y1 + y2) / 2.0])


def choose_closest_pair(matches, detections, cat_a, cat_b):
    a_matches = [m for m in matches if m["class_name"] == cat_a]
    b_matches = [m for m in matches if m["class_name"] == cat_b]
    if not a_matches or not b_matches:
        return None, None

    best_a = None
    best_b = None
    best_dist = float("inf")
    for ma in a_matches:
        center_a = box_center(detections["boxes"][ma["det_idx"]])
        for mb in b_matches:
            center_b = box_center(detections["boxes"][mb["det_idx"]])
            dist = np.linalg.norm(center_a - center_b)
            if dist < best_dist:
                best_dist = dist
                best_a = ma
                best_b = mb

    return best_a, best_b


def add_context_inf(img_path, cat_a, cat_b, model, coco):
    image = cv2.imread(img_path)
    if image is None:
        return None

    with Image.open(img_path) as pil_img:
        pil_img = pil_img.convert("RGB")
        keep_classes = [CLASSES.index(cat_a), CLASSES.index(cat_b)]
        detections = detect(pil_img, model, keep_classes=keep_classes)

    if detections is None or len(detections["boxes"]) < 2:
        return None

    gt_boxes, gt_names = get_gt_boxes(coco, img_path, [cat_a, cat_b])
    if len(gt_boxes) < 2:
        return None

    matches = match_detections_to_gt(detections, gt_boxes, gt_names)
    match_a, match_b = choose_closest_pair(matches, detections, cat_a, cat_b)
    if match_a is None or match_b is None:
        return None

    det_a = match_a["det_idx"]
    det_b = match_b["det_idx"]
    query_a = int(detections["query_ids"][det_a])
    query_b = int(detections["query_ids"][det_b])

    influence = detections["influence"]
    per_layer = detections["per_layer_influence"]

    target_inf_on_deleted = float(influence[query_a, query_b])
    deleted_inf_on_target = float(influence[query_b, query_a])
    per_layer_target_inf_on_deleted = per_layer[:, query_a, query_b].tolist()
    per_layer_deleted_inf_on_target = per_layer[:, query_b, query_a].tolist()

    return {
        "target_inf_on_deleted": target_inf_on_deleted,
        "deleted_inf_on_target": deleted_inf_on_target,
        "per_layer_target_inf_on_deleted": per_layer_target_inf_on_deleted,
        "per_layer_deleted_inf_on_target": per_layer_deleted_inf_on_target,
        "target_query": query_a,
        "deleted_query": query_b,
        "target_box": detections["boxes"][det_a].tolist(),
        "deleted_box": detections["boxes"][det_b].tolist(),
    }


def main():
    coco = COCO(COCO_ANN_FILE)
    model = build_model()
    df = pd.read_csv(INPUT_CSV)

    for index, row in df.iloc[::2].iterrows():
        cat_b = row["deleted"]
        cat_a = row["target"]
        img_paths = ast.literal_eval(row["img_paths"])

        print(f"processing pair: target={cat_a}, deleted={cat_b}")

        valid_img_paths = []
        target_inf_on_deleted_list = []
        deleted_inf_on_target_list = []
        per_image_records = []

        for i, rel_img_path in enumerate(img_paths):
            img_path = os.path.join(IMAGE_ROOT, rel_img_path)
            result = add_context_inf(img_path, cat_a, cat_b, model, coco)
            if result is None:
                continue

            valid_img_paths.append(img_path)
            target_inf_on_deleted_list.append(result["target_inf_on_deleted"])
            deleted_inf_on_target_list.append(result["deleted_inf_on_target"])
            per_image_records.append(
                {
                    "img_path": img_path,
                    **result,
                }
            )

            if len(valid_img_paths) >= MAX_VALID_IMAGES_PER_PAIR:
                break

            if (i + 1) % 100 == 0:
                print(f"  scanned={i + 1}, valid={len(valid_img_paths)}")

        cnt = len(valid_img_paths)
        if cnt > 0:
            target_mean = round(float(np.mean(target_inf_on_deleted_list)), 5)
            deleted_mean = round(float(np.mean(deleted_inf_on_target_list)), 5)
            target_median = float(np.median(target_inf_on_deleted_list))
            deleted_median = float(np.median(deleted_inf_on_target_list))
        else:
            target_mean = 0.0
            deleted_mean = 0.0
            target_median = 0.0
            deleted_median = 0.0

        df.loc[index, "context_inf_valid_cnt"] = cnt
        df.loc[index, "target_inf_on_deleted"] = target_mean
        df.loc[index, "deleted_inf_on_target"] = deleted_mean
        df.loc[index, "target_inf_on_deleted_median"] = target_median
        df.loc[index, "deleted_inf_on_target_median"] = deleted_median
        df.at[index, "decoder_attn_per_image_records"] = repr(per_image_records)

        if index + 1 in df.index:
            df.loc[index + 1, "context_inf_valid_cnt"] = cnt
            df.loc[index + 1, "target_inf_on_deleted"] = deleted_mean
            df.loc[index + 1, "deleted_inf_on_target"] = target_mean
            df.loc[index + 1, "target_inf_on_deleted_median"] = deleted_median
            df.loc[index + 1, "deleted_inf_on_target_median"] = target_median

        df.to_csv(OUTPUT_CSV, index=False)
        print(f"  valid={cnt}")
        print(f"  {cat_a} -> {cat_b}: {target_mean}")
        print(f"  {cat_b} -> {cat_a}: {deleted_mean}")

    print(f"saved to {OUTPUT_CSV}")


if __name__ == "__main__":
    main()
