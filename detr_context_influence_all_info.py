import os
import torchvision
import numpy as np

from mmcv.transforms import Compose
from mmdet.datasets.api_wrappers import COCO
import matplotlib.pyplot as plt
import matplotlib.patches as patches

import math

from PIL import Image

import torch.nn.functional as F

from PIL import Image
import requests
import cv2

# %config InlineBackend.figure_format = 'retina'

import torch
from torch import nn
from torchvision.models import resnet50
import torchvision.transforms as T



# %config InlineBackend.figure_format = 'retina'


torch.set_grad_enabled(True);

from torch.utils.tensorboard import SummaryWriter

device='cuda:0'
coco = COCO('../../data/coco/annotations/instances_train2017.json')

class DETRdemo(nn.Module):
    """
    Demo DETR implementation.

    Demo implementation of DETR in minimal number of lines, with the
    following differences wrt DETR in the paper:
    * learned positional encoding (instead of sine)
    * positional encoding is passed at input (instead of attention)
    * fc bbox predictor (instead of MLP)
    The model achieves ~40 AP on COCO val5k and runs at ~28 FPS on Tesla V100.
    Only batch size 1 supported.
    """
    def __init__(self, num_classes, hidden_dim=256, nheads=8,
                 num_encoder_layers=6, num_decoder_layers=6):
        super().__init__()
        self.writer = SummaryWriter()
        

        # create ResNet-50 backbone
        self.backbone = resnet50()
        del self.backbone.fc

        # create conversion layer
        self.conv = nn.Conv2d(2048, hidden_dim, 1)

        # create a default PyTorch transformer
        self.transformer = nn.Transformer(
            hidden_dim, nheads, num_encoder_layers, num_decoder_layers)

        # prediction heads, one extra class for predicting non-empty slots
        # note that in baseline DETR linear_bbox layer is 3-layer MLP
        self.linear_class = nn.Linear(hidden_dim, num_classes + 1)
        self.linear_bbox = nn.Linear(hidden_dim, 4)

        # output positional encodings (object queries)
        self.query_pos = nn.Parameter(torch.rand(100, hidden_dim))

        # spatial positional encodings
        # note that in baseline DETR we use sine positional encodings
        self.row_embed = nn.Parameter(torch.rand(50, hidden_dim // 2))
        self.col_embed = nn.Parameter(torch.rand(50, hidden_dim // 2))
        # 变量来存储每层的 q、k、v
        self.qs = []
        self.ks = []
        self.vs = []


    def forward(self, inputs):
        # propagate inputs through ResNet-50 up to avg-pool layer
        x = self.backbone.conv1(inputs)
        x = self.backbone.bn1(x)
        x = self.backbone.relu(x)
        x = self.backbone.maxpool(x)

        x = self.backbone.layer1(x)
        x = self.backbone.layer2(x)
        x = self.backbone.layer3(x)
        x = self.backbone.layer4(x)
        # print('x.shape',x.shape)
        dimens = x.shape
        n,c,h,w=dimens
        # convert from 2048 to 256 feature planes for the transformer
        h = self.conv(x)
        fm_conv=h
        # print('h.shape',h.shape)
        # construct positional encodings
        H, W = h.shape[-2:]
        pos = torch.cat([
            self.col_embed[:W].unsqueeze(0).repeat(H, 1, 1),
            self.row_embed[:H].unsqueeze(1).repeat(1, W, 1),
        ], dim=-1).flatten(0, 1).unsqueeze(1)
        
        # # propagate through the transformer
        # h = self.transformer(pos + 0.1 * h.flatten(2).permute(2, 0, 1),
        #                      self.query_pos.unsqueeze(1)).transpose(0, 1))
        

        # Encoder
        encoder_input = pos + 0.1 * h.flatten(2).permute(2, 0, 1)
        # print('encoder_input.shape',encoder_input.shape)
        
        encoder_input_reshaped = encoder_input.clone().permute(1, 2, 0).view(n, 256, H, W)
        # visualize_feature_maps(encoder_input_reshaped)
        
        encoder_output,qs,ks,vs= self.transformer.encoder(encoder_input.clone())  # Processed through the encoder only
        # print('encoder_output.shape',encoder_output.shape)
        q=qs[5]
        k=ks[5]
        v=vs[5]
        # Decoder
        query_embed = self.query_pos.unsqueeze(1)
        h = self.transformer.decoder(query_embed, encoder_output).transpose(0, 1)  # Processed through the decoder

        # # 计算 q、k、v
        # h_flattened = h.flatten(2).permute(2, 0, 1)
        # for layer in range(6):  # 假设有 6 层编码器
        #     q = self.query_pos.unsqueeze(1)  # 生成当前层的 q
        #     k = pos + 0.1 * h_flattened  # 生成当前层的 k
        #     v = h_flattened  # 生成当前层的 v

        #     # 存储 q、k、v
        #     self.qs.append(q)
        #     self.ks.append(k)
        #     self.vs.append(v)

        #     # Transformer 的 encoder 计算
        #     h_flattened = self.transformer.encoder(h_flattened)

        # decoder_output = self.transformer.decoder(self.query_pos.unsqueeze(1), h_flattened)
        # h = decoder_output.transpose(0, 1)

        return {
            'pred_logits': self.linear_class(h),
            'pred_boxes': self.linear_bbox(h).sigmoid(),
            'conv_features': x,
            'last_layer_q': q,
            'last_layer_k': k,
            'last_layer_v': v,
            'encoder_input': encoder_input,
            'encoder_output': encoder_output,
            'fm_conv':fm_conv,
            'encoder_input_reshaped':encoder_input_reshaped
            # 'encoder_layer6_q': self.qs[-1],  # 返回第六层的 q
            # 'encoder_layer6_k': self.ks[-1],  # 返回第六层的 k
            # 'encoder_layer6_v': self.vs[-1],   # 返回第六层的 v
            # 'dimens': dimens
        }


detr = DETRdemo(num_classes=91)
state_dict = torch.hub.load_state_dict_from_url(
    url='https://dl.fbaipublicfiles.com/detr/detr_demo-da2a99e9.pth',
    map_location='cpu', check_hash=True)
detr.load_state_dict(state_dict)
detr = detr.to(device)

# COCO classes
CLASSES = [
    'N/A', 'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus',
    'train', 'truck', 'boat', 'traffic light', 'fire hydrant', 'N/A',
    'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse',
    'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe', 'N/A', 'backpack',
    'umbrella', 'N/A', 'N/A', 'handbag', 'tie', 'suitcase', 'frisbee', 'skis',
    'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove',
    'skateboard', 'surfboard', 'tennis racket', 'bottle', 'N/A', 'wine glass',
    'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple', 'sandwich',
    'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake',
    'chair', 'couch', 'potted plant', 'bed', 'N/A', 'dining table', 'N/A',
    'N/A', 'toilet', 'N/A', 'tv', 'laptop', 'mouse', 'remote', 'keyboard',
    'cell phone', 'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'N/A',
    'book', 'clock', 'vase', 'scissors', 'teddy bear', 'hair drier',
    'toothbrush'
]

# colors for visualization
COLORS = [[0.000, 0.447, 0.741], [0.850, 0.325, 0.098], [0.929, 0.694, 0.125],
          [0.494, 0.184, 0.556], [0.466, 0.674, 0.188], [0.301, 0.745, 0.933]]

detr.eval();

# standard PyTorch mean-std input image normalization
transform = T.Compose([
    
    T.Resize(800),
    T.ToTensor(),
    T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# for output bounding box post-processing
def box_cxcywh_to_xyxy(x):
    x_c, y_c, w, h = x.unbind(1)
    b = [(x_c - 0.5 * w), (y_c - 0.5 * h),
         (x_c + 0.5 * w), (y_c + 0.5 * h)]
    return torch.stack(b, dim=1)

def rescale_bboxes(out_bbox, size):
    img_w, img_h = size
    b = box_cxcywh_to_xyxy(out_bbox)
    b = b * torch.tensor([img_w, img_h, img_w, img_h], dtype=torch.float32, device=device)
    return b

def detect(im, model, transform):
    if im.mode == 'L':
        print("Skipping grayscale image with 1 channel.")
        return None,None,None,None,None,None,None,None,None,None,None
    # mean-std normalize the input image (batch-size: 1)
    img = transform(im).unsqueeze(0).to(device)

    # demo model only support by default images with aspect ratio between 0.5 and 2
    # if you want to use images with an aspect ratio outside this range
    # rescale your image so that the maximum size is at most 1333 for best results
    if img.shape[-2] > 1600 or img.shape[-1] > 1600:
        print('demo model only supports images up to 1600 pixels on each side')
        print('img.shape[-2]',img.shape[-2])
        print('img.shape[-1]',img.shape[-1])
        return None,None,None,None,None,None,None,None,None,None,None

    # propagate through the model
    outputs = model(img)

    # keep only predictions with 0.7+ confidence
    
    probas = outputs['pred_logits'].softmax(-1)[0, :, :-1]
    q=outputs['last_layer_q']
    k=outputs['last_layer_k']
    v=outputs['last_layer_v']
    encoder_input=outputs['encoder_input']
    encoder_output=outputs['encoder_output']
    fm_conv=outputs['fm_conv']
    encoder_input_reshaped=outputs['encoder_input_reshaped']
    # h_pos_encoded=outputs['h_pos_encoded']
    
    keep = probas.max(-1).values > 0.6

    # convert boxes from [0; 1] to image scales
    bboxes_scaled = rescale_bboxes(outputs['pred_boxes'][0, keep], im.size)
    shape=outputs['conv_features'].shape
    # print(outputs.keys())
    # return probas[keep], bboxes_scaled, outputs['conv_features'],encoder_layer6_q,encoder_layer6_k,encoder_layer6_v,dimens
    return probas[keep], bboxes_scaled, outputs['conv_features'],q,k,v,shape,encoder_input,fm_conv,encoder_input_reshaped, encoder_output


color = {'green':(0,255,0),
        'blue':(255,165,0),
        'dark red':(0,0,139),
        'red':(0, 0, 255),
        'dark slate blue':(139,61,72),
        'aqua':(255,255,0),
        'brown':(42,42,165),
        'deep pink':(147,20,255),
        'fuchisia':(255,0,255),
        'yello':(0,238,238),
        'orange':(0,165,255),
        'saddle brown':(19,69,139),
        'black':(0,0,0),
        'white':(255,255,255)}

colors = ['green', 'blue', 'dark red', 'red', 'dark slate blue', 'aqua', 'brown',\
        'deep pink', 'fuchisia', 'yello', 'orange', 'saddle brown','white']

def draw_boxes(img, boxes, scores=None, tags=None, line_thick=2, line_color='white'):
    width = img.shape[1]
    height = img.shape[0]
    for i in range(len(boxes)):
        tmp_color = color[line_color] if line_color is not None else color[colors[i%len(colors)]]
        one_box = boxes[i]
        one_box = np.array([max(one_box[0], 0), max(one_box[1], 0),
                    min(one_box[2], width - 1), min(one_box[3], height - 1)])
        x1,y1,x2,y2 = np.array(one_box[:4]).astype(int)
        cv2.rectangle(img, (x1,y1), (x2,y2), tmp_color, line_thick)
        if scores is not None:
            text = "{} {:.3f}".format(CLASSES[tags[i]], scores[i])
            cv2.putText(img, text, (x1, y1 - 7), cv2.FONT_ITALIC, 1.0, tmp_color, line_thick)
    return img

def sim_qk(q, k):
    q_norm = F.normalize(q, dim=-1)  # (h*w, batch_size, c)
    k_norm = F.normalize(k, dim=-1)  # (h*w, batch_size, c)
    q_norm = q_norm.permute(1, 0, 2)  # (batch_size, h*w, c)
    k_norm = k_norm.permute(1, 0, 2)  # (batch_size, h*w, c)

    similarity = torch.bmm(q_norm, k_norm.transpose(1, 2)) #(batch_size, h*w, h*w)
    similarity=similarity.squeeze(0)
    sim_max = similarity.max(dim=-1, keepdim=True)[0]
    sim_min = similarity.min(dim=-1, keepdim=True)[0]
    sim_normalized = (similarity - sim_min) / (sim_max - sim_min)
    return sim_normalized

def add_context_inf(img_path, cat_a, cat_b):
    # preprepare data
    ori_img = cv2.imread(img_path)
    filename = os.path.basename(img_path)
    img_id = int(filename.split('.')[0])
    with Image.open(img_path) as img:
        ori_shape = img.size  # (width, height)
    ori_shape = (ori_shape[1], ori_shape[0]) # (height , width)
    # print('ori_shape',ori_shape)
    Ts = T.Resize((int(ori_shape[0]/8), int(ori_shape[1]/8)))
    Tl = T.Resize(ori_shape)
    img = Image.open(img_path)
    # scores, boxes, features = detect(img, detr, transform)

    scores, boxes, features, q, k, v,shape,encoder_input,fm_conv, encoder_input_reshaped, encoder_output = detect(img, detr, transform)
    if scores==None:
        return 0
    # print('features.shape',features.shape)
    # features_activate = F.relu(features.detach().cpu())
    # print('visualize_feature_maps(features_activate)')
    # visualize_feature_maps(features_activate)
    # encoder_input_reshaped_activate= F.relu(encoder_input_reshaped.clone().detach().cpu())
    # print('visualize_feature_maps(encoder_input_reshaped_activate)')
    # visualize_feature_maps(encoder_input_reshaped_activate)

    
    n,c,h,w=shape
    cat_a_index = CLASSES.index(cat_a)
    cat_b_index = CLASSES.index(cat_b)
    # print('cat_a_index',cat_a_index)
    # print('cat_b_index',cat_b_index)
    
    labels = scores.argmax(dim=1)
    keep = scores.max(dim=1).values > 0.7
    
    desired_classes = [cat_b_index, cat_a_index]
    keep_classes = torch.isin(labels, torch.tensor(desired_classes, device=labels.device))
    bboxes = boxes[keep & keep_classes]
    labels = labels[keep & keep_classes]
    scores = scores[keep & keep_classes]
    scores = scores.max(dim=1).values

    v_dt=v.detach().clone().permute(1, 2, 0).view(n, 256, h, w)
    v_activate = F.relu(v_dt.detach().cpu())
    # print('visualize_feature_maps(v_activate)')
    # visualize_feature_maps(v_activate)

    cos_sim_matrix = sim_qk(q,k)
    cos_sim_matrix=cos_sim_matrix.detach().cpu()

    output = torch.matmul(cos_sim_matrix.cpu(), v.detach().clone().cpu().squeeze(1))  
    
    output=output.unsqueeze(1)
    output = output.detach().clone().permute(1, 2, 0).view(n, 256, h, w)
    output_activate = F.relu(output.detach().cpu())
 
    ann_ids = coco.getAnnIds(imgIds=[img_id])
    ann_info = coco.loadAnns(ann_ids)  
    desired_names=[cat_a,cat_b]
    gt_bboxes = []
    category_ids = []
    all_categories = coco.loadCats(coco.getCatIds())
    category_name_to_id = {cat['name']: cat['id'] for cat in all_categories}

    desired_ids = [category_name_to_id[class_name] for class_name in desired_names if class_name in category_name_to_id]

    for ann in ann_info:
        category_id = ann['category_id']
        if category_id in desired_ids:
            bbox = torch.tensor(ann['bbox'])
            gt_bboxes.append(bbox)
            category_ids.append(category_id)

    gt_bboxes = torch.stack(gt_bboxes) if gt_bboxes else torch.empty((0, 4))  
    gt_bboxes[:,2:] = gt_bboxes[:,:2] + gt_bboxes[:,2:]
    category_ids = torch.tensor(category_ids)
    
    category_names = coco.loadCats(category_ids.tolist())  # 根据类别 ID 批量加载类别信息
    category_name_map = {cat['id']: cat['name'] for cat in category_names}  # 创建类别 ID 到名称的映射

    def bbox_iou(box1, box2):
        x1 = torch.max(box1[0], box2[0])
        y1 = torch.max(box1[1], box2[1])
        x2 = torch.min(box1[2], box2[2])
        y2 = torch.min(box1[3], box2[3])
    
        inter_area = torch.clamp(x2 - x1, min=0) * torch.clamp(y2 - y1, min=0)
        box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
        box2_area = (box2[2] - box2[0]) * (box2[3] - box2[1])
    
        iou = inter_area / (box1_area + box2_area - inter_area + 1e-6)
        return iou

    dt_bboxes = bboxes.clone().detach().cpu()
    box_inds = []
    gt_inds=[]
    selected_dt_bboxes = {}

    for gt_idx, (gt_bbox, gt_catid) in enumerate(zip(gt_bboxes, category_ids)):
        gt_cat = category_name_map[gt_catid.item()]
        
        selected_dt_bboxes[gt_idx] = (None, 0)
    
        max_iou = 0
        best_dt_index = None
    
        for idx, (dt_bbox, dt_catid) in enumerate(zip(dt_bboxes, labels.detach().cpu())):
            if CLASSES[dt_catid] == gt_cat:
                iou = bbox_iou(gt_bbox, dt_bbox)
                # print('iou',iou)
                if iou > max_iou and iou > 0.4:
                    max_iou = iou
                    best_dt_index = idx
    
        if best_dt_index is not None:
            selected_dt_bboxes[gt_idx] = (best_dt_index, max_iou)

    cata_dtinds=[]
    catb_dtinds=[]
    cata_gtinds=[]
    catb_gtinds=[]
    
    for i, (dt_ind,iou) in selected_dt_bboxes.items():
        if dt_ind is not None:            
            if CLASSES[labels[dt_ind]]==cat_b:
                catb_dtinds.append(dt_ind)
                catb_gtinds.append(i)

            if CLASSES[labels[dt_ind]]==cat_a:
                cata_dtinds.append(dt_ind)
                cata_gtinds.append(i)

    def bbox_center(bbox):
        x1, y1, x2, y2 = bbox
        return np.array([(x1 + x2) / 2, (y1 + y2) / 2])
    
    def find_closest_pair(cata_inds, catb_inds, dt_bboxes):
        min_distance = float('inf')
        closest_pair = (None, None)
        
        for idx_a in cata_inds:
            bbox_a = dt_bboxes[idx_a].numpy()  # 取出 `cat_a` 的检测框并转为 numpy
            center_a = bbox_center(bbox_a)     # 计算 `cat_a` 检测框的中心点
            
            for idx_b in catb_inds:
                bbox_b = dt_bboxes[idx_b].numpy()  # 取出 `cat_b` 的检测框并转为 numpy
                center_b = bbox_center(bbox_b)     # 计算 `cat_b` 检测框的中心点
                
                distance = np.linalg.norm(center_a - center_b)
                
                if distance < min_distance:
                    min_distance = distance
                    closest_pair = (idx_a, idx_b)
        
        return closest_pair[0],closest_pair[1]

    idx_a, idx_b=find_closest_pair(cata_dtinds, catb_dtinds, dt_bboxes)
    index_in_cata = cata_dtinds.index(idx_a) if idx_a is not None else None
    index_in_catb = catb_dtinds.index(idx_b) if idx_b is not None else None
                  
    
    if index_in_catb is not None:
        box_inds.append(catb_dtinds[index_in_catb])
        gt_inds.append(catb_gtinds[index_in_catb])

    if index_in_cata is not None:
        box_inds.append(cata_dtinds[index_in_cata])
        gt_inds.append(cata_gtinds[index_in_cata])

    ori_l=len(box_inds)
    if ori_l<2:
        print("best covered boxes < 2")
        return 0

    if CLASSES[labels[box_inds[0]]]!=cat_b:
        box_inds[0],box_inds[1]=box_inds[1],box_inds[0]
        gt_inds[0],gt_inds[1]=gt_inds[1],gt_inds[0]

    gt_bboxes=gt_bboxes[gt_inds]
    category_ids=category_ids[gt_inds]
    
    bboxes=bboxes[box_inds]
    labels = labels[box_inds]
    scores = scores[box_inds]

    dt_bboxes=dt_bboxes[box_inds]
    dt_labels = labels.detach().cpu()
    dt_scores = scores.detach().cpu()     

    im_copy=ori_img.copy()

    
    H, W, _ = im_copy.shape
    resize = T.Resize((H, W))

    # odam_maps = torch.zeros((len(box_inds), H, W))
    n,c,h,w=shape
    # grad_maps = torch.zeros((len(box_inds), 256, h, w))
    sum_grad = torch.zeros((h*w, 1, 256))
    grad_maps = torch.zeros((len(box_inds), 256, H, W))
    ori_grad_maps = torch.zeros((len(box_inds), 256, H, W))

    feat_maps = [] # level_maps changed to feat_maps
    grads = []
    final_grads = []


    cos_sim_matrix = sim_qk(q,k)
    cos_sim_matrix=cos_sim_matrix.detach().cpu()
    
    for cls_score, (xmin, ymin, xmax, ymax) in zip(scores, bboxes):
        out1 = xmin
        out2 = ymin
        out3 = -xmax
        out4 = -ymax
        out5 = cls_score
        ### comb
        maps = []
        grads = []
        for out in [out1,out2,out3,out4,out5]:        
            grad = torch.autograd.grad(
                out,
                v,
                retain_graph=True)[0]
            grad=grad.detach().cpu()
            sum_grad+=grad
            # grads.append(grad)
                
        # grads = torch.stack(grads, dim=0)
        # max_grads = torch.max(grads, dim=0)[0].detach().cpu()
        # output = torch.matmul(cos_sim_matrix, max_grads.squeeze(1))   
        output = torch.matmul(cos_sim_matrix, sum_grad.squeeze(1))   
        output=output.unsqueeze(1)
        output = output.detach().clone().permute(1, 2, 0).view(n, 256, h, w)

        final_grads.append(output)
        
    final_grads = torch.cat(final_grads, dim=0)
    final_grads = Tl(final_grads)
    grad_maps += final_grads.detach().cpu()

    show_num=min(5, len(box_inds))

    maps = grad_maps.clone().detach().cpu()
    inds = torch.arange(0, show_num).long()
    _,C,H,W = maps.shape
    
    maps1 = maps[inds]
    maps2 = maps[inds]

    N=len(maps1)

    maps1 = F.relu(maps1.reshape(N,-1).detach().cpu()).detach().cpu()
    maps2 = F.relu(maps2.reshape(N,-1).detach().cpu()).detach().cpu()

    max_value_obj1 = torch.max(maps1[0])  # 第一个物体
    max_value_obj2 = torch.max(maps1[1])  # 第二个物体


    affect = maps1.unsqueeze(1).detach().cpu() * maps2.unsqueeze(0).detach().cpu()

    j_sqr=(maps1.detach().cpu().unsqueeze(1)**2)

    j_sqr=j_sqr.sum(-1)   
    # print('j_sqr.shape',j_sqr.shape)
    print('j_sqr',j_sqr)
    print('j1j2', affect.sum(-1).detach().cpu())
    # print('j1j2.shape', affect.sum(-1).detach().cpu().shape)
    affect = affect.sum(-1).detach().cpu() / j_sqr

    
    print("affect", affect)
    print(f'{cat_a} affect on {cat_b}: {affect[1][0]}')
    print(f'{cat_b} affect on {cat_a}: {affect[0][1]}')

    # fig = plt.figure(figsize=(18, 12))

    
    affects=[]
    for k in range(2):
        # box_rec = dt_bboxes[k][:4]
        # label_red=labels[box_ind]
        # print("category in red box:", CLASSES[label_red])
        # ax = fig.add_subplot(1,3,k+1)
        # ax.axis('off')
        # rec = plt.Rectangle(box_rec[:2],box_rec[2]-box_rec[0],box_rec[3]-box_rec[1], fill=False,edgecolor='red')
        # ax.add_patch(rec)
        for j in range(2):
            if j != k:
                # box_rec = dt_bboxes[j][:4]    
                # rec = plt.Rectangle(box_rec[:2],box_rec[2]-box_rec[0],box_rec[3]-box_rec[1], fill=False,edgecolor='white')
                # ax.add_patch(rec)    
                affects.append('{:.4f}'.format(affect[1-k,1-j]))
    #             plt.annotate('{:.4f}'.format(affect[1-k,1-j]), xy=box_rec[:2], color='red',\
    #                          bbox=dict(boxstyle='round,pad=0.2', fc='white', alpha=0.5),fontsize=18)
                
        
    #     plt.imshow(ori_img[:,:,::-1]) 
        
    # plt.show()
    return affects
    
    
    
    
import pandas as pd
import ast
import csv
from scipy import stats

df=pd.read_csv('detr_context_influence_all_info_1000img.csv')

cat_a_values = df['target'].values
cat_b_values = df['deleted'].values
img_paths_tot = df['img_paths'].apply(ast.literal_eval)

# img_paths_tot = df_inf['valid_img_paths'].apply(ast.literal_eval)


cnts=[]
target_inf_on_deleted=[]
deleted_inf_on_target=[]

for index, row in df.iloc[::2].iterrows():

    inf=row['deleted_inf_on_target']

    cat_b = row['deleted']
    cat_a = row['target']
    prob = row['P(deleted|target)']

    img_paths=ast.literal_eval(row['img_paths'])
    


    print('cat_a',cat_a)
    print('cat_b',cat_b)

    cnt=0
    affect_lst=[]
    valid_img_paths=[]
    for i in range(min(20000, len(img_paths))):
        print(i)
        img_path=img_paths[i]
        img_path="../../data/"+img_path
        ori_img=cv2.imread(img_path)
        if ori_img is None:
            print(f"Failed to load image at {img_path}")
            continue
        affects=add_context_inf(img_path,cat_a,cat_b)
        if affects==0:
            print("error when computing affects")
            continue
        if affects != []:
            affect_lst.append(affects)
            cnt+=1
        else:
            continue
        valid_img_paths.append(img_path)
        if cnt>999:
            break
    print(cnt)
    cnts.append(cnt)
    print(affect_lst)
    if affect_lst:
        affect_lst = [[float(x) for x in sublist] for sublist in affect_lst]
        target_inf_on_deleted_list=[x[0] for x in affect_lst]
        deleted_inf_on_target_list=[x[1] for x in affect_lst]
        mean_first = round(sum([x[0] for x in affect_lst]) / len(affect_lst), 5)        
        mean_second = round(sum([x[1] for x in affect_lst]) / len(affect_lst), 5)
        deleted_inf_on_target.append(mean_second)
        target_inf_on_deleted.append(mean_first)
    else:
        mean_first = 0        
        mean_second = 0
        deleted_inf_on_target.append(mean_second)
        target_inf_on_deleted.append(mean_first)
    
    df.loc[index, 'context_inf_valid_cnt']=cnt
    df.loc[index, 'deleted_inf_on_target']=mean_second
    df.loc[index, 'target_inf_on_deleted']=mean_first
    df.loc[index+1, 'deleted_inf_on_target'] = mean_first
    df.loc[index+1, 'target_inf_on_deleted'] = mean_second
    
    # df.at[index, 'deleted_inf_on_target_list'] = deleted_inf_on_target_list
    # df.at[index, 'target_inf_on_deleted_list'] = target_inf_on_deleted_list    
    list1 = np.array(deleted_inf_on_target_list)
    list2 = np.array(target_inf_on_deleted_list)
    deleted_inf_on_target_median=np.median(list1)
    target_inf_on_deleted_median=np.median(list2)
    df.at[index, 'deleted_inf_on_target_median']=deleted_inf_on_target_median
    df.at[index, 'target_inf_on_deleted_median']=target_inf_on_deleted_median
    

    df.loc[index+1, 'deleted_inf_on_target_median'] = target_inf_on_deleted_median
    df.loc[index+1, 'target_inf_on_deleted_median'] = deleted_inf_on_target_median
    
    df.to_csv('detr_context_influence_all_info_1000img.csv', index=False)
    print(f"{cat_a}'s context influence on {cat_b} mean value: {mean_first}")
    print(f"{cat_b}'s context influence on {cat_a} mean value: {mean_second}")


print('cnts',cnts)
