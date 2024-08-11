import tempfile
import json
import os
from pycocotools.coco import COCO
from pycocotools import mask as maskUtils
import matplotlib.pyplot as plt
from icecream import ic
import cv2
import numpy as np
from copy import deepcopy
import uuid
import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image


## build the predictor

torch.autocast(device_type="cuda", dtype=torch.bfloat16).__enter__()

if torch.cuda.get_device_properties(0).major >= 8:
    # turn on tfloat32 for Ampere GPUs (https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

from sam2.build_sam import build_sam2_video_predictor

sam2_checkpoint = (
    "/bd_byta6000i0/users/sam2/kyyang/SurgicalSAM2/checkpoints/sam2_hiera_tiny.pt"
)
model_cfg = "sam2_hiera_t.yaml"




## define the helpers


def show_mask(mask, ax, obj_id=None, random_color=True):
    if random_color:
        color = np.concatenate([np.random.random(3), np.array([1])], axis=0)
    else:
        cmap = plt.get_cmap("tab10")
        cmap_idx = 0 if obj_id is None else obj_id
        color = np.array([*cmap(cmap_idx)[:3], 0.6])
    h, w = mask.shape[-2:]
    mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
    ax.imshow(mask_image)


def show_points(coords, labels, ax, marker_size=200):
    pos_points = coords[labels == 1]
    neg_points = coords[labels == 0]
    ax.scatter(
        pos_points[:, 0],
        pos_points[:, 1],
        color="green",
        marker="*",
        s=marker_size,
        edgecolor="white",
        linewidth=1.25,
    )
    ax.scatter(
        neg_points[:, 0],
        neg_points[:, 1],
        color="red",
        marker="*",
        s=marker_size,
        edgecolor="white",
        linewidth=1.25,
    )


def show_box(box, ax):
    x0, y0 = box[0], box[1]
    w, h = box[2] - box[0], box[3] - box[1]
    ax.add_patch(
        plt.Rectangle((x0, y0), w, h, edgecolor="green", facecolor=(0, 0, 0, 0), lw=2)
    )


def getSamplePointsFromMask(mask: np.ndarray) -> list:
    kernel = np.ones((3, 3), np.uint8)  # 可以调整核的大小来控制闭运算程度

    # 对 mask 进行闭运算
    closed_mask = cv2.morphologyEx(mask.astype(np.uint8), cv2.MORPH_CLOSE, kernel)

    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(
        closed_mask.astype(np.uint8)
    )
    center_points = []
    # 遍历每个连通区域
    for i in range(1, num_labels):  # 从 1 开始，因为 0 表示背景
        # 获取中心坐标
        sample_points = []

        center_x = centroids[i, 0]
        center_y = centroids[i, 1]

        sample_points.append([center_x, center_y])
        # 将中心坐标添加到列表中
        center_points.append(sample_points)

    return center_points


def mask_to_bbox(mask):
    """
    Extracts the bounding box from a binary mask.
    """
    pos = np.where(mask)
    if len(pos[0]) == 0:
        return None
    xmin, ymin = np.min(pos[1]), np.min(pos[0])
    xmax, ymax = np.max(pos[1]), np.max(pos[0])
    return [float(xmin), float(ymin), float(xmax - xmin + 1), float(ymax - ymin + 1)]


## load info

with open("/bd_byta6000i0/users/sam2/kyyang/endoscapes_video.json", "r") as f:
    video_info = json.load(f)

##load coco

annotation_file = "/bd_byta6000i0/users/dataset/MedicalImage/Endoscapes2023/raw/train_seg/annotation_coco.json"
coco = COCO(annotation_file)
num_categories = len(coco.cats)


coco_annotations = []
## predict
for video_order in range(len(video_info)):
    
    predictor = build_sam2_video_predictor(model_cfg, sam2_checkpoint)
    plt.close('all')

    with tempfile.TemporaryDirectory() as video_dir:

        for idx, frame in enumerate(video_info[video_order]['frames']):
            frame_name = formatted_number = str(idx).zfill(8)  # 填充到5位宽度
            dst_path = os.path.join(video_dir, f'{frame_name}.jpg')
            src_path = frame['path']
            os.symlink(src_path,dst_path)

        inference_state = predictor.init_state(video_path=video_dir)

        img_ids = coco.getImgIds()
        imgs = coco.loadImgs(img_ids)
        
        for img in imgs:
            ann_ids = coco.getAnnIds(imgIds=img["id"])
            if ann_ids == []:
                continue
            if img["video_id"] == video_info[video_order]["video_id"]:
                first_frame = img
                break
        for idx, frame in enumerate(video_info[video_order]["frames"]):
            if first_frame["file_name"] == frame["file_name"]:
                prompt_frame_id = idx
                break
        ic(first_frame['file_name'])

        ann_ids = coco.getAnnIds(imgIds=first_frame["id"])
        anns = coco.loadAnns(ann_ids)
        ann_count = 0
        all_points = []
        all_masks = np.zeros((first_frame['height'],first_frame['width']))

        for ann in anns:
            mask = coco.annToMask(ann)
            sample_points = getSamplePointsFromMask(mask)
            all_masks[mask==1] = ann["category_id"]

            for reigon_samples in sample_points:

                labels = np.ones(len(reigon_samples))
                points = np.array(reigon_samples)
                ann_obj_id = ann_count * (num_categories + 1) + ann["category_id"]
                # ic(ann_obj_id)
                _, out_obj_ids, out_mask_logits = predictor.add_new_points_or_box(
                    inference_state=inference_state,
                    frame_idx=prompt_frame_id,
                    obj_id=ann_obj_id,
                    points=points,
                    labels=labels,
                )
                ann_count += 1
                all_points.append(points)
                
        plt.figure()
        cmap = plt.get_cmap('tab10')  # 'tab10' 提供 10 种不同的颜色

        for i, points in enumerate(all_points):
            x = points[:, 0]  # 提取所有点的 x 坐标
            y = points[:, 1]  # 提取所有点的 y 坐标
            plt.scatter(x, y, color=cmap(i % 10))  # 绘制点，使用颜色映射自动分配颜色
        plt.imshow(all_masks)
        plt.savefig(f'/bd_byta6000i0/users/sam2/kyyang/output/img/prompt_{first_frame['file_name']}.jpg')

        video_segments = {}
        # video_segments contains the per-frame segmentation results
        for out_frame_idx, out_obj_ids, out_mask_logits in predictor.propagate_in_video(
            inference_state, reverse=True
        ):
            video_segments[out_frame_idx] = {
                out_obj_id: (out_mask_logits[i] > 0.0).cpu().numpy()
                for i, out_obj_id in enumerate(out_obj_ids)
            }
        for out_frame_idx, out_obj_ids, out_mask_logits in predictor.propagate_in_video(
            inference_state
        ):
            video_segments[out_frame_idx] = {
                out_obj_id: (out_mask_logits[i] > 0.0).cpu().numpy()
                for i, out_obj_id in enumerate(out_obj_ids)
            }
            
        ## visualize
        plt.figure(figsize=(6, 4))
        plt.title(f"frame {prompt_frame_id}")
        plt.imshow(Image.open(video_info[video_order]['frames'][prompt_frame_id]['path']))
        for out_obj_id, out_mask in video_segments[prompt_frame_id].items():
            show_mask(out_mask, plt.gca(), obj_id=out_obj_id)
        plt.savefig(f'/bd_byta6000i0/users/sam2/kyyang/output/img/{first_frame['file_name']}.jpg')
        ##

        for frame_id in range(len(video_info[video_order]["frames"])):
            current_frame = video_info[video_order]["frames"]
            if current_frame[frame_id]["id"] == None:
                continue

            merged_mask = {}

            ## merge the mask
            for key, mask in video_segments[frame_id].items():
                remainder = key % (num_categories + 1)
                mask = np.logical_or.reduce(mask, axis=0)
                if remainder not in merged_mask:
                    merged_mask[remainder] = mask
                else:
                    merged_mask[remainder] = np.logical_or(merged_mask[remainder], mask)
            # ic(merged_mask)
            # break
            for key, mask in merged_mask.items():
                rle = maskUtils.encode(np.asfortranarray(mask))
                rle['counts'] = rle["counts"].decode('utf-8')
                annotation = {
                    "id": str(uuid.uuid4()),
                    "image_id": current_frame[frame_id]["id"],
                    "category_id": key,
                    "segmentation": rle,
                    "bbox": mask_to_bbox(mask),
                    "area": int(np.sum(mask)),
                    "iscrowd": 0,
                }
                coco_annotations.append(annotation)
    del predictor
    torch.cuda.empty_cache()
    with open(f"/bd_byta6000i0/users/sam2/kyyang/output/predict/{video_order}_predict_coco.json", "w") as f:
        json.dump(coco_annotations, f, indent=4)

img_ids = coco.getImgIds()
cat_ids = coco.getCatIds()
coco_images = coco.loadImgs(img_ids)
coco_cats = coco.loadCats(cat_ids)
predict_data = {
    "images": coco_images,
    "annotations": coco_annotations,
    "categories": coco_cats,
}

with open("predict_coco.json", "w") as f:
    json.dump(predict_data, f, indent=4)
