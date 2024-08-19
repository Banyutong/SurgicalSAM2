import pickle
from matplotlib import pyplot as plt
import numpy as np

def sort_dicts_by_field(data, field_name, reverse=False):
    """
    按照指定字段的值对字典列表进行排序。

    参数：
        data: 字典列表
        field_name: 用于排序的字段名称
        reverse: 是否降序排列 (默认为 False，即升序排列)

    返回：
        排序后的新字典列表
    """
    return sorted(data, key=lambda item: item.get(field_name), reverse=reverse)


def plot_scores(result_path, output_path):
    
    with open(result_path, "rb") as f:
        result = pickle.load(f)
        
    for video in result["videos"]:
        frames = video["frames"]
        frames = sort_dicts_by_field(frames, "order_in_video")
        
        y_iou = {cat_id: [] for cat_id in frames[0]["cat_scores"].keys()}
        y_mae = {cat_id: [] for cat_id in frames[0]["cat_scores"].keys()}
        y_dice = {cat_id: [] for cat_id in frames[0]["cat_scores"].keys()}
        avg_iou = []
        avg_mae = []
        avg_dice = []

        x = []
        
        for frame in frames:
            x.append(frame['order_in_video'])
            
            for key, cat_scores in frame['cat_scores'].items():
                y_iou[key].append(cat_scores['iou'])
                y_mae[key].append(cat_scores['mae'])
                y_dice[key].append(cat_scores['dice'])
                
            avg_iou.append(frame['avg_scores']['iou'])
            avg_mae.append(frame['avg_scores']['mae'])
            avg_dice.append(frame['avg_scores']['dice'])
            
            
        fig, axs = plt.subplots(3, 1, figsize=(10, 15))

    # 绘制IOU图
        for cat_id, iou_scores in y_iou.items():
            axs[0].plot(x, iou_scores, label=f'Category {cat_id}', linestyle='--')
        axs[0].plot(x, avg_iou, label='Average', linewidth=2.5)
        axs[0].set_title('IOU Scores')
        axs[0].set_xlabel('Frame Order in Video')
        axs[0].set_ylabel('IOU')
        axs[0].legend()

        # 绘制MAE图
        for cat_id, mae_scores in y_mae.items():
            axs[1].plot(x, mae_scores, label=f'Category {cat_id}', linestyle='--')
        axs[1].plot(x, avg_mae, label='Average', linewidth=2.5)
        axs[1].set_title('MAE Scores')
        axs[1].set_xlabel('Frame Order in Video')
        axs[1].set_ylabel('MAE')
        axs[1].legend()

        # 绘制DICE图
        for cat_id, dice_scores in y_dice.items():
            axs[2].plot(x, dice_scores, label=f'Category {cat_id}', linestyle='--')
        axs[2].plot(x, avg_dice, label='Average', linewidth=2.5)
        axs[2].set_title('DICE Scores')
        axs[2].set_xlabel('Frame Order in Video')
        axs[2].set_ylabel('DICE')
        axs[2].legend()

        # 调整子图之间的间距
        plt.tight_layout()
        
        plt.suptitle(f'video: {video['video_id']} Metrics', fontsize=14)
        
        plt.subplots_adjust(top=0.95)

        plt.savefig(f"{output_path}/plot/eval_video_{video['video_id']}.png")

        plt.close()
