from glob import glob
from visualize import visualize

predict_list = glob("output/**/predict.json", recursive=True)

gt_path = "coco_annotations.json"

for predict_json in predict_list:
    prompt_info = predict_json.replace("predict.json", "prompt.pkl")
    visualize(gt_path, predict_json, prompt_info)
