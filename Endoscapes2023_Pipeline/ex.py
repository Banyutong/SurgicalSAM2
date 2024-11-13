import yaml
import multiprocessing
from inference import inference


if __name__ == "__main__":
    multiprocessing.set_start_method("spawn")

    with open("configs/dense_points_random.yaml", "r") as f:
        config = yaml.safe_load(f)

    for experiment in config["config"]:
        print(type(experiment))
        p = multiprocessing.Process(target=inference, kwargs=experiment)
        p.start()
        p.join()  # 等待当前子进程完成
