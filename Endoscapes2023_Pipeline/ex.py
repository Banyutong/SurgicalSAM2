import yaml
import multiprocessing
from inference import inference
from icecream import ic
from loguru import logger


if __name__ == "__main__":
    multiprocessing.set_start_method("spawn")
    logger.add("logs/ex.log", mode="w")
    with open("config.yaml", "r") as f:
        config = yaml.safe_load(f)

    for experiment in config["config"]:
        print(type(experiment))
        logger.info(f"Running experiment \n{yaml.dump(experiment)} \n")
        p = multiprocessing.Process(target=inference, kwargs=experiment)
        p.start()
        p.join()  # 等待当前子进程完成
