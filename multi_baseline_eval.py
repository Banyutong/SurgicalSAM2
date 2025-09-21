# run_cli.py
import subprocess, os, concurrent.futures
from baseline_eval import discover_combo_configs  # 父进程随便 import torch 都行


def job(cfg_path, gpu):
    subprocess.run(
        ["python", "baseline_eval.py", "--combo-file", str(cfg_path)],
        env={**os.environ, "CUDA_VISIBLE_DEVICES": str(gpu)},
        check=True,
    )


tasks = discover_combo_configs()
tasks = [task for task in tasks if 'endovis18' in str(task)]
with concurrent.futures.ThreadPoolExecutor(5) as pool:
    futures = [pool.submit(job, cfg, idx % 1) for idx, cfg in enumerate(tasks)]
    for f in concurrent.futures.as_completed(futures):
        f.result()  # 有异常就抛
