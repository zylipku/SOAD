import itertools
import os
import queue
from multiprocessing import Lock, Queue

import torch
import torch.multiprocessing as mp

from src.assimilation.assimilator import Assimilator
from src.assimilation.test_configs import TestCaseConfig

obs_types = (["vor2vel"],)  # ["arctan3x"], ["sin3x"]
methods = ("soad",)  # "soad"  # 'soad-full'
rand_mask_ratios = (
    1.0,
    # 0.25,
    # 0.0625,
    # 0.01,
    # 2,
    # 4,
    # 10,
)
seednos = (0,)  # 1, 2, 3, 4
time_strides = (1,)  # 2, 4, 8

unknown_background_noise = True
has_background_prior = True

SAVE_ROOT = "results2/ass_test"


def worker(gpu_id, jobs: Queue, lock):
    while True:
        try:
            with lock:
                if jobs.empty():
                    break
                obs_type, method, rand_mask_ratio, time_stride, seedno = jobs.get_nowait()
                print(f"GPU {gpu_id}: {(obs_type, method, rand_mask_ratio, time_stride, seedno)=}")

            config = TestCaseConfig(
                ds_name="qg",
                obs_types=obs_type,
                rand_mask_ratio=rand_mask_ratio if rand_mask_ratio <= 1.0 else None,
                subsample_stride=rand_mask_ratio if rand_mask_ratio > 1 else None,
                time_stride=time_stride,
                method=method,
                seedno=seedno,
                seq_length=9 + 1 if unknown_background_noise else 9,
            )
            work = Assimilator(
                config,
                device=torch.device("cuda", int(gpu_id)),
                has_background_prior=has_background_prior,
                unknown_background_noise=unknown_background_noise,
            )
            ass_out = work.run(enable_progress_bar=False)
            evaluation = work.evaluate()
            rmse = evaluation["rmse"]
            print(f"GPU {gpu_id}: RMSE = {rmse}")

            os.makedirs(SAVE_ROOT, exist_ok=True)
            save_path = os.path.join(SAVE_ROOT, f"{obs_type}_{method}_{rand_mask_ratio}_{time_stride}_{seedno}.pt")

            torch.save(
                {
                    "obs_type": obs_type,
                    "method": method,
                    "rand_mask_ratio": rand_mask_ratio,
                    "seed_no": seedno,
                    "time_stride": time_stride,
                    "config": config,
                    "ass_out": ass_out,
                    "rmse": rmse,
                },
                save_path,
            )

        except queue.Empty:
            break
        except Exception as e:
            raise e


if __name__ == "__main__":
    try:
        if mp.get_start_method() != "spawn":
            mp.set_start_method("spawn")
    except RuntimeError:
        pass

    jobs = Queue()
    lock = Lock()

    for obs_type, method, rand_mask_ratio, time_stride, seedno in itertools.product(
        obs_types, methods, rand_mask_ratios, time_strides, seednos
    ):
        jobs.put((obs_type, method, rand_mask_ratio, time_stride, seedno))

    # Start worker processes for each GPU
    device_idxs = os.environ.get("CUDA_VISIBLE_DEVICES", "0").split(",")
    processes = []
    for gpu_id in device_idxs:
        p = mp.Process(target=worker, args=(gpu_id, jobs, lock))
        p.start()
        processes.append(p)

    # Wait for all processes to finish
    for p in processes:
        p.join()
