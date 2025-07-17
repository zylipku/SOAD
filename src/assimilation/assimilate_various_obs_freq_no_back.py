import itertools
import os
import queue
from multiprocessing import Lock, Queue
from pprint import pp

import rootutils
import torch
import torch.multiprocessing as mp

rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)
from src.assimilation.assimilator import Assimilator  # noqa: E402
from src.assimilation.test_configs import TestCaseConfig  # noqa: E402

SAVE_ROOT = f"ass_results/{os.path.splitext(os.path.basename(__file__))[0]}"


def worker(gpu_id, jobs: Queue, lock):
    while True:
        try:
            with lock:
                if jobs.empty():
                    break
                params = jobs.get_nowait()
                pp(params | {"gpu id": gpu_id})

            config = TestCaseConfig(
                ds_name="qg",
                obs_types=params["obs_type"],
                rand_mask_ratio=params["rand_mask_ratio"] if params["rand_mask_ratio"] <= 1.0 else None,
                subsample_stride=params["rand_mask_ratio"] if params["rand_mask_ratio"] > 1 else None,
                time_stride=params["time_stride"],
                method=params["method"],
                seedno=params["seedno"],
                seq_length=9,
            )
            work = Assimilator(
                config,
                device=torch.device("cuda", int(gpu_id)),
                has_background_prior=False,
                unknown_background_noise=False,
            )
            ass_out = work.run(enable_progress_bar=False)
            evaluation = work.evaluate()
            rmse = evaluation["rmse"]
            print(f"GPU {gpu_id}: RMSE = {rmse}")

            os.makedirs(SAVE_ROOT, exist_ok=True)
            save_path = os.path.join(
                SAVE_ROOT,
                f"{params['obs_type']}_{params['method']}_seedno{params['seedno']}_"
                f"{params['rand_mask_ratio']}_"
                f"{params['time_stride']}.pt",
            )

            torch.save(
                params
                | {
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
    params_to_loop = {
        "obs_type": (["vor2vel"], ["arctan3x"], ["sin3x"]),  # ["arctan3x"], ["sin3x"]
        "method": ("soad",),  # "soad"  # 'soad-full'
        "rand_mask_ratio": (
            1.0,
            0.25,
            0.0625,
            0.01,
            2,
            4,
            10,
        ),
        "seedno": (0, 1, 2, 3, 4),  # 1, 2, 3, 4
        "time_stride": (
            1,
            2,
            4,
            8,
        ),
    }

    try:
        if mp.get_start_method() != "spawn":
            mp.set_start_method("spawn")
    except RuntimeError:
        pass

    jobs = Queue()
    lock = Lock()

    # Generate all combinations and put as dicts into jobs
    keys = list(params_to_loop.keys())
    values = [params_to_loop[k] for k in keys]
    for combination in itertools.product(*values):
        job_dict = dict(zip(keys, combination, strict=True))
        jobs.put(job_dict)

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
