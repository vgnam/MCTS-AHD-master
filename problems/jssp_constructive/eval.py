import numpy as np
from os import path
import sys
import logging
sys.path.insert(0, "../../../")
sys.path.insert(0, "/kaggle/working/MCTS-AHD")
try:
    from gpt import determine_next_operation_v2 as determine_next_operation
except ImportError:
    from gpt import determine_next_operation


def eval_heuristic(processing_times: np.ndarray) -> float:
    """
    Evaluate a constructive heuristic for the Job Shop Scheduling Problem (JSSP).

    Parameters
    ----------
    processing_times : np.ndarray
        Shape (n_jobs, n_machines)

    Returns
    -------
    makespan : float
        Total completion time
    """
    assert processing_times.ndim == 2

    n_jobs, n_machines = processing_times.shape

    machine_status = [0] * n_machines
    job_status = [0] * n_jobs

    all_operations = [
        (j, m, processing_times[j, m])
        for j in range(n_jobs)
        for m in range(n_machines)
    ]

    while all_operations:
        feasible_operations = []
        for j, m, _ in all_operations:
            if job_status[j] <= machine_status[m]:
                feasible_operations.append((j, m, processing_times[j, m]))

        if feasible_operations:
            current_status = {
                "machine_status": machine_status,
                "job_status": job_status
            }
            next_op = determine_next_operation(
                current_status,
                feasible_operations
            )
        else:
            next_op = all_operations[0]

        j, m, p = next_op
        start_time = max(job_status[j], machine_status[m])
        end_time = start_time + p

        job_status[j] = end_time
        machine_status[m] = end_time
        all_operations.remove(next_op)

    return max(job_status)


if __name__ == "__main__":
    print("[*] Running ...")

    # CLI arguments (same style as TSP)
    # problem_size = int(sys.argv[1])   # number of jobs
    # root_dir = sys.argv[2]            # unused, kept for interface compatibility
    # mode = sys.argv[3]                # train / val
    # assert mode in ["train", "val"]

    problem_size = 50
    mode = "train"

    basepath = path.join(path.dirname(__file__), "dataset")

    # Dataset existence check (same logic style)
    if not path.isfile(path.join(basepath, "train50_dataset.npy")):
        from gen_inst import GetData

        print("[*] Generating datasets...")
        generator = GetData(n_instances=64)
        generator.save_dataset("train", 50, 10)
        generator.save_dataset("val", 20, 10)
        generator.save_dataset("val", 50, 10)
        generator.save_dataset("val", 100, 20)

    # -------- TRAIN MODE --------
    if mode == "train":
        dataset_path = path.join(basepath, f"train{problem_size}_dataset.npy")
        data = np.load(dataset_path, allow_pickle=True)

        n_instances = len(data)
        print(f"[*] Dataset loaded: {dataset_path} with {n_instances} instances.")

        objs = []
        for i in range(n_instances):
            processing_times, n_jobs, n_machines = data[i]
            obj = eval_heuristic(processing_times)
            print(f"[*] Instance {i}: {obj}")
            objs.append(obj)

        print("[*] Average:")
        print(np.mean(objs))

    # -------- VALIDATION MODE --------
    else:
        for problem_size in [20, 50, 100]:
            dataset_path = path.join(basepath, f"val{problem_size}_dataset.npy")
            logging.info(f"[*] Evaluating {dataset_path}")

            data = np.load(dataset_path, allow_pickle=True)
            n_instances = len(data)

            objs = []
            for i in range(n_instances):
                processing_times, n_jobs, n_machines = data[i]
                obj = eval_heuristic(processing_times)
                objs.append(obj)

            print(f"[*] Average for {problem_size}: {np.mean(objs)}")

