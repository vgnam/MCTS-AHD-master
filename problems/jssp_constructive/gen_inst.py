import os
import numpy as np


class GetData:
    def __init__(self, n_instances: int):
        self.n_instances = n_instances

    def _generate_instances(self, n_jobs: int, n_machines: int):
        instances = []
        for _ in range(self.n_instances):
            processing_times = np.random.randint(
                10, 100, size=(n_jobs, n_machines)
            )
            instances.append(
                (processing_times, n_jobs, n_machines)
            )
        return instances

    def save_dataset(
            self,
            split: str,
            n_jobs: int,
            n_machines: int,
            save_dir: str = "dataset"
    ):
        os.makedirs(save_dir, exist_ok=True)

        file_name = f"{split}{n_jobs}_dataset.npy"
        file_path = os.path.join(save_dir, file_name)

        np.random.seed(2024)
        data = self._generate_instances(n_jobs, n_machines)

        np.save(file_path, np.array(data, dtype=object), allow_pickle=True)
        print(f"Saved {file_path}")


generator = GetData(n_instances=64)

generator.save_dataset("train", 50, 10)
generator.save_dataset("val", 20, 10)
generator.save_dataset("val", 50, 10)
generator.save_dataset("val", 100, 20)
generator.save_dataset("test", 20, 10)
generator.save_dataset("test", 50, 10)
generator.save_dataset("test", 100, 10)
