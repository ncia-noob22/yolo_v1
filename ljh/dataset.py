import os
import torch
from torch.utils.data import Dataset, DataLoader

data_dir = "/data/torch/VOCdetection"


class CustomDataset(Dataset):
    def __init__(self):
        super().__init__()


class CustomDataLoader(DataLoader):
    def __init__(
        self,
        dataset,
        batch_size,
        shuffle,
        sampler,
        batch_sampler,
        num_workers,
        collate_fn,
        pin_memory,
        drop_last,
        timeout,
        worker_init_fn,
        multiprocessing_context,
        generator,
        prefetch_factor,
        persistent_workers,
    ):
        super().__init__(
            dataset,
            batch_size,
            shuffle,
            sampler,
            batch_sampler,
            num_workers,
            collate_fn,
            pin_memory,
            drop_last,
            timeout,
            worker_init_fn,
            multiprocessing_context,
            generator,
            prefetch_factor=prefetch_factor,
            persistent_workers=persistent_workers,
        )
