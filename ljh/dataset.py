import os
import pandas as pd
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader


class CustomDataset(Dataset):
    def __init__(self, path_csv, dir_img, dir_lbl, S, B, C):
        super().__init__()
        self.annot = pd.read_csv(path_csv)
        self.dir_img = dir_img
        self.dir_lbl = dir_lbl
        self.S = S
        self.B = B
        self.C = C

    def __len__(self):
        return len(self.annot)

    def __getitem__(self, idx):
        path_lbl = os.path.join(self.dir_img, self.annot.iloc[idx, 1])
        with open(path_lbl, "r") as f:
            bboxes = []
            for lbl in f.readlines():
                cl, x, y, w, h = (float(x) for x in lbl.split("\n"))
                bboxes.append((cl, x, y, w, h))


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
