# # dataset_satellite.py
# import os, glob, re
# import numpy as np
# import torch
# from torch.utils.data import Dataset
# from preprocessing.dataset_generation import combine_datasets
# from PIL import Image

# def _extract_year_from_path(path: str) -> str:
#     base = os.path.basename(path)
#     m = re.search(r"\d{4}", base)
#     if m is None:
#         raise ValueError(f"Could not extract year from: {base}")
#     return m.group(0)

# def _build_tiff_path_from_npy(tgt_npy_path: str,
#                               reach_folder: str,
#                               target_root: str) -> str:
#     year = _extract_year_from_path(tgt_npy_path)
#     tiff_dir = os.path.join(target_root, reach_folder)

#     if not os.path.isdir(tiff_dir):
#         raise FileNotFoundError(f"Target directory not found: {tiff_dir}")

#     tiff_paths = sorted(
#         glob.glob(os.path.join(tiff_dir, f"{year}_*.tif"))
#         + glob.glob(os.path.join(tiff_dir, f"{year}_*.tiff"))
#     )

#     if len(tiff_paths) == 0:
#         raise FileNotFoundError(
#             f"No tiffs found for year {year} in {tiff_dir}"
#         )
#     if len(tiff_paths) > 1:
#         raise ValueError(
#             f"Expected 1 tiff for year {year} in {tiff_dir}, "
#             f"found {len(tiff_paths)}: {tiff_paths}"
#         )

#     return tiff_paths[0]

# def build_samples(train_val_test: str,
#                   dir_folders: str,
#                   target_root: str,
#                   year_target: int = 5,
#                   scaled_classes: bool = True):
#     samples = []
#     for folder in os.listdir(dir_folders):
#         if train_val_test in folder:
#             reach_folder = folder
#             reach_id = int(folder.split("_r", 1)[1])

#             inputs, targets = combine_datasets(
#                 train_val_test,
#                 reach_id,
#                 year_target=year_target,
#                 dir_folders=dir_folders,
#                 scaled_classes=scaled_classes,
#             )

#             for inp_paths, tgt_npy_path in zip(inputs, targets):
#                 tiff_path = _build_tiff_path_from_npy(
#                     tgt_npy_path, reach_folder, target_root
#                 )
#                 samples.append((inp_paths, tiff_path))
#     return samples

# class LazyDataset(Dataset):
#     def __init__(self, samples, dtype=torch.float32):
#         self.samples = samples
#         self.dtype = dtype

#     def __len__(self):
#         return len(self.samples)

#     def __getitem__(self, idx):
#         inp_paths, tiff_path = self.samples[idx]

#         xs = [np.load(p, mmap_mode="r") for p in inp_paths]
#         x_np = np.stack(xs, axis=0)

#         if x_np.ndim != 4:
#             raise ValueError(f"Expected (T,12,H,W), got {x_np.shape}")

#         T, M, H, W = x_np.shape

#         if T == 4:
#             # First 3 years: all frames
#             first_3_years = x_np[:3, :, :, :]   # shape (3, 12, H, W)
#             first_3_years = first_3_years.reshape(-1, H, W)  # (36, H, W)

#             # 4th year: first 3 frames only
#             fourth_year = x_np[3, :3, :, :]     # shape (3, H, W)

#             # Concatenate along the "channel" axis
#             x_np = np.concatenate([first_3_years, fourth_year], axis=0)  # (39, H, W)
#         else:
#             x_np = x_np.reshape(T*M, H, W)

        



#         # x_np = x_np.reshape(T * M, H, W)

#         if not os.path.exists(tiff_path):
#             raise FileNotFoundError(f"Missing TIFF: {tiff_path}")

#         with Image.open(tiff_path) as img:
#             y_np = np.array(img)

#         if y_np.ndim == 3:
#             y_np = y_np[:, :, 0]

#         x = torch.from_numpy(x_np).to(self.dtype)
#         label = torch.from_numpy(y_np).long()
#         valid_mask = (label != 0).float()
#         y_bin = (label == 2).long()

#         return x, y_bin, valid_mask


# dataset_satellite.py
import os, glob, re
import numpy as np
import torch
from torch.utils.data import Dataset
from preprocessing.dataset_generation import combine_datasets
from PIL import Image

def _extract_year_from_path(path: str) -> str:
    base = os.path.basename(path)
    m = re.search(r"\d{4}", base)
    if m is None:
        raise ValueError(f"Could not extract year from: {base}")
    return m.group(0)

def _build_tiff_path_from_npy(tgt_npy_path: str,
                             reach_folder: str,
                             target_root: str) -> str:
    year = _extract_year_from_path(tgt_npy_path)
    tiff_dir = os.path.join(target_root, reach_folder)

    if not os.path.isdir(tiff_dir):
        raise FileNotFoundError(f"Target directory not found: {tiff_dir}")

    tiff_paths = sorted(
        glob.glob(os.path.join(tiff_dir, f"{year}_*.tif")) +
        glob.glob(os.path.join(tiff_dir, f"{year}_*.tiff"))
    )

    if len(tiff_paths) == 0:
        raise FileNotFoundError(f"No tiffs found for year {year} in {tiff_dir}")
    if len(tiff_paths) > 1:
        raise ValueError(
            f"Expected 1 tiff for year {year} in {tiff_dir}, "
            f"found {len(tiff_paths)}: {tiff_paths}"
        )

    return tiff_paths[0]

def build_samples(train_val_test: str,
                  dir_folders: str,
                  target_root: str,
                  year_target: int = 5,
                  scaled_classes: bool = True):
    samples = []
    for folder in os.listdir(dir_folders):
        if train_val_test in folder:
            reach_folder = folder
            reach_id = int(folder.split("_r", 1)[1])

            inputs, targets = combine_datasets(
                train_val_test,
                reach_id,
                year_target=year_target,
                dir_folders=dir_folders,
                scaled_classes=scaled_classes,
            )

            for inp_paths, tgt_npy_path in zip(inputs, targets):
                tiff_path = _build_tiff_path_from_npy(
                    tgt_npy_path, reach_folder, target_root
                )
                samples.append((inp_paths, tiff_path))
    return samples

class LazyDataset(Dataset):
    """
    Returns:
      x: (1, D, H, W) where D = expected_T * months (default 4*12=48)
      y: (H, W) with {0,1,IGNORE}
      valid_mask: (H, W)
    """
    def __init__(self, samples, dtype=torch.float32, expected_T=4, months=12, pad_value=0.0):
        self.samples = samples
        self.dtype = dtype
        self.expected_T = expected_T
        self.months = months
        self.pad_value = pad_value

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        inp_paths, tiff_path = self.samples[idx]

        xs = [np.load(p, mmap_mode="r") for p in inp_paths]
        x_np = np.stack(xs, axis=0)  # (T, 12, H, W)

        if x_np.ndim != 4:
            raise ValueError(f"Expected (T,12,H,W), got {x_np.shape}")

        T, M, H, W = x_np.shape
        if M != self.months:
            raise ValueError(f"Expected M={self.months}, got M={M} in {x_np.shape}")

        # keep most recent expected_T years
        if T > self.expected_T:
            x_np = x_np[-self.expected_T:]
            T = self.expected_T

        # pad missing years at the front
        if T < self.expected_T:
            pad_years = self.expected_T - T
            pad_block = np.full((pad_years, M, H, W), self.pad_value, dtype=x_np.dtype)
            x_np = np.concatenate([pad_block, x_np], axis=0)
            T = self.expected_T

        # last year: keep only first 3 months, pad the rest
        k = 3
        x_np[-1, k:, :, :] = self.pad_value

        # flatten (year,month) -> depth D
        x_depth = x_np.reshape(T * M, H, W)          # D = expected_T*12 (48)
        x = torch.from_numpy(x_depth).to(self.dtype).unsqueeze(0)  # (1,D,H,W)

        # target
        with Image.open(tiff_path) as img:
            y_np = np.array(img)
        if y_np.ndim == 3:
            y_np = y_np[:, :, 0]

        label = torch.from_numpy(y_np).long()
        IGNORE = 255
        y = torch.empty_like(label)
        y[label == 1] = 0
        y[label == 2] = 1
        y[label == 0] = IGNORE
        valid_mask = (y != IGNORE)

        return x, y, valid_mask