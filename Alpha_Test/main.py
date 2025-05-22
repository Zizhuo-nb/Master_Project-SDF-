import torch
import open3d as o3d
from function.Read_Data import PointCloudDataReader as Read
import numpy as np
from function.Compute import DataProcessor as Process
from torch.utils.data import TensorDataset, DataLoader

def main():
    reader = Read()
    processor = Process(reader)
    dirs, sample_distance, d = processor.sample_distance()
    Q_w = processor.sample_points_coordinate()
    print("dirs shape:", dirs.shape if hasattr(dirs, "shape") else dirs)
    print("sample_distance shape:", sample_distance.shape)
    print("sample_points_coordinate (Q_w) shape:", Q_w.shape)
    print(type(Q_w))
####################################Dataset###################################
    tau = 0.04
    surf_d = d[:, None]                  # (R,1)
    surf_d = np.repeat(surf_d, 11, axis=1)  # (R,k)

    psdf_2d = (surf_d - sample_distance) / tau
    psdf_2d = np.clip(psdf_2d, -1.0, 1.0)  

    all_pts = Q_w.reshape(-1,3)
    center = all_pts.mean(axis=0)
    scales = (all_pts.max(axis=0) - all_pts.min(axis=0)).max()
    coords_n = (all_pts - center[None, :]) / scales  
    psdf = psdf_2d.reshape(-1)
    print(coords_n.shape)
    print(psdf.shape)
##############################################################################
    pts_tensor  = torch.from_numpy(coords_n).float()   # [M,3]
    psdf_tensor = torch.from_numpy(psdf).float()       # [M]

    dataset = TensorDataset(pts_tensor, psdf_tensor)
    loader  = DataLoader(dataset,
                        batch_size=4096,
                        shuffle=True,
                        num_workers=4)
if __name__ == "__main__":
    main()