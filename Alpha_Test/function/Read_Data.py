import open3d as o3d
from pathlib import Path
import numpy as np
from typing import Union
import yaml
#####This .py file is mainly for reading data and trun it into o3d data type#####
###############For any question, please contact zzhang@uni-bonn.de###############
class PointCloudDataReader:
    defalt_root= Path(r"cow_and_lady\processed")

    def __init__(self, base_dir: Union[str, Path]=None):
        self.root = Path(base_dir) if base_dir is not None else self.defalt_root
        self.pcd_path = Path(r"cow_and_lady\processed\point_clouds")
        self.calib_path = self.root.parent/"calibration.yaml"
        self.txt_path = self.root/"gt_poses.txt"


    def read_point_cloud(self):

        pcd_path = sorted((self.pcd_path).glob("*.pcd"))[0]
        pcd = o3d.io.read_point_cloud(str(pcd_path))
        PCD_coordinate = np.asarray(pcd.points)
        return PCD_coordinate



    def get_rotation_yaml(self):

        with open(self.calib_path, 'r') as f:
            doc = yaml.safe_load(f)
        T_V_C = np.array(doc['T_V_C'], dtype=np.float64) 
        R_vc = T_V_C[:3, :3]  
        t_vc = T_V_C[:3,  3]   
        return R_vc,t_vc
    


    def get_rotation_txt(self):
        
        pcd_path = sorted((self.pcd_path).glob("*.pcd"))[0]
        sec,nsec = map(int,pcd_path.stem.split("_"))
        stamp = sec + nsec*1e-9
        rows = np.loadtxt(self.txt_path).reshape(-1,8)
        j = np.abs(rows[:,0]-stamp).argmin()
        tx,ty,tz = rows[j,1:4]
        qx,qy,qz,qw = rows[j,4:8]
        Rotation_Matrix = o3d.geometry.get_rotation_matrix_from_quaternion([qx,qy,qz,qw])
        translation = np.asarray([tx,ty,tz])
        return Rotation_Matrix,translation