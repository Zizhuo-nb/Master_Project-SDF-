#####This .py file is mainly for computing data and trun it into o3d data type#####
###############For any question, please contact zzhang@uni-bonn.de###############
from function.Read_Data import PointCloudDataReader
import numpy as np

class DataProcessor:
    def __init__(self,reader: PointCloudDataReader = None):
        self.reader = PointCloudDataReader() or reader
        self.R_vc,self.t_vc = reader.get_rotation_yaml()
        self.Rotation_Matrix,self.translation = reader.get_rotation_txt()
        self.P_cam = reader.read_point_cloud()



    def cam_coordinate(self):
        O_vicon = self.t_vc
        O_camera_w = self.translation+(self.Rotation_Matrix @ O_vicon)
        return O_camera_w



    def pcd_coordinate(self):
        P_vicon = (self.R_vc @ self.P_cam.T).T + self.t_vc
        P_w = (self.Rotation_Matrix @ P_vicon.T).T + self.translation
        return P_w
    
    def sample_distance(self):
        dir = self.pcd_coordinate() - self.cam_coordinate()
        d = np.linalg.norm(dir,axis=1)
        dirs = dir/d[:,None]
        k = 10
        scale = np.linspace(1e-5,1,k)
        log_scale1 = np.log(scale)
        log_norm = (log_scale1-log_scale1.min())/(log_scale1.max()-log_scale1.min())
        sample_distance_alpha = (log_norm[None,:] * d[:,None])*1.05
        d_reshape = d.reshape(len(d),1)
        combined = np.hstack([sample_distance_alpha,d_reshape])
        sample_distance = np.sort(combined,axis=1)
        return dirs,sample_distance,d
    

    def sample_points_coordinate(self):
        dirs,sample_distance,d= self.sample_distance()
        Q_w = self.cam_coordinate()[None, None, :] \
              + dirs[:, None, :] * sample_distance[:, :, None]  
        Q_w = Q_w.reshape(-1, 3)
        return Q_w