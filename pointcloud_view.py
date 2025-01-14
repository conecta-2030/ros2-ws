import numpy as np
import json
import open3d as o3d

def load_and_visualize(npy_file, json_file):

    point_cloud = np.load(npy_file)
    o3d_cloud = o3d.geometry.PointCloud()
    o3d_cloud.points = o3d.utility.Vector3dVector(point_cloud[:, :3])
    o3d_cloud.paint_uniform_color([0.5, 0.5, 0.5])
    
    with open(json_file, 'r') as f:
        bboxes = json.load(f)
    
    bbox_objects = []
    for bbox in bboxes:
        center = np.array(bbox["center"])
        size = np.array(bbox["size"])
        rotation = bbox["rotation"]
        
        bbox_obj = o3d.geometry.OrientedBoundingBox()
        bbox_obj.center = center
        bbox_obj.extent = size
        bbox_obj.color = (1, 0, 0)
        
        R = o3d.geometry.OrientedBoundingBox.get_rotation_matrix_from_xyz([0, 0, rotation])
        bbox_obj.rotate(R, center=bbox_obj.center)
        
        bbox_objects.append(bbox_obj)
    
    o3d.visualization.draw_geometries([o3d_cloud, *bbox_objects], 
                                      window_name="Point Cloud with Bounding Boxes",
                                      point_show_normal=False)

if __name__ == "__main__":

    point_cloud_file = "dataset/pointcloud/1736876599.0384912.npy"
    bbox_file = "dataset/pointcloud/1736876599.0384912.json"
    
    load_and_visualize(point_cloud_file, bbox_file)
