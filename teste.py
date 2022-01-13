import numpy as np
from openni import openni2
from openni import _openni2
import cv2 as cv
import open3d
import copy
import time
from numba import jit, cuda

SAVE_POINTCLOUDS = False

exit_flag=False
width=640,
height=480,
""" old parameters
fx=654.75924295
fy=842.74853381
cx=325.50911362
cy=619.35980012
"""
fx=606.21778264910711
fy=606.21778264910711
cx=499.5
cy=349.5
#news = set()
@jit             
def process_images(color_image, depth_image):
    color_image = np.frombuffer(color_image.get_buffer_as_uint8(), np.uint8)
    #cv.imwrite("Color.png",color_image.reshape(480,640,3))
    color_image = color_image.reshape(480 * 640, 3)
    
    depth_image = np.frombuffer(depth_image.get_buffer_as_uint16(), np.uint16)

    new_points = np.zeros((640*480,3), dtype=np.float64)

    for i in range(480):
        for j in range(640):
            z = depth_image[i*640+j]/10
            x = (j-cx)*z/fx
            y = (i-cy)*z/fy
            new_points[i*640+j, 0] = x
            new_points[i*640+j, 1] = y
            new_points[i*640+j, 2] = z
            #news.add(z)

    return color_image, new_points

    
def exit_key(vis,cena1, cena2):
    #print(news)
    global exit_flag
    exit_flag=True
    
def main():
    global width, height, fx, fy, cx, cy, exit_flag
    # Init openni
    openni_dir = "/home/pedro/OpenNI-Linux-x64-2.3/Redist"
    openni2.initialize(openni_dir)
    copy_textured_mesh = open3d.io.read_triangle_mesh('drum.obj')
    copy_textured_mesh.scale(10,copy_textured_mesh.get_center())
    # Open astra depth stream (using openni)
    depth_device = openni2.Device.open_any()
    color_stream = depth_device.create_color_stream()
    depth_stream = depth_device.create_depth_stream()
    depth_stream.set_video_mode(_openni2.OniVideoMode(pixelFormat = _openni2.OniPixelFormat.ONI_PIXEL_FORMAT_DEPTH_100_UM, resolutionX = 640, resolutionY = 480, fps = 30))
    color_stream.set_video_mode(_openni2.OniVideoMode(pixelFormat = _openni2.OniPixelFormat.ONI_PIXEL_FORMAT_RGB888, resolutionX = 640, resolutionY = 480, fps = 30))
    depth_stream.start()
    color_stream.start()
    
    depth_device.set_image_registration_mode(openni2.IMAGE_REGISTRATION_DEPTH_TO_COLOR)

    visualizer=open3d.visualization.VisualizerWithKeyCallback()
    visualizer.create_window("Pointcloud", width=1000, height=700)

    ctr = visualizer.get_view_control()
    parameters = open3d.io.read_pinhole_camera_parameters("ScreenCamera_2022-01-10-17-11-47.json")
    cam = ctr.convert_to_pinhole_camera_parameters()
    cam.extrinsic = np.array([[0.98930435683780737,
		0.078989956342454412,
		0.12262738820813895,
		0.0],[
		-0.0067297944331818974,
		0.86450697581859415,
		-0.50257576406734517,
		0.0],[
		-0.14571067019480621,
		0.49637513591839566,
		0.85579210386248294,
		0.0],[
		-23.560800573547034,
		26.969907112825666,
		434.30442709125299,
		1.0]])
    ctr.convert_from_pinhole_camera_parameters(cam)
    visualizer.register_key_action_callback(73,exit_key)
    visualizer.poll_events()

    # Create initial pointcloud
    pointcloud = open3d.geometry.PointCloud()
    visualizer.add_geometry(pointcloud)
    #visualizer.add_geometry(copy_textured_mesh)
    Axes = open3d.geometry.TriangleMesh.create_coordinate_frame(1)
    #visualizer.add_geometry(Axes)
    visualizer.run
    
    first = True
    while not exit_flag:

        # Get color image
        color_image = color_stream.read_frame()
        depth_image = depth_stream.read_frame()


        color_image, new_points = process_images(color_image, depth_image)
        
        pointcloud.points = open3d.utility.Vector3dVector(new_points)
        pointcloud.colors = open3d.utility.Vector3dVector(color_image.astype(np.float64)/255)
        print(pointcloud)
        pointcloud = pointcloud.voxel_down_sample(voxel_size=0.05)
        print("downsamped??",pointcloud)

        if first:
            visualizer.reset_view_point(True)
            first = False
        # Update visualizer
        visualizer.update_geometry(pointcloud)
        visualizer.poll_events()
        visualizer.update_renderer()

    depth_stream.stop()
    color_stream.stop()
    openni2.unload()
    visualizer.destroy_window()

if __name__ == "__main__":
    main()
