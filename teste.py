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
fx=654.75924295
fy=842.74853381
cx=325.50911362
cy=619.35980012

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


    return color_image, new_points

    
def exit_key(vis,cena1, cena2):
    print(cena1)
    print(cena2)
    global exit_flag
    exit_flag=True
    
def main():
    global width, height, fx, fy, cx, cy, exit_flag
    # Init openni
    openni_dir = "/home/pedro/OpenNI-Linux-x64-2.3/Redist"
    openni2.initialize(openni_dir)

    # Open astra depth stream (using openni)
    depth_device = openni2.Device.open_any()
    color_stream = depth_device.create_color_stream()
    depth_stream = depth_device.create_depth_stream()
    depth_stream.set_video_mode(_openni2.OniVideoMode(pixelFormat = _openni2.OniPixelFormat.ONI_PIXEL_FORMAT_DEPTH_100_UM, resolutionX = 640, resolutionY = 480, fps = 30))
    color_stream.set_video_mode(_openni2.OniVideoMode(pixelFormat = _openni2.OniPixelFormat.ONI_PIXEL_FORMAT_RGB888, resolutionX = 640, resolutionY = 480, fps = 30))
    depth_stream.start()
    color_stream.start()
    
    depth_device.set_image_registration_mode(openni2.IMAGE_REGISTRATION_DEPTH_TO_COLOR)

    # Create pointcloud visualizer
    render = open3d.visualization.rendering.OffscreenRenderer(640,480)
    center = [0, 0, 0]  # look_at target
    eye = [0, 0, 10]  # camera position
    up = [0, 1, 0]  # camera orientation
    render.scene.camera.look_at(center, eye, up)
    visualizer=open3d.visualization.VisualizerWithKeyCallback()
    visualizer.create_window("Pointcloud", width=1000, height=700)

    visualizer.register_key_action_callback(73,exit_key)
    visualizer.poll_events()


    # Create initial pointcloud
    pointcloud = open3d.geometry.PointCloud()
    visualizer.add_geometry(pointcloud)
    Axes = open3d.geometry.TriangleMesh.create_coordinate_frame(1)
    ctr = visualizer.get_view_control()
    parameters = open3d.io.read_pinhole_camera_parameters("ScreenCamera_2022-01-10-17-11-47.json")
    ctr.convert_from_pinhole_camera_parameters(parameters)
    first = True
    while not exit_flag:

        # Get color image
        color_image = color_stream.read_frame()
        depth_image = depth_stream.read_frame()


        color_image, new_points = process_images(color_image, depth_image)
        
        pointcloud.points = open3d.utility.Vector3dVector(new_points)
        pointcloud.colors = open3d.utility.Vector3dVector(color_image.astype(np.float64)/255)

        if first:
            visualizer.reset_view_point(True)
            first = False
        #pointcloud = pointcloud.voxel_down_sample(voxel_size=0.05)
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
