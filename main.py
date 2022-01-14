import numpy as np
import cv2 as cv2
import open3d
from numba import jit
import time
from copy import deepcopy
import json

# False to use recorded video, True if live
USE_CAMERA = False

if USE_CAMERA:
    from openni import openni2
    from openni import _openni2

exit_flag=False
width=640
height=480

@jit
def process_images(color_stream, depth_stream):
    color_image = color_stream.read_frame()
    depth_image = depth_stream.read_frame()

    color_image = np.frombuffer(color_image.get_buffer_as_uint8(), np.uint8)
    color_image = color_image.reshape(480 * 640, 3)
    
    depth_image = np.frombuffer(depth_image.get_buffer_as_uint16(), np.uint16)

    return color_image, depth_image
    
def exit_key(vis,cena1, cena2):
    global exit_flag
    exit_flag=True
    
def main():
    global width, height, exit_flag, USE_CAMERA
    
    if USE_CAMERA:
        # Init openni
        openni_dir = "/home/pedro/OpenNI-Linux-x64-2.3/Redist"
        openni2.initialize(openni_dir)

        # Open astra color and depth stream (using openni)
        depth_device = openni2.Device.open_any()
        color_stream = depth_device.create_color_stream()
        depth_stream = depth_device.create_depth_stream()
        depth_stream.set_video_mode(_openni2.OniVideoMode(pixelFormat = _openni2.OniPixelFormat.ONI_PIXEL_FORMAT_DEPTH_100_UM, resolutionX = 640, resolutionY = 480, fps = 30))
        color_stream.set_video_mode(_openni2.OniVideoMode(pixelFormat = _openni2.OniPixelFormat.ONI_PIXEL_FORMAT_RGB888, resolutionX = 640, resolutionY = 480, fps = 30))
        depth_stream.start()
        color_stream.start()
        
        depth_device.set_image_registration_mode(openni2.IMAGE_REGISTRATION_DEPTH_TO_COLOR)
    
    # read drum, position it and duplicate it
    drum_n_1 = open3d.io.read_triangle_mesh('drum.obj')
    drum_n_1.scale(7,drum_n_1.get_center())
    drum_n_1.rotate(drum_n_1.get_rotation_matrix_from_xyz((np.pi/3,0,0)), center=(0,0,0))
    drum_n_2 = deepcopy(drum_n_1)
    drum_n_1.translate((-150,0,400))
    drum_n_2.translate((150,0,400))

    visualizer=open3d.visualization.VisualizerWithKeyCallback()
    visualizer.create_window("Pointcloud", width=1000, height=700)

    ctr = visualizer.get_view_control()
    
    # get intrinsic parameters
    f = open("PinholeCameraParameters.json","r")
    parameters = json.loads(f.read())["intrinsic"]
    f.close()
    parameters = open3d.camera.PinholeCameraIntrinsic(
        width=parameters["width"],
        height=parameters["height"],
        fx=parameters["intrinsic_matrix"][0],
        fy=parameters["intrinsic_matrix"][4],
        cx=parameters["intrinsic_matrix"][6],
        cy=parameters["intrinsic_matrix"][7]
        )
    
    # extrinsic parameters
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
    visualizer.add_geometry(drum_n_1)
    visualizer.add_geometry(drum_n_2)

    Axes = open3d.geometry.TriangleMesh.create_coordinate_frame(10)
    visualizer.add_geometry(Axes)
    
    if not USE_CAMERA:
        with np.load('images.npz') as data:
            depth_images=data["depth_images"]
            color_images=data["color_images"]
            i=0
    
    first = True
    while not exit_flag:
        if USE_CAMERA:
            color_image, depth_image = process_images(color_stream, depth_stream)
        else:
            color_image = color_images[i]
            color_image = color_image.reshape(480, 640,3)

            depth_image = depth_images[i].reshape(480, 640)
            depth_image = depth_image.astype(np.float32)

            i+=1
            time.sleep(0.2)
        
        color_image = np.ascontiguousarray(color_image)
        depth_image = np.ascontiguousarray(depth_image)
        rgbd = open3d.geometry.RGBDImage.create_from_color_and_depth(
            open3d.geometry.Image(color_image),
            open3d.geometry.Image(depth_image),
            depth_scale=50,
            depth_trunc=800,
            convert_rgb_to_intensity=False
        )
        
        new_pointcloud = open3d.geometry.PointCloud.create_from_rgbd_image(
            rgbd,
            intrinsic=parameters,
        )
        
        # flip pointcloud
        new_pointcloud.transform([[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]])
        
        # Set rendered pointcloud to recorded pointcloud
        pointcloud.points = new_pointcloud.points
        pointcloud.colors = new_pointcloud.colors
        
        #print(pointcloud)
        #pointcloud = pointcloud.voxel_down_sample(voxel_size=0.05)
        #print("downsamped??",pointcloud)

        if first:
            visualizer.reset_view_point(True)
            first = False
        # Update visualizer
        visualizer.update_geometry(pointcloud)
        visualizer.poll_events()
        visualizer.update_renderer()
        
    if USE_CAMERA:
        depth_stream.stop()
        color_stream.stop()
        openni2.unload()
    visualizer.destroy_window()

if __name__ == "__main__":
    main()
