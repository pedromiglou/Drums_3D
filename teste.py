import numpy as np
from openni import openni2
from openni import _openni2
import cv2 as cv
import open3d
import copy
import time

SAVE_POINTCLOUDS = False

exit_flag=False
width=640,
height=480,
fx=654.75924295
fy=842.74853381
cx=325.50911362
cy=619.35980012

# def get_rgbd(color_capture, depth_stream, depth_scale=1000, depth_trunc=4, convert_rgb_to_intensity=False):
    
#     # Get color image
#     _, color_image = color_capture.read()
#     if  np.shape(color_image) == ():
#     	# Failed Reading
#         print("Image file could not be open")
#         exit(-1)
#     print(type(color_image))
#     color_image = color_image[:, ::-1, ::-1]
#     # Get depth image
#     depth_frame = depth_stream.read_frame()
#     #depth_image = np.frombuffer(depth_frame.get_buffer_as_uint8(), np.uint8)
#     depth_image = depth_image.reshape(depth_frame.height, depth_frame.width)
#     depth_image = depth_image.astype(np.float32)

#     # Create rgbd image from depth and color
#     color_image = np.ascontiguousarray(color_image)
#     depth_image = np.ascontiguousarray(depth_image)
#     print(color_image)
#     cv.imshow("color_image", color_image)
#     rgbd = open3d.geometry.RGBDImage.create_from_color_and_depth(
#         open3d.geometry.Image(color_image),
#         open3d.geometry.Image(depth_image),
#         depth_scale=depth_scale,
#         depth_trunc=depth_trunc,
#         convert_rgb_to_intensity=convert_rgb_to_intensity
#     )

#     return rgbd
def get_rgbd(color_stream, depth_stream, depth_scale=1000, depth_trunc=4, convert_rgb_to_intensity=False):
    
    # Get color image
    color_image = color_stream.read_frame()
    color_image = np.frombuffer(color_image.get_buffer_as_uint8(), np.uint8)
    color_image = color_image.reshape(480, 640, 3)
    #color_image = cv.cvtColor(color_image, cv.COLOR_BGR2RGB)
    #color_image = color_image[:, ::-1, ::-1]
    cv.imwrite("Color.jpg", color_image)
    cv.imshow("rgb", color_image)
    cv.waitKey( 0 )
    # Get depth image
    depth_frame = depth_stream.read_frame()
    depth_image = np.frombuffer(depth_frame.get_buffer_as_uint16(), np.uint16)
    depth_image = depth_image.reshape(depth_frame.height, depth_frame.width)
    cv.imwrite("Depth.jpg", depth_image)
    print(depth_image.shape)
    #depth_image = depth_image.astype(np.float32)
    """
    f = open("depth.txt", "w")
    for i in range(480):
        for j in range(640):
            f.write(str(depth_image[i,j])+"\n")
    f.close()
    print(depth_image)
    """
    # Create rgbd image from depth and color
    color_image = np.ascontiguousarray(color_image)
    depth_image = np.ascontiguousarray(depth_image)
    rgbd = open3d.geometry.RGBDImage.create_from_color_and_depth(
        open3d.geometry.Image(color_image),
        open3d.geometry.Image(depth_image),
        depth_scale=depth_scale,
        depth_trunc=depth_trunc,
        convert_rgb_to_intensity=convert_rgb_to_intensity
    )

    return color_image, depth_image, rgbd

    
def exit_key(vis):
    print("VIS")
    exit_flag=True
    vis.destroy_window()
    
def main():
    global width, height, fx, fy, cx, cy
    # Init openni
    openni_dir = "/home/pedro/OpenNI-Linux-x64-2.3/Redist"
    openni2.initialize(openni_dir)
    # Open astra color stream (using opencv)
    #color_capture = cv.VideoCapture(-1)
    #color_capture.set(cv.CAP_PROP_FPS, 5)

    # Open astra depth stream (using openni)
    depth_device = openni2.Device.open_any()
    color_stream = depth_device.create_color_stream()
    depth_stream = depth_device.create_depth_stream()
    depth_stream.set_video_mode(_openni2.OniVideoMode(pixelFormat = _openni2.OniPixelFormat.ONI_PIXEL_FORMAT_DEPTH_100_UM, resolutionX = 640, resolutionY = 480, fps = 30))
    color_stream.set_video_mode(_openni2.OniVideoMode(pixelFormat = _openni2.OniPixelFormat.ONI_PIXEL_FORMAT_RGB888, resolutionX = 640, resolutionY = 480, fps = 30))
    depth_stream.start()
    color_stream.start()
    
    # frame = color_stream.read_frame()
    # frame = np.frombuffer(frame.get_buffer_as_uint8(), np.uint8)
    # frame = frame.reshape(480, 640, 3)
    # frame = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
    # cv.imwrite("Foto.jpg", frame)
    # cv.imshow("rgb", frame)
    # cv.waitKey( 0 )
    depth_device.set_image_registration_mode(openni2.IMAGE_REGISTRATION_DEPTH_TO_COLOR)

    # Create pointcloud visualizer
    vis=open3d.visualization.VisualizerWithKeyCallback()
    visualizer = open3d.visualization.Visualizer()
    #visualizer.create_window("Pointcloud", width=1000, height=700)
    vis.register_key_callback(32,exit_key)
    vis.poll_events()
    # Camera intrinsics of the astra pro
    astra_camera_intrinsics = open3d.camera.PinholeCameraIntrinsic(
        )


    # Create initial pointcloud
    pointcloud = open3d.geometry.PointCloud()
    visualizer.add_geometry(pointcloud)
    Axes = open3d.geometry.TriangleMesh.create_coordinate_frame(1)

    first = True
    prev_timestamp = 0
    num_stored = 0

    cv.waitKey()

    color_image, depth_image, rgbd = get_rgbd(color_stream, depth_stream)



    new_points = np.zeros((640*480,3), dtype=np.float64)

    for i in range(480):
        for j in range(640):
            z = depth_image[i, j]/10
            x = (j-cx)*z/fx
            y = (i-cy)*z/fy
            new_points[i*640+j, 0] = x
            new_points[i*640+j, 1] = y
            new_points[i*640+j, 2] = z
    
    new_pointcloud = open3d.geometry.PointCloud()
    pointcloud.points = open3d.utility.Vector3dVector(new_points)
    pointcloud.colors = open3d.utility.Vector3dVector(color_image.reshape([color_image.shape[0]*color_image.shape[1], color_image.shape[2]]).astype(np.float64)/255)

    # Convert images to pointcloud
    print(new_points)
    """
    new_pointcloud = open3d.geometry.PointCloud.create_from_rgbd_image(
        rgbd,
        intrinsic=astra_camera_intrinsics,
    )

    
    new_pointcloud = open3d.geometry.PointCloud.create_from_depth_image(
        rgbd.depth,
        intrinsic=astra_camera_intrinsics,
    )
    """
    print(new_pointcloud.points)
    # Flip pointcloud
    open3d.visualization.draw_geometries([pointcloud, Axes])
    new_pointcloud.transform([[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]])
    # Set rendered pointcloud to recorded pointcloud
    print(new_pointcloud.points)
    pointcloud.points = new_pointcloud.points
    pointcloud.colors = new_pointcloud.colors

    # Save pointcloud each n seconds
    if SAVE_POINTCLOUDS and time.time() > prev_timestamp + 5:
        filename = "pointcloud-%r.pcd" % num_stored
        open3d.io.write_point_cloud(filename, new_pointcloud)
        num_stored += 1
        print("Stored: %s" % filename)
        prev_timestamp = time.time()

    # Reset viewpoint in first frame to look at the scene correctly
    # (e.g. correct bounding box, direction, distance, etc.)
    if first:
        visualizer.reset_view_point(True)
        first = False

    # Update visualizer
    visualizer.poll_events()
    visualizer.update_renderer()
    depth_stream.stop()
    openni2.unload()
    cv.destroyWindow( "rgb" )

if __name__ == "__main__":
    main()



#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# from primesense import openni2, _openni2
# import platform
# import numpy as np
# import cv2

# # Initialize OpenNI
# if platform.system() == "Windows":
#     openni2.initialize("C:/Program Files/OpenNI2/Redist")  # Specify path for Redist
# else:
#     openni2.initialize("/home/pedro/OpenNI-Linux-x64-2.3/Redist")  # can also accept the path of the OpenNI redistribution

# # Connect and open device
# dev = openni2.Device.open_any()
# # Create depth stream
# depth_stream = dev.create_depth_stream()
# depth_stream.set_video_mode(_openni2.OniVideoMode(pixelFormat = _openni2.OniPixelFormat.ONI_PIXEL_FORMAT_DEPTH_100_UM, resolutionX = 640, resolutionY = 480, fps = 30))
# depth_stream.start()
# cv2.namedWindow("Depth View", cv2.WINDOW_AUTOSIZE)
# while cv2.waitKey(1) == -1:
#     frame = depth_stream.read_frame()
#     print(frame.height)
#     method_list = [func for func in dir(frame) if callable(getattr(frame, func))]
#     print(method_list)
#     frame_data=frame.get_buffer_as_uint16()
#     print(frame_data)
#     img = np.frombuffer(frame_data, dtype=np.uint16)
#     print(len(img))
#     img.shape = (1, 480, 640)
#     img = np.concatenate((img, img, img), axis=0)
#     img = np.swapaxes(img, 0, 2)
#     img = np.swapaxes(img, 0, 1)
#     cv2.imshow("image", img)
#     # Trimming depth_array
#     max_distance = 800
#     min_distance = 0
#     out_of_range = img > max_distance
#     too_close_range = img < min_distance
#     img[out_of_range] = max_distance
#     img[too_close_range] = min_distance

#     # Scaling depth array
#     depth_scale_factor = 255.0 / (max_distance - min_distance)
#     depth_scale_offset = -(min_distance * depth_scale_factor)
#     depth_array_norm = img * depth_scale_factor + depth_scale_offset

#     rgb_frame = cv2.applyColorMap(depth_array_norm.astype(np.uint8), cv2.COLORMAP_JET)

#     # Replacing invalid pixel by black color
#     #rgb_frame[np.where(img == 0)] = [0, 0, 0]

#     # Display image
#     rgb_frame = cv2.resize(rgb_frame, (800, 600), interpolation=cv2.INTER_AREA)
#     cv2.imshow("Depth View", rgb_frame)

# depth_stream.stop()
# openni2.unload()