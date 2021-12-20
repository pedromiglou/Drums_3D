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

def get_rgbd(color_stream, depth_stream, depth_scale=1000, depth_trunc=4, convert_rgb_to_intensity=False):
    
    # Get color image
    color_image = color_stream.read_frame()
    color_image = np.frombuffer(color_image.get_buffer_as_uint8(), np.uint8)
    color_image = color_image.reshape(480, 640, 3)
    #color_image = cv.cvtColor(color_image, cv.COLOR_BGR2RGB)
    #color_image = color_image[:, ::-1, ::-1]
    cv.imwrite("Color.jpg", color_image)
    cv.imshow("rgb", color_image)
    
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
    global exit_flag
    print("VIS")
    exit_flag=True
    
def main():
    global width, height, fx, fy, cx, cy, exit_flag
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
    render = open3d.visualization.rendering.OffscreenRenderer(640,480)
    center = [0, 0, 0]  # look_at target
    eye = [0, 0, 10]  # camera position
    up = [0, 1, 0]  # camera orientation
    render.scene.camera.look_at(center, eye, up)
    visualizer=open3d.visualization.VisualizerWithKeyCallback()
    visualizer.create_window("Pointcloud", width=1000, height=700)
    #visualizer.create_window()

    #visualizer.create_window("Pointcloud", width=1000, height=700)
    visualizer.register_key_action_callback(32,exit_key)
    visualizer.poll_events()


    # Create initial pointcloud
    pointcloud = open3d.geometry.PointCloud()
    visualizer.add_geometry(pointcloud)
    Axes = open3d.geometry.TriangleMesh.create_coordinate_frame(1)

    first = True
    while not exit_flag:

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
        
        pointcloud.points = open3d.utility.Vector3dVector(new_points)
        pointcloud.colors = open3d.utility.Vector3dVector(color_image.reshape([color_image.shape[0]*color_image.shape[1], color_image.shape[2]]).astype(np.float64)/255)
        

        #open3d.visualization.draw_geometries([pointcloud, Axes])
        if first:
            visualizer.reset_view_point(True)
            first = False

        # Update visualizer
        visualizer.update_geometry(pointcloud)
        visualizer.poll_events()
        visualizer.update_renderer()
    print("Sai")
    depth_stream.stop()
    color_stream.stop()
    openni2.unload()
    visualizer.destroy_window()
    cv.destroyWindow("rgb")

if __name__ == "__main__":
    main()
