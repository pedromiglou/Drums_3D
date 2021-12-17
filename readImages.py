import numpy as np
from openni import openni2
from openni import _openni2
import cv2 as cv
import open3d
import copy
import time

SAVE_POINTCLOUDS = False

exit_flag=False
fps_scale=6
def get_rgbd(color_stream, depth_stream, depth_scale=1000, depth_trunc=4, convert_rgb_to_intensity=False):
    # Get color image
    color_image = color_stream.read_frame()
    color_image = np.frombuffer(color_image.get_buffer_as_uint8(), np.uint8)
    color_image = color_image.reshape(480, 640, 3)
    #color_image = cv.cvtColor(color_image, cv.COLOR_BGR2RGB)
    color_image = color_image[:, ::-1, ::-1]

    # Get depth image
    depth_frame = depth_stream.read_frame()
    depth_image = np.frombuffer(depth_frame.get_buffer_as_uint16(), np.uint16)
    depth_image = depth_image.reshape(480, 640)

    return depth_image, color_image
    
def main():
    global fps_scale
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
    depth_images = np.empty([],(0,9))
    depth_device.set_image_registration_mode(openni2.IMAGE_REGISTRATION_DEPTH_TO_COLOR)

    for _ in range(fps_scale):
        depth_image1, color_image1 = get_rgbd(color_stream, depth_stream)
        np.append(depth_image1)
    for _ in range(fps_scale):
        depth_image2, color_image2 = get_rgbd(color_stream, depth_stream)
        np.append(depth_images, depth_image2)

    for _ in range(fps_scale):
        depth_image3, color_image3 = get_rgbd(color_stream, depth_stream)
        np.append(depth_images, depth_image3)

    for _ in range(fps_scale):
        depth_image4, color_image4 = get_rgbd(color_stream, depth_stream)
        np.append(depth_images, depth_image4)
    for _ in range(fps_scale):
        depth_image5, color_image5 = get_rgbd(color_stream, depth_stream)
        np.append(depth_images, depth_image5)
    for _ in range(fps_scale):
        depth_image6, color_image6 = get_rgbd(color_stream, depth_stream)
        np.append(depth_images, depth_image6)
    for _ in range(fps_scale):
        depth_image7, color_image7 = get_rgbd(color_stream, depth_stream)
        np.append(depth_images, depth_image7)
    for _ in range(fps_scale):
        depth_image8, color_image8 = get_rgbd(color_stream, depth_stream)
        np.append(depth_images, depth_image8)
    for _ in range(fps_scale):
        depth_image9, color_image9 = get_rgbd(color_stream, depth_stream)
        np.append(depth_images, depth_image9)
    for _ in range(fps_scale):
        depth_image10, color_image10 = get_rgbd(color_stream, depth_stream)
        np.append(depth_images, depth_image10)
    np.savez("images.npz",
        depth_images=depth_images,
        color_images=np.array([color_image1, color_image2, color_image3, color_image4, color_image5, color_image6, color_image7, color_image8, color_image9, color_image10])
    )

if __name__ == "__main__":
    main()
