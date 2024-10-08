import numpy as np
import open3d
from copy import deepcopy
import json
import math
import threading
import os
import pygame
from openni import openni2
from openni import _openni2

os.environ['SDL_AUDIODRIVER'] = 'alsa'
pygame.init()
pygame.mixer.init()

exit_flag=False
width=640
height=480


def play_music(file, speed):
    """play sound from a certain file"""
    sound = pygame.mixer.Sound(file)
    amplitude = speed*0.01
    sound.set_volume(amplitude)
    sound.play()


def get_new_frame(color_stream, depth_stream):
    """return a color and a depth frame from a color and a depth stream"""
    color_image = color_stream.read_frame()
    depth_image = depth_stream.read_frame()

    color_image = np.frombuffer(color_image.get_buffer_as_uint8(), np.uint8)
    color_image = color_image.reshape(480, 640, 3)
    
    depth_image = np.frombuffer(depth_image.get_buffer_as_uint16(), np.uint16)
    depth_image = depth_image.reshape(480, 640)
    depth_image = depth_image.astype(np.float32)

    return color_image, depth_image


def detect_hands(pointcloud, min, max):
    """search for the point clusters closest to the camera inside a certain z-axis interval"""
    
    # crop and downsample the initial pointcloud to heavily reduce the number of points
    bounds = open3d.geometry.AxisAlignedBoundingBox(np.array([-10000000, -10000000, min], dtype=np.float64), np.array([10000000, 10000000, max], dtype=np.float64))
    new_pointcloud = pointcloud.crop(bounds)
    new_pointcloud.transform([[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]])
    new_pointcloud = new_pointcloud.voxel_down_sample(1)
    
    points = new_pointcloud.points
    
    if new_pointcloud.is_empty():
        labels = []
    else:
        # obtain the labels for each point
        labels = np.array(new_pointcloud.cluster_dbscan(eps=5, min_points=5))

    # organize the points into lists of points that belong to the same cluster
    labeled_points=dict()
    for i in range(len(points)):
        if labels[i] in labeled_points:
            labeled_points[labels[i]].append(points[i])
        elif labels[i]!=-1:
            labeled_points[labels[i]] = [points[i]]
    
    # calculate centroids
    label_stats = dict()
    for i in range(len(points)):
        if labels[i] in label_stats:
            label_stats[labels[i]][0] += points[i][0]
            label_stats[labels[i]][1] += points[i][1]
            label_stats[labels[i]][2] += points[i][2]
            label_stats[labels[i]][3] += 1
        elif labels[i]!=-1:
            label_stats[labels[i]] = [points[i][0], points[i][1], points[i][2], 1]
    
    for k, v in label_stats.items():
        label_stats[k] = [v[0]/v[3], v[1]/v[3], v[2]/v[3]]
    
    # filter the two centroids closest to the camera
    hand_labels = []
    for k,v in label_stats.items():
        hand_labels.append((k,v))
        
        if len(hand_labels)>2:
            hand_labels = sorted(hand_labels, key = lambda x: x[1][2])[:2]
    
    hand_labels = sorted(hand_labels, key = lambda x: x[1][0])

    centroids = [x[1] for x in hand_labels]
    
    # get the points that belong to the hands
    if len(hand_labels)==0:
        hand_points = np.zeros((1,3))
    
    elif len(hand_labels)==1:
        hand_points = np.vstack(labeled_points[hand_labels[0][0]])

    else:
        hand_points = np.vstack(labeled_points[hand_labels[0][0]] + labeled_points[hand_labels[1][0]])

    hand_points = open3d.utility.Vector3dVector(hand_points)

    return hand_points, centroids


def exit_key(vis, v1, v2):
    """callback function to exit open3D environment"""
    global exit_flag
    exit_flag=True
    

def movement_speed(prev_centroids, centroids, this_drum):
    """obtain the speed of the movement / distance difference between the actual and last position of the centroid closest to a drum"""
    if len(prev_centroids)==len(centroids):
        if len(centroids)==1:
            return math.sqrt((prev_centroids[0][0]-centroids[0][0])**2 + (prev_centroids[0][1]-centroids[0][1])**2 + (prev_centroids[0][2]-centroids[0][2])**2)

        if len(centroids)==2:
            if math.sqrt((this_drum[0]-centroids[0][0])**2 + (this_drum[1]-centroids[0][1])**2 + (this_drum[2]-centroids[0][2])**2) > math.sqrt((this_drum[0]-centroids[1][0])**2 + (this_drum[1]-centroids[1][1])**2 + (this_drum[2]-centroids[1][2])**2):
                return math.sqrt((prev_centroids[1][0]-centroids[1][0])**2 + (prev_centroids[1][1]-centroids[1][1])**2 + (prev_centroids[1][2]-centroids[1][2])**2)
            else:
                return math.sqrt((prev_centroids[0][0]-centroids[0][0])**2 + (prev_centroids[0][1]-centroids[0][1])**2 + (prev_centroids[0][2]-centroids[0][2])**2)


def main():
    global width, height, exit_flag
    
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
    drum_n_1.scale(1.7,drum_n_1.get_center())
    drum_n_1.rotate(drum_n_1.get_rotation_matrix_from_xyz((-np.pi/3,0,np.pi/6)), center=(0,0,0))
    drum_n_1.compute_triangle_normals()
    drum_n_2 = deepcopy(drum_n_1)
    drum_n_1.translate((-40,0,-70))
    drum_n_2.translate((20,0,-80))
    bbox1 = drum_n_1.get_oriented_bounding_box()
    bbox2 = drum_n_2.get_oriented_bounding_box()

    visualizer=open3d.visualization.VisualizerWithKeyCallback()
    visualizer.create_window("Pointcloud", width=1000, height=700)

    ctr = visualizer.get_view_control()
    camera_params = open3d.io.read_pinhole_camera_parameters("PinholeCameraParameters.json")
    ctr.convert_from_pinhole_camera_parameters(camera_params)

    # get intrinsic parameters from file
    f = open("PinholeCameraParameters.json","r")
    parameters = json.loads(f.read())
    f.close()
    i_parameters = open3d.camera.PinholeCameraIntrinsic(
        width=parameters['intrinsic']["width"],
        height=parameters['intrinsic']["height"],
        fx=parameters['intrinsic']["intrinsic_matrix"][0],
        fy=parameters['intrinsic']["intrinsic_matrix"][4],
        cx=parameters['intrinsic']["intrinsic_matrix"][6],
        cy=parameters['intrinsic']["intrinsic_matrix"][7]
        )

    visualizer.register_key_action_callback(73,exit_key)

    # Create initial pointcloud
    pointcloud = open3d.geometry.PointCloud()
    visualizer.add_geometry(pointcloud)
    visualizer.add_geometry(drum_n_1)
    visualizer.add_geometry(drum_n_2)
    visualizer.update_renderer()
    
    # set up open3D camera point of view
    ctr.translate(0,300)
    ctr.rotate(1000,300)
    ctr.set_lookat(np.array([-15,-50,0]))
    ctr.set_zoom(2)
    
    # initialize variables used inside the the cicle
    touching_d1 = False
    touching_d2 = False
    prev_centroids = []
    
    while not exit_flag:
        color_image, depth_image = get_new_frame(color_stream, depth_stream)
        
        # create RGBD images after getting new frame
        color_image = np.ascontiguousarray(color_image)
        depth_image = np.ascontiguousarray(depth_image)
        rgbd = open3d.geometry.RGBDImage.create_from_color_and_depth(
            open3d.geometry.Image(color_image),
            open3d.geometry.Image(depth_image),
            depth_scale=50,
            depth_trunc=8000,
            convert_rgb_to_intensity=False
        )
        
        # obtain pointcloud
        new_pointcloud = open3d.geometry.PointCloud.create_from_rgbd_image(
            rgbd,
            intrinsic=i_parameters,
        )

        # crop points at zero depth
        bounds = open3d.geometry.AxisAlignedBoundingBox(np.array([-10000000, -10000000, 1], dtype=np.float64), np.array([10000000, 10000000, 10000000], dtype=np.float64))
        new_pointcloud = new_pointcloud.crop(bounds)

        # detect hands
        hand_points, centroids = detect_hands(new_pointcloud, 20, 125)

        # calculate hand movement speed and then verify if a sound should happen and its intensity
        speed = movement_speed(prev_centroids, centroids, bbox1.get_center())
        if len(bbox1.get_point_indices_within_bounding_box(hand_points)):
            if not touching_d1 and speed != None and speed > 5:

                x=threading.Thread(target=play_music,args=('Kick_2.wav',speed))
                x.start()
                drum_n_1.paint_uniform_color(np.array([0.4,0.7,0.4], dtype=np.float64))
            drum_n_1.paint_uniform_color(np.array([0.7,0.4,0.4], dtype=np.float64))
            visualizer.update_geometry(drum_n_1)
            touching_d1=True
        else:
            drum_n_1.paint_uniform_color(np.array([0.8,0.8,0.8], dtype=np.float64))
            visualizer.update_geometry(drum_n_1)
            touching_d1=False

        speed = movement_speed(prev_centroids, centroids, bbox2.get_center())
        if len(bbox2.get_point_indices_within_bounding_box(hand_points)):
            
            if not touching_d2 and speed != None and speed > 5:
                x=threading.Thread(target=play_music,args=('Kick_3.wav',speed))
                x.start()
            drum_n_2.paint_uniform_color(np.array([0.7,0.4,0.4], dtype=np.float64))
            visualizer.update_geometry(drum_n_2)
            touching_d2=True
        else:
            drum_n_2.paint_uniform_color(np.array([0.8,0.8,0.8], dtype=np.float64))
            visualizer.update_geometry(drum_n_2)
            touching_d2=False
    
        prev_centroids = centroids
        
        # flip pointcloud
        new_pointcloud.transform([  [1, 0, 0, 0],
                                    [0, -1, 0, 0],
                                    [0, 0, -1, 0],
                                    [0, 0, 0, 1]])

        # Set rendered pointcloud to recorded pointcloud
        pointcloud.points = new_pointcloud.points
        pointcloud.colors = new_pointcloud.colors
        
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