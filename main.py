import numpy as np
import cv2 as cv2
import open3d
import time
from copy import deepcopy
import json
import subprocess
from playsound import playsound
import os
import sys

if os.name == 'nt':
    proc = subprocess.Popen(['ffplay.exe', '-i', '-'], stdin=subprocess.PIPE)
else:
    proc = subprocess.Popen(['ffplay', '-i', '-'], stdin=subprocess.PIPE)

# False to use recorded video, True if live
USE_CAMERA = False

if USE_CAMERA:
    from openni import openni2
    from openni import _openni2

exit_flag=False
width=640
height=480

def playSound():
    f = open('Kick_2.wav', 'rb')
    try:
        proc.stdin.write(f.read())
    except:
        sys.exit(0)

def process_images(color_stream, depth_stream):
    color_image = color_stream.read_frame()
    depth_image = depth_stream.read_frame()

    color_image = np.frombuffer(color_image.get_buffer_as_uint8(), np.uint8)
    color_image = color_image.reshape(480 * 640, 3)
    
    depth_image = np.frombuffer(depth_image.get_buffer_as_uint16(), np.uint16)

    return color_image, depth_image

def closestPointsCloud(pointcloud, min, max):
    bounds = open3d.geometry.AxisAlignedBoundingBox(np.array([-10000000, -10000000, min], dtype=np.float64), np.array([10000000, 10000000, max], dtype=np.float64))
    new_pointcloud = pointcloud.crop(bounds)
    new_pointcloud = new_pointcloud.uniform_down_sample(20)
    colors = new_pointcloud.colors
    points = new_pointcloud.points
    
    #labels = DBSCAN([tuple(p) for p in points], lambda p1, p2: math.sqrt((p2[0] - p1[0])**2 + (p2[1] - p1[1])**2 + (p2[2] - p1[2])**2), 10, 10)
    labels = np.array(new_pointcloud.cluster_dbscan(eps=10, min_points=10))    
    
    for i in range(len(points)):
        #if points[i][2]>=min and points[i][2]<=max:
        if labels[i]==3:
            colors[i][0], colors[i][1], colors[i][2] = 255, 0, 0
    
    new_pointcloud.colors=colors
    
    labeled_points=dict()
    for i in range(len(points)):
        if labels[i] in labeled_points:
            labeled_points[labels[i]].append(points[i])
        else:
            labeled_points[labels[i]] = [points[i]]
    
    
    label_stats = dict()
    for i in range(len(points)):
        if labels[i] in label_stats:
            label_stats[labels[i]][0] += points[i][0]
            label_stats[labels[i]][1] += points[i][1]
            label_stats[labels[i]][2] += points[i][2]
            label_stats[labels[i]][3] += 1
        else:
            label_stats[labels[i]] = [points[i][0], points[i][1], points[i][2], 1]
    
    for k, v in label_stats.items():
        label_stats[k] = [v[0]/v[3], v[1]/v[3], v[2]/v[3]]
    
    
    hand_labels = []
    for k,v in label_stats.items():
        hand_labels.append((k,v[2]))
        
        if len(hand_labels)>2:
            hand_labels = sorted(hand_labels, key = lambda x: x[1])[:2]
    

    if len(hand_labels)==0:
        hand_points = []
    
    elif len(hand_labels)==1:
        hand_points = labeled_points[hand_labels[0][0]]
    else:
        hand_points = labeled_points[hand_labels[0][0]] + labeled_points[hand_labels[1][0]]
    
    
    return new_pointcloud, hand_points

def exit_key(vis,cena1, cena2):
    global exit_flag
    exit_flag=True
    
def main():
    global width, height, exit_flag, USE_CAMERA
    
    if USE_CAMERA:
        # Init openni
        openni_dir = "/home/pedro/OpenNI-Linux-x64-2.3/Redist"
        openni2.initialize(openni_dir)

        scene = open3d.t.geometry.RaycastingScene()
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
    drum_n_1_tmesh = open3d.t.geometry.TriangleMesh.from_legacy(drum_n_1)
    drum_n_2_tmesh = open3d.t.geometry.TriangleMesh.from_legacy(drum_n_2)
    drum_n_1.translate((-60,0,-70))
    drum_n_2.translate((0,0,-80))
    drum_n_1_tmesh.translate((-60,0,-70))
    drum_n_2_tmesh.translate((0,0,-80))
    #drum1 = scene.add_triangles(drum_n_1_tmesh)
    drum2 = scene.add_triangles(drum_n_2_tmesh)
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
            depth_image = depth_images[i]

            i+=1
            time.sleep(0.2)
        
        color_image = color_image.reshape(480, 640,3)

        depth_image = depth_image.reshape(480, 640)
        depth_image = depth_image.astype(np.float32)
        
        color_image = np.ascontiguousarray(color_image)
        depth_image = np.ascontiguousarray(depth_image)
        rgbd = open3d.geometry.RGBDImage.create_from_color_and_depth(
            open3d.geometry.Image(color_image),
            open3d.geometry.Image(depth_image),
            depth_scale=50,
            depth_trunc=8000,
            convert_rgb_to_intensity=False
        )
        new_pointcloud = open3d.geometry.PointCloud.create_from_rgbd_image(
            rgbd,
            intrinsic=parameters,
        )

        bounds = open3d.geometry.AxisAlignedBoundingBox(np.array([-10000000, -10000000, 1], dtype=np.float64), np.array([10000000, 10000000, 10000000], dtype=np.float64))
        new_pointcloud = new_pointcloud.crop(bounds)

        halfpointcloud, hand_points = closestPointsCloud(new_pointcloud, 20, 150)
        for point in hand_points:
            if scene.compute_distance(open3d.core.Tensor([[point[0], point[1], point[2]]], dtype=open3d.core.Dtype.Float32))<=200:
                playSound()
                break
        #halfpointcloud.estimate_normals()
        # if halfpointcloud.has_normals():
        #     poisson_mesh = open3d.geometry.TriangleMesh.create_from_point_cloud_poisson(halfpointcloud, depth=8, width=0, scale=1.1, linear_fit=False)[0]
        
        # flip pointcloud
        new_pointcloud.transform([[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]])
        
        # Set rendered pointcloud to recorded pointcloud
        pointcloud.points = new_pointcloud.points
        pointcloud.colors = new_pointcloud.colors
        

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
    proc.terminate()

if __name__ == "__main__":
    main()
