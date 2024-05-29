#!/usr/bin/env python3

import json
import os
import numpy as np
import configargparse
import cv2
import shutil
import copy
import random
import time

from geometry_msgs.msg import Point, Quaternion
from math import sin, cos, acos, pi, atan, log, tan, sqrt

from sko.DE import DE

from trajectory_handler import TrajectoryHandler
from robot_control import get_robot_handler
from scene_handler import get_scene_handler
from camera_handler import get_camera_handler, load_camera_properties
from quartonian_handler import QuartonianHandler
from sfm_handler import get_sfm_handler
from db_handler import ImgCaptureDatabaseHandler
from turntable_handler import get_turntable_handler
from foreground_segmentor import get_foreground_segmentor
from experiment_handler import ExperimentHandler
from reconstruction_handler import get_reconstruction_handler

def closest_point_2_lines(oa, da, ob, db): # returns point closest to both rays of form o+t*d, and a weight factor that goes to 0 if the lines are parallel
	da = da / np.linalg.norm(da)
	db = db / np.linalg.norm(db)
	c = np.cross(da, db)
	denom = np.linalg.norm(c)**2
	t = ob - oa
	ta = np.linalg.det([t, db, c]) / (denom + 1e-10)
	tb = np.linalg.det([t, da, c]) / (denom + 1e-10)
	if ta > 0:
		ta = 0
	if tb > 0:
		tb = 0
	return (oa+ta*da+ob+tb*db) * 0.5, denom

class ViewCapture():
    def __init__(self, args):

        self.validate_args(args)

        self.args = args

        self.setup_lambdas()

        self.instantiate_main_objects()

        self.args.main_obj_position = Point(args.main_obj_position[0], args.main_obj_position[1], args.main_obj_position[2]) 

        self.object_center = Point(args.main_obj_position.x, args.main_obj_position.y, args.main_obj_position.z + args.main_obj_size[2]/2)

        self.generate_objects()

        self.controllers = []
        self.configure_robot_controllers()

        self.frame_data = {}

        self.sfm_handler = None
        if args.adjust_transforms:
            self.sfm_handler = get_sfm_handler(self.args.sfm_package, self.args.experiment_name, self.args.log_dir, 
                                           img_dir=self.experiment_handler.get_rgb_dir(), 
                                           mask_dir=self.experiment_handler.get_mask_dir() if self.args.use_mask_in_sfm and self.args.segment_method is not None else None)

        # Up vector for the environment 
        self.up = Point(0.0, 0.0, 1.0)

        self.start_rad = None

    def validate_args(self, args):
        if not os.path.isdir(args.log_dir):
            raise Exception("Unable to locate log directory")
        
        if len(args.main_obj_size) != 0 and len(args.main_obj_size) != 3:
            raise Exception("Main Object Size must have 3 values (x,y,z)")
        
        if args.obj_stand_thickness <= 0:
            raise Exception("Object stand thickness must be larger than 0m")

        if args.capture_radius <= 0:
            raise Exception("Capture Radius must be larger than 0m")

        if len(args.aabb) != 6 and len(args.aabb) != 0:
            raise Exception("AABB must have 6 values (x1,x2,y1,y2,z1,z2)")
        
        if args.rings * args.sectors <= 0:
            raise Exception("Number of rings and sectors must be larger than 0")

        if args.num_move_attempts <= 0:
            raise Exception("Number of move attempts must be larger than 0")
        
        if args.planning_time <= 0:
            raise Exception("Planning time must be larger than 0 seconds")
        
        if args.num_move_attempts <= 0:
            raise Exception("Number of move attempts be larger than 0 seconds")

        if args.use_turntable and args.turntable_connection_port is None:
            raise Exception("Must have a USB connection link for turntable device")

        if len(args.main_obj_position) != 3:
            raise Exception("Main Object Position must have 3 values (x,y,z)")

    def setup_lambdas(self):
        # Accepts a list of three values and returns a ros Point object
        self.convert_list_to_point = lambda point_list: Point(point_list[0], point_list[1], point_list[2])

        self.is_train_img = lambda point_idx: False if (self.args.test_incrementation > 0 and point_idx % self.args.test_incrementation == 0) else True

        self.img_file_name = lambda point_idx: str(point_idx).zfill(4) + "_train.png" if self.is_train_img(point_idx) else str(point_idx).zfill(4) + "_eval.png"

        self.aabb_scale = lambda scale: min(2 ** round(log(scale * 4, 2)), 128)

    def instantiate_main_objects(self):
        self.db_handler = ImgCaptureDatabaseHandler(self.args.log_dir)

        self.quartonian_handler = QuartonianHandler()

        self.scene_handler = get_scene_handler(self.args.scene_handler_type)

        self.experiment_handler = ExperimentHandler(self.args.log_dir, self.args.experiment_name)

        self.config_experiment()

        self.trajectory_handler = TrajectoryHandler(self.args.restricted_x, self.args.restricted_y, 
                                                    1.0, self.args.restricted_z, 
                                                    log_dir=self.experiment_handler.get_experiment_dir())
        
        self.turntable_handler = get_turntable_handler(self.args.turntable_handler, 
                                                       self.args.turntable_connection_port) if self.args.use_turntable else None

        self.foreground_segmenter = get_foreground_segmentor(self.args.segment_method) if self.args.segment_method else None

    def config_experiment(self):

        # If a previously saved experiment directory exists with the same name as the current experiment
        if self.experiment_handler.does_experiment_exist():
            print("Previous experiment data located")
            print()

            # Argument continue experiment is set
            if self.args.continue_experiment:
                print("WARNING: You have chosen to continue an existing experiment with this name")
                print("All parameters from the previous experiment are now being used")
                
                prev_config = self.experiment_handler.get_experiment_config()

                if prev_config is not None:
                    self.args.__dict__ = prev_config
                else:
                    print("WARNING: Failed to load previous experiment config info!")

                # Check that database contains information for the current experiment
                if self.db_handler.get_experiment_with_name(self.args.experiment_name):
                    self.db_handler.set_current_experiment(self.args.experiment_name)

                    self.args.continue_experiment = True
                    
                # If not, alert user that a new experiment will be created in the database
                else:
                    print("ERROR: Experiment with this name was not found in the database.")
                    print("Starting again from scratch but keeping existing directory data!")

                    self.db_handler.create_new_experiment(self.args.experiment_name)

                    self.experiment_handler.create_new_dir(self.args)

            # Argument replace stored experiment is set
            elif self.args.replace_stored_experiment:
                print("WARNING: You have chosen to replace previous experiments with this name")
                print("Press exit to cancel or press any button to remove all previous experiment data")

                user_input = input()
                if user_input == "exit":
                    exit()
                
                self.experiment_handler.remove_current_experiment()

                # Experiment with this name is removed from the database
                self.db_handler.remove_experiment_with_name(self.args.experiment_name)

                self.db_handler.create_new_experiment(self.args.experiment_name)

                # New experiment is instantiated
                self.experiment_handler.create_new_dir(self.args)

            else:
                db_name_i = 1

                new_experiment_name = self.args.experiment_name + "_" + str(db_name_i)

                # Keeps the same experiment name but attaches an end digit, while loop and increment
                # until an experiment with this name does not exist in the database
                while self.db_handler.get_experiment_with_name(new_experiment_name):

                    # Experiment found but experiment directory not found
                    if not os.path.join(self.args.log_dir, new_experiment_name):
                        print("WARNING: Experiment with name {new_experiment_name} found in database, but cannot find experiment directory")
                        print("Press exit to cancel or press any button to remove all previous experiment data")

                        user_input = input()
                        if user_input == "exit":
                            exit()

                        self.db_handler.remove_experiment_with_name(new_experiment_name)
                        
                        break
                    
                    db_name_i+= 1
                    new_experiment_name = self.args.experiment_name + "_" + str(db_name_i)
                
                print(f"Created new experiment with name: {new_experiment_name}")

                self.args.experiment_name = new_experiment_name

                self.db_handler.create_new_experiment(self.args.experiment_name)

                self.experiment_handler.set_experiment_name(self.args.experiment_name)

                # New experiment is instantiated
                self.experiment_handler.create_new_dir(self.args)

        # Experiment directory with this experiment name could not be found
        else:
            print("No previous experiment directory located")

            # If the database contains an experiment with this name, then it should be removed, since the accompaning 
            # images for this experiment are not present
            if self.db_handler.get_experiment_with_name(self.args.experiment_name):
                print("ERROR: Experiment with this name was found in the database but has no directory data.")
                print("Press exit to cancel or press any button to remove all previous experiment data")

                user_input = input()
                if user_input == "exit":
                    exit()
                
                self.db_handler.remove_experiment_with_name(self.args.experiment_name)

            print("Creating new experiment")

            self.db_handler.create_new_experiment(self.args.experiment_name)

            # New experiment is instantiated
            self.experiment_handler.create_new_dir(self.args)

    def get_all_camera_properties(self): 
        self.camera_properties = {}

        for controller_i, controller in enumerate(self.controllers):
            for robot_j, robot in enumerate(controller.robots):
                self.camera_properties[robot.get_global_id()] = robot.get_camera_handler().get_camera_properties()

    def configure_robot_controllers(self):

        with open(self.args.robot_settings_path, "r") as f:
            controller_descriptions = json.load(f)
        
        global_robot_id = 0

        for controller_description in controller_descriptions:
            new_controller = get_robot_handler(self.args.robot_handler_type, controller_description["move_group"])

            new_controller.set_controller_name(controller_description["name"])

            if "robots" not in controller_description:
                raise Exception("Controller must have at least one defined robot")

            for robot_description in controller_description["robots"]:
                new_robot_idx = new_controller.create_new_robot()
                
                if "camera_transform" in robot_description:
                    new_controller.robots[new_robot_idx].set_endeffector_transform_name(robot_description["end_effector_transform"])

                if "base_transform" in robot_description:
                    new_controller.robots[new_robot_idx].set_base_transform_name(robot_description["base_transform"])

                if "urdf_path" in robot_description:
                    new_controller.robots[new_robot_idx].set_urdf_file_path(robot_description["urdf_path"])
                
                if "reach" in robot_description:
                    new_controller.robots[new_robot_idx].set_reach(robot_description["reach"])

                if not self.args.discard_img_capturing:
                    camera_properties = None
                    if "camera_properties_file_path" in robot_description:
                        camera_properties = load_camera_properties(robot_description["camera_properties_file_path"])

                    if "camera_transform" in robot_description:
                        new_controller.robots[new_robot_idx].set_camera_transform_name(robot_description["camera_transform"])

                    if "camera_topic" in robot_description:
                        camera_handler = get_camera_handler(self.args.camera_handler_type, camera_topic=robot_description["camera_topic"],
                                                            serial_number=robot_description["camera_serial_no"], crop_w=self.args.crop_width, 
                                                            crop_h=self.args.crop_height)
                    else:


                        camera_handler = get_camera_handler(self.args.camera_handler_type, serial_number=robot_description["camera_serial_no"],
                                                            stream_w=int(camera_properties["w"]) if camera_properties is not None else 1920, 
                                                            stream_h=int(camera_properties["h"]) if camera_properties is not None else 1080,
                                                            fps=int(robot_description["camera_fps"]) if "camera_fps" in robot_description else 30, 
                                                            crop_w=self.args.crop_width,  crop_h=self.args.crop_height)

                    camera_handler.set_camera_properties(camera_properties)

                    if not camera_handler.is_connected():
                        raise Exception("No camera connection could be established, check that the camera image is being published correctly")

                    new_controller.robots[new_robot_idx].set_camera_handler(camera_handler)
                
                if "starting_position" in robot_description:
                    starting_position = self.convert_list_to_point(robot_description["starting_position"])

                    new_controller.add_position(starting_position, new_robot_idx)
                
                global_robot_id += 1

                new_controller.robots[new_robot_idx].set_global_id(global_robot_id)

            new_controller.execute_plan()

            self.controllers.append(new_controller)

        if len(self.controllers) < 1:
            raise Exception("Robot description file must include valid instance of a controller")

        if not self.args.discard_img_capturing:
            self.get_all_camera_properties()

    def generate_objects(self):
        # Creates the main object to capture in the environment with box size
        self.scene_handler.add_box("main_obj", self.object_center, 
                                    self.args.main_obj_size, attach=False)

        # If a stand is to be autogenerated (use if not part of the URDF file)
        if self.args.auto_add_obj_stand:
            
            # Avoids the main object memory being overwritten
            stand_position = self.convert_list_to_point([self.args.main_obj_position.x, 
                                                    self.args.main_obj_position.y, 
                                                    self.args.main_obj_position.z])
            
            stand_size = [self.args.obj_stand_thickness, self.args.obj_stand_thickness, stand_position.z]
            stand_position.z /= 2 

            self.scene_handler.add_box("stand", stand_position, stand_size, attach=False)

        if self.args.auto_add_floor:
            floor_position = self.convert_list_to_point([0,0,0])

            floor_size = [5, 5, 0.005]

            self.scene_handler.add_box("floor", floor_position, floor_size, attach=False)
        
        if self.args.auto_add_ceiling:
            ceiling_position = self.convert_list_to_point([0, 0, self.args.ceiling_height])

            ceiling_size = [5, 5, 0.005]

            self.scene_handler.add_box("ceilng", ceiling_position, ceiling_size, attach=False)

        # If set, will include extra scene objects defined in the object scene file
        if self.args.scene_objs is not None:
            print("Adding scene objects")

            with open(self.args.scene_objs, "r") as scene_file:
                json_file = json.load(scene_file)

                objects = json_file["objects"]

                # Loops through each of the specified objects in the object file
                for scene_object in objects:

                    print()

                    # Converts position and size to RC coordinates
                    obj_pos = self.convert_list_to_point(scene_object["position"])

                    # Adds the defined object type to the environment
                    if scene_object["type"] == "mesh":
                        self.scene_handler.add_mesh(scene_object["name"], obj_pos, scene_object["size"],
                                                scene_object["mesh_file_name"], attach=bool(scene_object["attach"]),)
                    elif scene_object["type"] == "box":
                        self.scene_handler.add_box(scene_object["name"], obj_pos, scene_object["size"], 
                                                attach=True if "attach" in scene_object.keys() else False)
                    elif scene_object["type"] == "sphere":
                        self.scene_handler.add_sphere(scene_object["name"], obj_pos, scene_object["size"], 
                                                attach=True if "attach" in scene_object.keys() else False)
                    else:
                        raise Exception("Object type " + object["type"] + " is not currently supported")

    def get_closest_robot_to_point(self, controller_distances, robot_traversal_pos, priorised_robot=None):
        closest_robot_idx = None
        closest_controller_idx = None
        closest_distance = 0

        for distance_info in controller_distances:

            point_to_base = [robot_traversal_pos.x - distance_info["world_to_base"][0], 
                            robot_traversal_pos.y - distance_info["world_to_base"][1], 
                            robot_traversal_pos.z - distance_info["world_to_base"][2]]

            point_to_base = [point_to_base[0] - self.args.main_obj_position.x,
                             point_to_base[1] - self.args.main_obj_position.y,
                             point_to_base[2] - self.args.main_obj_position.z]

            distance = sqrt((point_to_base[0] * point_to_base[0]) + 
                            (point_to_base[1] * point_to_base[1]) + 
                            (point_to_base[2] * point_to_base[2]))

            """print(distance_info["robot_idx"])
            print(point_to_base)
            print(distance)
            print()"""

            # FIX TO BE GLOBAL ID
            if priorised_robot is not None and distance_info["robot_idx"] == priorised_robot and distance_info["reach"] is not None:

                if distance < distance_info["reach"] - 0.1:
                    return distance_info["controller_idx"], distance_info["robot_idx"] 

            if distance_info["reach"] is not None:
                distance -= distance_info["reach"]

            if distance < closest_distance or closest_robot_idx is None:
                closest_distance = distance
                closest_robot_idx = distance_info["robot_idx"]
                closest_controller_idx = distance_info["controller_idx"]

        return closest_controller_idx, closest_robot_idx  #0, 1

    def assign_robots_to_points(self, points: list, start_idx = 0, priorised_robot=None):
        controller_distances = []
        controller_points = {}

        for controller_idx, controller in enumerate(self.controllers):
            controller_points[str(controller_idx)] = {}

            for robot_idx, robot in enumerate(controller.robots):

                controller_distances.append(
                        {"controller_idx": controller_idx, 
                        "robot_idx": robot_idx, 
                        "world_to_base": controller.get_robot_transform("base", robot_idx)[0], 
                        "reach": robot.get_reach()}
                    )

                controller_points[str(controller_idx)][str(robot_idx)] = []
        
        for point_idx, point in enumerate(points):
            controller_idx, robot_idx = self.get_closest_robot_to_point(controller_distances, point, priorised_robot=priorised_robot)
            
            controller_points[str(controller_idx)][str(robot_idx)].append((point_idx + start_idx, point))

        return controller_points

    # Takes a snapshot from the current camera
    def take_snapshot(self, img_name: str, camera_handler, capture_depth=False, segment_foreground=None):

        # Capture current RGB image from ROS topic
        rgb_image = camera_handler.get_current_rgb_image()

        if rgb_image is None:
            return False

        captured = self.experiment_handler.write_img(img_name, rgb_image, "rgb")
        if not captured:
            return False

        self.frame_data[img_name]["file_path"] = self.experiment_handler.get_relative_file_paths(img_name, "rgb", "original")

        depth_img = None
        # If depth is set to be captured
        if capture_depth:
            depth_img = camera_handler.get_current_depth_image()

            if depth_img is not None:
                captured = self.experiment_handler.write_img(img_name, depth_img, "depth")

                if captured:
                    self.frame_data[img_name]["depth_file_path"] = self.experiment_handler.get_relative_file_paths(img_name, "depth", "original")

        # Attempt to remove background
        if segment_foreground is not None:
            segment_img, mask = self.foreground_segmenter.segment_foreground(rgb_image, depth_img=depth_img)

            captured = self.experiment_handler.write_img(img_name, segment_img, "segmented")
            
            if captured:
                self.frame_data[img_name]["segmented_file_path"] = self.experiment_handler.get_relative_file_paths(img_name, "segmented", "original")

            mask_name = img_name+".png"

            captured = self.experiment_handler.write_img(mask_name, mask, "mask")

            if captured:
                self.frame_data[img_name]["mask_file_path"] = self.experiment_handler.get_relative_file_paths(mask_name, "mask", "original")

        self.frame_data[img_name]["sharpness"] = camera_handler.calculate_img_sharpness(rgb_image)

        return True

    def calculate_real_transformation_matrix(self, translation, rotation):
        # Converts the robot transform into a format that will work with NeRF
        convertion_matrix = [[0,0,-1,0],    
                            [-1,0,0,0],
                            [0,1,0,0],
                            [0,0,0,1]]
        
        # Move transformation matrix to center around the object to be captured
        translation[0] -= self.object_center.x
        translation[1] -= self.object_center.y
        translation[2] -= self.object_center.z

        raw_trans_matrix = self.quartonian_handler.get_tranformation_matrix_from_transform(translation, rotation)
        
        trans_matrix = [[0 for i in range(4)] for j in range(4)]

        # Multiply the transformation matrix by the convertion matrix
        for i, row in enumerate(raw_trans_matrix):
            for j, _ in enumerate(row):
                for k, _ in enumerate(row):
                    trans_matrix[i][j] += raw_trans_matrix[i][k] * convertion_matrix[k][j]

                trans_matrix[i][j] = round(trans_matrix[i][j], 14) 

        return trans_matrix

    def calculate_fake_transformation_matrix(self, current_pos, normalise=True):

        original_point_copy = copy.deepcopy(current_pos)
        
        # Translate current point to be relative to the origin
        point_to_origin_norm = self.quartonian_handler.SubtractVectors(current_pos, self.object_center)

        if normalise:
            point_to_origin_norm = self.quartonian_handler.Normalize(point_to_origin_norm)

        transformed_point = Point(point_to_origin_norm.x, point_to_origin_norm.y, point_to_origin_norm.z)

        # Calculate quartonian for the point to have direction towards the origin
        rotation = self.quartonian_handler.QuaternionLookRotation(transformed_point, self.up)

        if normalise:
            translation_list = [transformed_point.x, transformed_point.y, transformed_point.z]
        else:
            translation_list = [original_point_copy.x, original_point_copy.y, original_point_copy.z]

        rotation_list = [rotation.x, rotation.y, rotation.z, rotation.w]

        # Convert rotation and translation to a transformation matrix
        trans_matrix = self.quartonian_handler.get_tranformation_matrix_from_transform(translation_list, rotation_list)

        return np.around(trans_matrix, decimals=14).tolist()

    def calculate_turntable_point_and_rotation(self, avg_base_pos, current_pos):

        avg_base_pos_2d = np.array([avg_base_pos.x, avg_base_pos.y])
        current_pos_2d = np.array([current_pos.x, current_pos.y])

        # Calculate the dot product of the two vectors
        dot_product = np.dot(avg_base_pos_2d, current_pos_2d)
        
        # Calculate the magnitudes of the vectors
        magnitude1 = np.linalg.norm(avg_base_pos_2d)
        magnitude2 = np.linalg.norm(current_pos_2d)
        
        # Calculate the cosine of the angle between the vectors
        cos_theta = dot_product / (magnitude1 * magnitude2)

        # Calculate the angle in radians
        angle = acos(cos_theta)

        d = ((avg_base_pos_2d[0]) * (current_pos_2d[1])) - ((avg_base_pos_2d[1]) * (current_pos_2d[0]))
        if d < 0:
            angle -= (pi * 2) 
        else:
            angle = -angle

        # Calculate the rotation matrix
        rotation_matrix = np.array([[cos(angle), -sin(angle)],
                                    [sin(angle), cos(angle)]])

        # Rotate the second vector to align with the first vector
        rotated_vector = np.dot(rotation_matrix, current_pos_2d)

        rotated_current_pos = np.append(rotated_vector, [current_pos.z])
        rotated_current_pos = self.convert_list_to_point(rotated_current_pos)

        return rotated_current_pos, -angle

    def capture_and_add_transform(self, point_idx, point, controller_idx, robot_idx):

        transform = None

        img_name = self.img_file_name(point_idx)

        self.frame_data[img_name] = {}

        """test_global_matrix = np.array([[ 9.99559230e-01,  2.90068032e-02,  1.62364856e-03, -8.78285484e-04],
                                        [-2.89655001e-02,  9.99335493e-01, -2.13510113e-02, -1.39397966e-02],
                                        [-2.25166800e-03,  2.12362162e-02,  9.99727829e-01,  1.03362132e-02],
                                        [ 0.00000000e+00,  0.00000000e+00,  0.00000000e+00,  1.00000000e+00]])"""

        test_global_matrix = np.array([[ 0.99997419, -0.00683568, -0.00570868, -0.0062643 ],
                                        [ 0.00687168,  0.99995663,  0.00373654,  0.01157257],
                                        [ 0.00566746, -0.00372804,  1.00000457, -0.02016575],
                                        [ 0.        ,  0.        ,  0.        ,  1.        ]])
                                        

        if not self.args.discard_img_capturing:

            captured = self.take_snapshot(img_name, self.controllers[int(controller_idx)].robots[robot_idx].get_camera_handler(), 
                                          capture_depth=self.args.capture_depth,
                                          segment_foreground= self.args.segment_method)

            if not captured:
                raise Exception(f"Unable to capture image: {img_name}")

            if self.args.use_fake_transforms or self.args.use_turntable:
                transform = self.calculate_fake_transformation_matrix(point, normalise=False)
            else:
                transform = self.controllers[int(controller_idx)].get_robot_transform("camera", robot_idx)

                transform = self.calculate_real_transformation_matrix(transform[0], transform[1])

            for i in range(3):
                transform[i][3] *= self.args.transform_scale

            if int(robot_idx) == 1:
                print("Multiplying Matrix")

                transform = np.matmul(np.array(transform), test_global_matrix).tolist()

            self.frame_data[img_name]["transform_matrix"] = transform
            self.frame_data[img_name]["is_train"] = self.is_train_img(point_idx)
            self.frame_data[img_name]["global_robot_id"] = self.controllers[int(controller_idx)].robots[robot_idx].get_global_id()

        self.trajectory_handler.pos_verdict(point_idx, True)

        self.db_handler.add_point_to_experiment(point_idx, True, self.frame_data[img_name])

    def copy_undistorted_to_transforms(self):
        
        self.experiment_handler.config_gaussian_dir(self.sfm_handler.dense_dir)

        for img_name, data in self.frame_data.items():
            data["undistorted_file_path"] = self.experiment_handler.get_relative_file_paths(img_name, "undistorted", "undistorted")

    # Calculate error as difference between transformed and goal matrix
    def calculate_error(self, transformed_matrix, goal_matrix):
        return np.linalg.norm(goal_matrix-transformed_matrix)

    # Optimises the optimised matrix for remapping original to global transforms
    def optimise_global_matrix(self, gm):
        # Reshape optimised matrix into standerdised format
        gm_transform = np.array(gm).reshape((3,4))
        gm_transform_full = np.concatenate((gm_transform, np.array([[0,0,0,1]])), axis=0)

        error = 0

        # Iterate through all transforms
        for img_name, frame in self.frame_data.items():

            original_transform = np.array(frame["transform_matrix"])
            adjusted_transform = np.array(frame["ba_transform_matrix"])

            # Calculate original matrix multiplied by the optimised matrix 
            transformed_matrix = np.matmul(adjusted_transform, gm_transform_full)

            # Add to overall error
            error += self.calculate_error(transformed_matrix, original_transform)

        # Logs the current moving average error
        if self.current_iter % self.log_rate == 0 and self.current_iter != 0:
            self.moving_avg_error = round(self.moving_avg_error/self.log_rate, 4)

            print(self.current_iter)
            print(self.moving_avg_error)
            print(gm_transform_full)
            print()
            
            self.error_logs.append(self.moving_avg_error)
            self.moving_avg_error = 0
        else:
            self.moving_avg_error += error

        self.current_iter += 1

        return error 

    def reset_transform_origin(self):

        original_centre_attention = np.array([0.0,0.0,0.0])

        for img_name, frame in self.frame_data.items():
            translation = np.array(frame["transform_matrix"])[:3, 3]

            original_centre_attention += translation

        original_centre_attention /= len(self.frame_data)

        current_origin = np.array([0.0,0.0,0.0])

        for img_name, frame in self.frame_data.items():
            translation = np.array(frame["ba_transform_matrix"])[:3, 3]

            current_origin += translation

        current_origin /= len(self.frame_data)
        
        print(current_origin)
        print(original_centre_attention)
        print()

        avg_radius = 0.

        for img_name, frame in self.frame_data.items():
            transform = np.array(frame["ba_transform_matrix"])

            transform[:3, 3] -= current_origin

            avg_radius += np.linalg.norm(transform[:3, 3])

            self.frame_data[img_name]["ba_transform_matrix"] = transform

        avg_radius /= len(self.frame_data)

        print(self.args.capture_radius)
        print(avg_radius)

        radius_scale = avg_radius/self.args.capture_radius

        print(radius_scale)

        for img_name, frame in self.frame_data.items():
            print(f"Recentered transform: {transform[:3, 3]}")

            transform = np.array(frame["ba_transform_matrix"])

            transform[:3, 3] /= radius_scale

            transform[:3, 3] += original_centre_attention

            self.frame_data[img_name]["ba_transform_matrix"] = transform.tolist()

        """# Max and mimumum values for the different translation values in the optimised transformation matrix 
        bound_range = 10

        # Sets rotation range to (-1 : 1) and translation range to (-bound : bound)
        lb = [-bound_range if (i+1) % 4 else -2 for i in range(12)]
        ub = [bound_range if (i+1) % 4 else 2 for i in range(12)]
        n_dims= 12
        size_pop = n_dims * 5

        self.log_rate = 200
        self.moving_avg_error = 0
        self.error_logs = []
        self.current_iter = 0
        self.moving_avg_error = 0

        # Performs Differential Evolution optimisation algorithm
        de = DE(func=self.optimise_global_matrix, n_dim=12, size_pop=size_pop, max_iter=250,
                lb=lb, ub=ub)

        # Return the optimised matrix
        best_matrix, best_err = de.run()

        # Convert final optimised matrix into standerdised format 
        final_gm_transform = np.array(best_matrix).reshape((3,4))

        gm_transform_full = np.concatenate((final_gm_transform, np.array([[0,0,0,1]])), axis=0)

        print("Opimised mapping matrix")
        print(gm_transform_full)

        for img_name, frame in self.frame_data.items():
            adjusted_transform = frame["ba_transform_matrix"]

            new_transform = np.matmul(adjusted_transform, gm_transform_full)

            self.frame_data[img_name]["ba_transform_matrix"] = new_transform.tolist()"""


    def adjust_captured_transforms(self):

        sfm_settings = {"sfm_type": self.args.sfm_type, "feature_mode": self.args.feature_mode, "undistort_imgs": self.args.undistort_imgs}

        transforms = []

        for _, frame in self.frame_data.items():
            transforms.append({"file_path": frame["file_path"], "transform_matrix": frame["transform_matrix"]})
        
        if self.args.img_pair_path:
            self.sfm_handler.set_img_name_pair_path(self.args.img_pair_path)
        else:
            if self.args.use_relative_img_pairs:
                point_pairs = self.trajectory_handler.get_adjacent_successful_positions(img_pair_range=self.args.img_pair_range)
                img_pairs = {}

                for point, adjacent_points in point_pairs.items():
                    img_pairs[self.img_file_name(point)] = [self.img_file_name(adjacent_point) for adjacent_point in adjacent_points]

                self.sfm_handler.generate_img_key_matches_adjacent(img_pairs)
                
            else:
                self.sfm_handler.generate_img_name_matches_linear(img_pair_range=self.args.img_pair_range)

        image_cams = {}
        for frame, info in self.frame_data.items():
            image_cams[frame] = info["global_robot_id"]

        if self.args.extend_reconstruction_per_robot and len(self.camera_properties) > 1:
            print("Adjusting transforms using sfm and image registration")
            adjusted_frames = self.sfm_handler.adjust_transforms_multiple(transforms, camera_info=self.camera_properties, sfm_settings=sfm_settings, image_cams=image_cams)

        else:
            print("Adjusting transforms using sfm and global bundle adjustment")
            adjusted_frames = self.sfm_handler.adjust_transforms(transforms, camera_info=self.camera_properties, sfm_settings=sfm_settings, image_cams=image_cams)

        for img_name, trans_data in self.frame_data.items():
            if img_name in adjusted_frames:
                trans_data["ba_transform_matrix"] = adjusted_frames[img_name]
            else:
                print(f"WARNING: SfM failed to find a position for {img_name}!")
        
        #self.reset_transform_origin()

        if self.args.undistort_imgs:
            self.copy_undistorted_to_transforms()

            if self.foreground_segmenter:
                print("Segmenting undistorted images")

                for i, img_name in enumerate(adjusted_frames):
                    print(i)

                    try:
                        undistorted_img = self.experiment_handler.read_img(img_name, "undistorted")
                        mask_img = self.experiment_handler.read_img(img_name+".png", "mask")
                    except Exception:
                        continue

                    segment_img, _ = self.foreground_segmenter.segment_foreground(undistorted_img)

                    self.experiment_handler.write_img(img_name, segment_img, "undistorted_segmented")



    def run_models():
        for model in self.reconstruction_models:
            current_model = get_reconstruction_handler(model, self.args.log_dir, self.args.experiment_name, 
                                                       render_camera_path=self.args.render_camera_path)

            for transform in self.reconstruct_transforms:
                current_model.set_transform_type(transform)

                current_model.run_model()

    def generate_transform_file(self, img_type, transform_type, transform_file_type,
                                      include_mask=False, include_depth=False):
        
        train_transforms = {"frames": []}
        test_transforms = {"frames": []}
        all_transforms = {"frames": []}

        include_camera_properties = self.camera_properties is not None

        use_individual_cams = False
        if include_camera_properties:
            use_individual_cams = (len(self.camera_properties) != 1)

        if include_camera_properties and not use_individual_cams:
            train_transforms = {**list(self.camera_properties.values())[0], **train_transforms}
            all_transforms = {**list(self.camera_properties.values())[0], **all_transforms}
            if self.args.test_incrementation > 0:
                test_transforms = {**list(self.camera_properties.values())[0], **test_transforms}
        
        aabb_scale = self.aabb_scale(self.args.transform_scale)
        train_transforms["aabb_scale"] = aabb_scale
        all_transforms["aabb_scale"] = aabb_scale
        if self.args.test_incrementation > 0:
            test_transforms["aabb_scale"] = aabb_scale

        # Order transform file in descending file name order
        self.frame_data = dict(sorted(self.frame_data.items(), key=lambda item: item[0]))

        for img_name, frame_info in self.frame_data.items():

            new_frame = {}
            if transform_type == "original":
                if "transform_matrix" in frame_info:
                    new_frame["transform_matrix"] = frame_info["transform_matrix"]
                else:
                    print(f"WARNING: Unable to add original transform to image transform {img_name}")
                    continue

            elif transform_type == "adjusted":
                if "ba_transform_matrix" in frame_info:
                    new_frame["transform_matrix"] = frame_info["ba_transform_matrix"]
                else:
                    print(f"WARNING: Unable to add adjusted transform to image transform {img_name}")
                    continue

            if img_type == "rgb":
                if "file_path" in frame_info:
                    new_frame["file_path"] = frame_info["file_path"]
                else:
                    print(f"WARNING: Unable to add rgb file path to image transform {img_name}")
            elif img_type == "segmented":
                if "segmented_file_path" in frame_info:
                    new_frame["file_path"] = frame_info["segmented_file_path"]
                else:
                    print(f"WARNING: Unable to add segmented file path to image transform {img_name}")
            elif img_type == "undistorted":
                new_frame["file_path"] = new_frame["undistorted_file_path"]

            if include_mask and "mask_file_path" in frame_info:
                new_frame["mask_path"] = frame_info["mask_file_path"]

            if include_depth and "depth_file_path" in frame_info:
                new_frame["depth_file_path"] = frame_info["depth_file_path"]

            if "sharpness" in frame_info:
                new_frame["sharpness"] = frame_info["sharpness"]

            if include_camera_properties and use_individual_cams:
                frame_camera_properties = self.camera_properties[frame_info["global_robot_id"]]

                new_frame = {**frame_camera_properties, **new_frame}
            
            all_transforms["frames"].append(new_frame)

            if frame_info["is_train"]:
                train_transforms["frames"].append(new_frame)
            else:
                test_transforms["frames"].append(new_frame)
        
        self.experiment_handler.write_transforms_file(all_transforms, transform_file_type, "transforms.json")
        self.experiment_handler.write_transforms_file(train_transforms, transform_file_type, "transforms_train.json")
        self.experiment_handler.write_transforms_file(test_transforms, transform_file_type, "transforms_test.json")

    def save_transforms(self):

        self.generate_transform_file("rgb", "original", "original")

        priority_transform_type = "original" if not self.args.adjust_transforms else "adjusted"

        if self.args.adjust_transforms:
            self.generate_transform_file("rgb", priority_transform_type, "adjusted")

        if self.args.segment_method is not None:
            self.generate_transform_file("rgb", priority_transform_type, "mask", include_mask=True)
            self.generate_transform_file("segmented", priority_transform_type, "segmented")
        
        if self.args.capture_depth:
            self.generate_transform_file("rgb", priority_transform_type, "depth", include_depth=True)

            if self.args.segment_method is not None:
                self.generate_transform_file("rgb", priority_transform_type, "depth_mask", include_depth=True, include_mask=True)
        
        #if self.args.adjust_transforms and self.args.undistort_imgs:
        #    self.generate_transform_file("undistorted", priority_transform_type, "undistorted", include_depth=True, include_mask=True)

    def load_db_point(self, point_idx: int, camera_handler):

        current_point = self.db_handler.get_point_with_num(point_idx)

        if current_point is not None and current_point[2] == 1:
            
            print(f"Point {point_idx} has already been traveresed previously, skipping...")
            print()

            img_name = self.img_file_name(point_idx)

            self.frame_data[img_name] = {}

            self.frame_data[img_name]["file_path"] = self.experiment_handler.get_relative_file_paths(img_name, "rgb", "original")

            current_transform = np.array([])
            if current_point[3] is not None and len(current_point[3]) != 0:
                current_transform = np.array(json.loads(current_point[3]))

            self.frame_data[img_name]["transform_matrix"] = current_transform.tolist() #json.loads(current_point[3])

            self.frame_data[img_name]["is_train"] = self.is_train_img(point_idx)

            if current_point[4] == 1:
                self.frame_data[img_name]["depth_file_path"] = self.experiment_handler.get_relative_file_paths(img_name, "depth", "original")
            if current_point[5] == 1:
                self.frame_data[img_name]["mask_file_path"] = self.experiment_handler.get_relative_file_paths(img_name+".png", "mask", "original")
            if current_point[6] == 1:
                self.frame_data[img_name]["segmented_file_path"] = self.experiment_handler.get_relative_file_paths(img_name, "segmented", "original")  
            if current_point[7] is not None:
                self.frame_data[img_name]["global_robot_id"] = current_point[7]

            file_path = os.path.join(self.experiment_handler.get_experiment_dir(), self.experiment_handler.get_img_dir("rgb"), img_name)

            if camera_handler is not None:
                self.frame_data[img_name]["sharpness"] = camera_handler.calculate_img_sharpness(camera_handler.load_img(file_path))

            return True

        return False

    def traverse_points(self):
        turntable_rotations_and_real_points = {}
        
        avg_base_pos = self.controllers[0].get_average_base_position()
        avg_base_pos[0] += self.args.main_obj_position.x
        avg_base_pos[1] += self.args.main_obj_position.y

        avg_base_pos = [0.7, 0, 0]

        avg_base_pos = self.convert_list_to_point(avg_base_pos)

        for sector_pos_idx, sector_positions in self.trajectory_handler.get_next_sector():

            print(f"Beginning new sector {sector_pos_idx}")

            if self.args.use_turntable:
                rotation_points = []

                for point_idx, point in enumerate(sector_positions):
                    relative_point, rotation_angle = self.calculate_turntable_point_and_rotation(avg_base_pos, point)

                    if self.start_rad is None:
                        self.start_rad = rotation_angle
                    else:
                        if rotation_angle < self.start_rad:
                            rotation_angle += (pi * 2)

                    turntable_rotations_and_real_points[sector_pos_idx + point_idx] = (point, rotation_angle)
                    rotation_points.append(relative_point)

                points_per_controller = self.assign_robots_to_points(rotation_points, start_idx=sector_pos_idx, priorised_robot=self.args.priorised_robot)

            else:
                points_per_controller = self.assign_robots_to_points(sector_positions, start_idx=sector_pos_idx, priorised_robot=self.args.priorised_robot)

            for controller_idx, robot_points in points_per_controller.items():
                has_points = False

                traversed_points = []

                while True:

                    current_points = [] 

                    for robot_idx in range(len(robot_points.keys())):

                        for (point_idx, point) in robot_points[str(robot_idx)]:

                            if point_idx not in traversed_points:

                                print(f"Adding point {point_idx} using robot {robot_idx} to movement plan")

                                traversed_points.append(point_idx)

                                if self.args.continue_experiment:
                                    if self.load_db_point(point_idx, self.controllers[0].robots[0].get_camera_handler()):
                                        print(f"Point {point_idx} has aleady been traversed (skipping)")

                                        self.trajectory_handler.pos_verdict(point_idx, True)

                                        continue
                                    #else:
                                    #    self.trajectory_handler.pos_verdict(point_idx, False)
                                    #    continue

                                has_points = True

                                quartonian = self.quartonian_handler.QuaternionLookRotation(self.quartonian_handler.SubtractVectors(self.object_center, point), self.up)

                                self.controllers[int(controller_idx)].add_position_and_orientation(point, quartonian, int(robot_idx))

                                if not self.args.parallelise_robots:

                                    print("Attempting to move robot")

                                    success = self.controllers[int(controller_idx)].execute_plan(avoid_self_capture_paths=self.args.avoid_self_capture_paths if not self.args.discard_img_capturing else False)

                                    if success:
                                        print(f"Successfully moved to point {point_idx}")

                                        if self.args.use_turntable:
                                            rotation_angle = turntable_rotations_and_real_points[point_idx][1]

                                            print(f"Rotating to angle {round(rotation_angle, 3)}")

                                            self.turntable_handler.rotate_to_pos(rotation_angle, use_rads=True)
                                        else:
                                            time.sleep(0.4)

                                        real_point = point if not self.args.use_turntable else turntable_rotations_and_real_points[point_idx][0]

                                        self.capture_and_add_transform(point_idx, real_point, int(controller_idx), robot_idx)
                                    else:
                                        self.trajectory_handler.pos_verdict(point_idx, False)

                                        print("Failed to move point!")

                                    print()
                                
                                else:
                                    current_points.append((point_idx, point, robot_idx))

                                break

                    print()

                    if not has_points:
                        break
                    
                    if self.args.parallelise_robots:

                        success = self.controllers[int(controller_idx)].execute_plan(avoid_self_capture_paths=self.args.avoid_self_capture_paths if not self.args.discard_img_capturing else False)

                        for (point_idx, point, robot_idx) in current_points:

                            if success:
                                print(f"Successfully moved to point {point_idx}")

                                if self.args.use_turntable:
                                    rotation_angle = turntable_rotations_and_real_points[point_idx][1]
                                    
                                    print(f"Rotating to angle {round(rotation_angle, 3)}")
                                    
                                    self.turntable_handler.rotate_to_pos(rotation_angle, use_rads=True)

                                real_point = point if not self.args.use_turntable else turntable_rotations_and_real_points[point_idx][0]

                                self.capture_and_add_transform(point_idx, real_point, int(controller_idx), robot_idx)

                            else:
                                print(f"Failed to move to point {point_idx}")
                                print()

                                if len(current_points) > 1:
                                    print(f"Attempting to move serially with robot {robot_idx}")

                                    quartonian = self.quartonian_handler.QuaternionLookRotation(self.quartonian_handler.SubtractVectors(self.object_center, point), self.up)

                                    self.controllers[int(controller_idx)].add_position_and_orientation(point, quartonian, int(robot_idx))

                                    serial_success = self.controllers[int(controller_idx)].execute_plan(avoid_self_capture_paths=self.args.avoid_self_capture_paths if not self.args.discard_img_capturing else False)

                                    if serial_success: 
                                        print(f"Successfully moved to point {point_idx}")

                                        if self.args.use_turntable:
                                            rotation_angle = turntable_rotations_and_real_points[point_idx][1]
                                            
                                            print(f"Rotating to angle {round(rotation_angle, 3)}")

                                            self.turntable_handler.rotate_to_pos(rotation_angle, use_rads=True)

                                        real_point = point if not self.args.use_turntable else turntable_rotations_and_real_points[point_idx][0]

                                        self.capture_and_add_transform(point_idx, real_point, int(controller_idx), robot_idx)
                                    else:

                                        self.trajectory_handler.pos_verdict(point_idx, False)

                                        print("Failed to move to point!")
                                
                                else:
                                    self.trajectory_handler.pos_verdict(point_idx, False)

                        print()
                            
                    has_points = False

    def capture_initial_calibration(self, main_robot_idx=1):
        print("Capturing calibration target")

        offset = 0.55
        crosshair_height = 0.1

        cross_hair = Point(self.args.main_obj_position.x, self.args.main_obj_position.y, self.args.main_obj_position.z+crosshair_height)

        transformed_point = Point(self.args.main_obj_position.x+offset, self.args.main_obj_position.y, self.args.main_obj_position.z+crosshair_height)

        quartonian = self.quartonian_handler.QuaternionLookRotation(self.quartonian_handler.SubtractVectors(cross_hair, transformed_point), self.up)

        self.controllers[0].add_position_and_orientation(transformed_point, quartonian, main_robot_idx)

        if self.args.use_turntable:
            print("Resetting Turntable")
            self.turntable_handler.rotate_to_pos(0, use_rads=True)

        success = self.controllers[0].execute_plan(avoid_self_capture_paths=False)

        if not success:
            print("Could not move to crosshair capture position")
            exit(0)

        camera_handler = self.controllers[0].robots[main_robot_idx].get_camera_handler()

        camera_handler.crop_h = 300
        camera_handler.crop_w = 300

        rgb_image = camera_handler.get_current_rgb_image()

        camera_properties = camera_handler.get_camera_properties()

        camera_handler.crop_h = camera_properties["h"]
        camera_handler.crop_w = camera_properties["w"]

        cv2.imwrite(os.path.join(self.args.log_dir, self.args.experiment_name, f"Crosshair.png"), rgb_image)

        print("Captured crosshair target")
        print()


    def calibrate_turntable_centre(self, main_robot_idx=0, cam_height=0.7):
        print("Please place a checkerboard (or similar pattern) on your turntable")
        x = input(">")

        offset = 0.000000001

        transformed_point = Point(self.args.main_obj_position.x + offset, self.args.main_obj_position.y, self.args.main_obj_position.z+cam_height)

        quartonian = self.quartonian_handler.QuaternionLookRotation(self.quartonian_handler.SubtractVectors(self.args.main_obj_position, transformed_point), self.up)

        self.controllers[main_robot_idx].add_position_and_orientation(transformed_point, quartonian, main_robot_idx)

        success = self.controllers[main_robot_idx].execute_plan(avoid_self_capture_paths=False)

        #if not success:
        #    print("Failed to move to position")
        #    exit(0)

        camera_handler = self.controllers[main_robot_idx].robots[0].get_camera_handler()

        if self.turntable_handler is None:
            self.turntable_handler = get_turntable_handler(self.args.turntable_handler, 
                                                        self.args.turntable_connection_port)


        num_steps = 16
        for i in range(num_steps):
            rads = ((pi*2)/num_steps) * i

            self.turntable_handler.rotate_to_pos(rads, use_rads=True)

            camera_handler.crop_h = 100
            camera_handler.crop_w = 100

            rgb_image = camera_handler.get_current_rgb_image()

            cv2.imwrite(os.path.join(self.args.log_dir, self.args.experiment_name, f"{i}.png"), rgb_image)

        self.turntable_handler.rotate_to_pos(0, use_rads=True)

    def run(self):
        print("Beginning view capturing")
        print()

        self.trajectory_handler.calculate_sphere_points(self.object_center, self.args.capture_radius, 
                                                        rings=self.args.rings, sectors=self.args.sectors, 
                                                        aabb=self.args.aabb if len(self.args.aabb) != 0 else None)

        if self.args.visualise:
            self.trajectory_handler.visualise_predicted_valid_points(save=self.args.save_fig)

        #self.calibrate_turntable_centre()

        #exit(0)

        #if not (self.experiment_handler.does_experiment_exist() and self.args.continue_experiment):

        #if not os.path.exists(os.path.join(self.experiment_handler.experiment_dir_path, "Crosshair.png")):
        #    self.capture_initial_calibration()

        #exit(0)

        self.traverse_points()

        if self.args.retry_failed_pos:
            print("Retrying previously failed positions!")

            self.args.parallelise_robots = False

            self.trajectory_handler.reset_traversed_positions()

            self.traverse_points()

        if self.args.visualise:
            self.trajectory_handler.visualise_traversed_points(save=self.args.save_fig)
        
        if self.args.adjust_transforms:
            print("Adjusting Transforms")
            print()

            self.adjust_captured_transforms()

        print("Saving transforms")
        print()
        self.save_transforms()

        if self.args.reconstruction_models != []:
            print("Running models")
            print()
            self.run_models()

        if self.args.use_turntable:
            print("Resetting Turntable")
            self.turntable_handler.rotate_to_pos(self.start_rad, use_rads=True)

        print("Finished")
        print()

# Configues the arguments added by the user
def config_parser():

    parser = configargparse.ArgumentParser()

    # Experiment Settings
    parser.add_argument("--config", is_config_file=True, help="Path to configuration file containing a list of configuration arguments")
    parser.add_argument("--log_dir", type=str, default="", help="Path to directory to store log information (images and transforms) after running an experiment")
    parser.add_argument("--experiment_name", type=str, required=True, help="Name of this current experiment")
    parser.add_argument("--visualise", action="store_true", default=False, help="Set to generate 3D scatter diagrams for predicted and traversed positions")
    parser.add_argument("--save_fig", action="store_true", default=False, help="Set to save the generated diagrams to a PNG image")
    parser.add_argument("--continue_experiment", action="store_true", default=False, 
                        help="Set to continue an experiment with the same name as the ’experiment name’ argument. \
                            This means that already traversed positions will be skipped, and failed positions will be retried")
    parser.add_argument("--replace_stored_experiment", action="store_true", default=False, 
                        help="Set to remove all stored information about a previous experiment with the same name as the ’experiment name’ argument and restart the experiment")

    # Reconstruction Methods
    parser.add_argument("--reconstruction_models", action="append", default=[], help="")
    parser.add_argument("--reconstruct_transforms", action="append", default=[], help="")
    parser.add_argument("--render_camera_path", type=str, default="", help="")

    # Robot Settings
    parser.add_argument("--robot_settings_path", type=str, default="" , required=True, help="")
    parser.add_argument("--robot_handler_type", type=str, default="moveit", help="")
    parser.add_argument("--parallelise_robots", type=bool, default=True, help="")
    parser.add_argument("--avoid_self_capture_paths", type=bool, default=True, help="")
    parser.add_argument("--priorised_robot", type=int, default=None, help="")

    # Scene and Object Properties
    parser.add_argument("--scene_handler_type", type=str, default="moveit", help="")
    parser.add_argument("--main_obj_position", type=float, action="append", default=[],
                        help="The (x,y,z) position of the object relative to the centre of the robot’s WCS- which is traditionally the base of the robot")
    parser.add_argument("--main_obj_size", type=float,  action="append", default=[], 
                        help="The size of the object (x,y,z) in metres. Used for avoiding collisions in the robot path planning")
    parser.add_argument("--auto_add_obj_stand", action="store_true", default=False,
                        help="Set to automatically add a stand for the main object in the path planning (avoids collisions)")
    parser.add_argument("--obj_stand_thickness", type=float, default=0.35,
                        help="The thickness (x,y) of the generated stand in meters, this should match the world measurements of the real stand")
    parser.add_argument("--auto_add_floor", action="store_true", default=False,
                        help="")
    parser.add_argument("--auto_add_ceiling", action="store_true", default=False,
                        help="")
    parser.add_argument("--ceiling_height", type=float, default=2.7,
                        help="")
    parser.add_argument("--scene_objs", type=str, default=None, help="Path to a scene file that loads different objects into the path planning environment. \
                        Each specified object must declare a valid shape type (cube, sphere, mesh) and dimensions")
    parser.add_argument("--transform_scale", type=float, default=1, help="The scale to alter the transform positions")
    
    # View Generation and Camera Handling
    parser.add_argument("--capture_radius", type=float, required=True,
                        help="The distance from the center of the main object to all of the generated camera positions")
    parser.add_argument("--rings", type=int, default=7, 
                         help="The number of rings used for generating the positions in the sphere")
    parser.add_argument("--sectors", type=int, default=14, 
                         help="The number of sectors used for generating the positions in the sphere")
    parser.add_argument("--aabb", type=float, action="append", default=[], help="A axis-aligned bounding box, with (x,y,z) values between 0 and 1, that is used to filter \
                        points that are not positioned inside of this volume during view generation")

    parser.add_argument("--camera_handler_type", type=str, default="ros", help="")
    parser.add_argument("--capture_depth", action="store_true", default=False,
                        help="Set to capture depth values from the camera depth topic (only set if a depth camera is being used)")
    parser.add_argument("--crop_width", type=int, default=None,
                        help="The number of pixels to crop the image width. Defaut does not crop the image.")
    parser.add_argument("--crop_height", type=int, default=None,
                        help="The number of pixels to crop the image height. Defaut does not crop the image.")

    parser.add_argument("--segment_method", default=None,
                        help="Method used to segment the foreground from the background")

    # Movement Handling
    parser.add_argument("--planning_time", type=float, default=2.0, 
                         help="The maximum number of seconds to take for each planning attempt before failing")
    parser.add_argument("--num_move_attempts", type=int, default=3, 
                         help="The number of attempts to move to specific position before failing and beginning next \
                            position movement attempt")
    parser.add_argument("--planning_algorithm", type=str, default=None, 
                        help="The planning algorithm to use for calculating trajectories between positions (default will \
                            use the currently set moveit planning algorithm)")
    parser.add_argument("--retry_failed_pos", action="store_true", default=False,
                         help="Set to reattempt to move to previously failed positions after movement to all points have been attempted.")
    parser.add_argument("--discard_robot_movement", action="store_true", default=False,
                        help="Skips movement of the robot and assumes the move failed (use for debugging)")

    # Turntable Usage
    parser.add_argument("--use_turntable", action="store_true", default=False,
                        help="Set to use a turntable to rotate the object rather than moving the arm to positions \
                             around the object (drastically changes point and transform calculations)")
    parser.add_argument("--turntable_connection_port", type=str, default="/dev/ttyUSB0", help="")
    parser.add_argument("--turntable_handler", type=str, default="zaber", help="")

    # Robot RC Settings (default for UR5 robot)
    parser.add_argument("--restricted_x", type=float, default=0.05, help="Minumum X value for a self-collision")
    parser.add_argument("--restricted_y", type=float, default=0.05, help="Minumum Y value for a self-collision")
    parser.add_argument("--restricted_z", type=float, default=-0.1, help="Minumum Z value for a self-collision")

    # Dataset Generation
    parser.add_argument("--use_fake_transforms", default=False, action="store_true",
                        help="Set to use pre-calculated transforms, \
                              rather than the transforms generated by the robot in the TF ROS topic")
    parser.add_argument("--transform_joint", type=str, default="camera_color_frame",
                        help="The joint to use for generating the transformation from the robot base to the camera.")
    parser.add_argument("--test_incrementation", type=int, default=0, 
                        help="The number of training images per test image \
                             (0 will not assign any images as test images in the dataset)")
    parser.add_argument("--test_position_offset", type=int, default=0, 
                        help="The amount to offset each of the test positions (between 0m and 0.5m)")
    parser.add_argument("--test_angle_offset", type=int, default=0, 
                        help="The amount to offset each of the test orientations (between 0 and 30 degrees)")
    parser.add_argument("--discard_img_capturing", default=False, action="store_true",
                        help="If set, does not attempt to capture images or \
                              use any camera settings (use for debugging)")

    # Adjust Transforms
    parser.add_argument("--sfm_package", type=str, default="COLMAP",
                        help="The SfM package to use for adjusting the transforms (either COLMAP or MVG)")
    parser.add_argument("--adjust_transforms", type=bool, default=True,
                        help="If set, sfm techniques will be used to optimise the given transforms using OpenMVG")
    parser.add_argument("--sfm_type", type=str, default="RECONSTRUCTION",
                        help="The SfM algorithm to use for adjusting the transforms")
    parser.add_argument("--feature_mode", type=str, default="HIGH",
                        help="The extensiveness of the feature extraction step")
    parser.add_argument("--img_pair_path", type=str, default=None,
                        help="The path to an img pair txt file that can be used in feature matching")
    parser.add_argument("--use_relative_img_pairs", type=bool, default=True,
                        help="The range of adjacent images to include for each image in relative image pair generation")
    parser.add_argument("--img_pair_range", type=int, default=3,
                        help="The range of adjacent images to include for each image in relative image pair generation")
    parser.add_argument("--undistort_imgs", type=bool, default=True,
                        help="Undistorts the images after camera pose adjustment (COLMAP only)")
    parser.add_argument("--use_mask_in_sfm", type=bool, default=True,
                        help="Only extract features after applying the mask (recommended for turntable experiments with background)")
    parser.add_argument("--extend_reconstruction_per_robot", type=bool, default=False)

    args = parser.parse_args()

    args.experiment_name = args.experiment_name.replace(" ", "_").lower()

    if args.main_obj_position == []:
        args.main_obj_position == [0.0, 0.0, 0.0]

    print("Experiment: " + args.experiment_name)
    print()

    return args

def main():
    args = config_parser()

    view_capture = ViewCapture(args)

    view_capture.run()

if __name__ == '__main__':
    main()
