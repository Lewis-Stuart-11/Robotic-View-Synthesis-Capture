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

class Calibrator():
    def __init__(self, args):
        self.quartonian_handler = QuartonianHandler()
        
        self.turntable_handler = get_turntable_handler(args.turntable_handler, 
                                                       args.turntable_connection_port) if args.use_turntable else None 

        self.controllers = []
        self.configure_robot_controllers()

        self.up = Point(0.0, 0.0, 1.0)

        self.args = args

        self.calibration_dir = os.path.join(args.log_dir, "calibration")



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

    def calibrate_turntable_centre(self, main_robot_idx=0, cam_height=0.8):
        print("Please place a checkerboard (or similar pattern) on your turntable")
        x = input(">")

        offset = 0.000000001

        transformed_point = Point(self.args.main_obj_position.x + offset, self.args.main_obj_position.y, self.args.main_obj_position.z+cam_height)

        quartonian = self.quartonian_handler.QuaternionLookRotation(quartonian_handler.SubtractVectors(self.args.main_obj_position, transformed_point), self.up)

        self.controllers[main_robot_idx].add_position_and_orientation(transformed_point, quartonian, main_robot_idx)

        success = self.controllers[main_robot_idx].execute_plan(avoid_self_capture_paths=False)

        #if not success:
        #    print("Failed to move to position")
        #    exit(0)

        num_steps = 16
        for i in range(num_steps):
            rads = ((pi*2)/num_steps) * i

            self.turntable_handler.rotate_to_pos(rads, use_rads=True)

            camera_handler.crop_h = 50
            camera_handler.crop_w = 50

            rgb_image = camera_handler.get_current_rgb_image()

            cv2.imwrite(os.path.join(self.args.log_dir, self.args.experiment_name, f"{i}.png"), rgb_image)

        self.turntable_handler.rotate_to_pos(0, use_rads=True)

    def calculate_multi_robot_error():
        pass

    def detect_plant_properties():
        pass






# Configues the arguments added by the user
def config_parser():

    parser = configargparse.ArgumentParser()

    # Experiment Settings
    parser.add_argument("--config", is_config_file=True, help="Path to configuration file containing a list of configuration arguments")

    parser.add_argument("--calibrate_turntable", type=bool, default=True)
    parser.add_argument("--calculate_multi_robot_error", type=bool, default=True)
    parser.add_argument("--detect_plant_properties", type=bool, default=True)

    return args

def main():
    args = config_parser()

    calibrator = Calibrator()

    if args.calibrate_turntable:
        calibrator.calibrate_turntable()

    if args.calculate_multi_robot_error:
        calibrator.calculate_multi_robot_error()
    
    if args.detect_plant_properties:
        calibrate.detect_plant_properties()

if __name__ == '__main__':
    main()
