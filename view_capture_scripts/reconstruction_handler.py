import os
import json
import copy
import configargparse
import pathlib

from abc import ABC, abstractmethod

class Reconstruction_Model_Handler(ABC):

    @abstractmethod
    def train_model(self, transforms_path, output_dir, experiment_name="nerf"):
        pass

    @abstractmethod
    def calculate_pointcloud(self, checkpoint, output_dir):
        pass
    
    @abstractmethod
    def render_model(self, checkpoint, output_path, render_camera_path=""):
        pass

    @abstractmethod
    def evaluate_results(self, checkpoint, output_path):
        pass

    @abstractmethod
    def run_model(self, render=True, evaluate=True, generate_pointcloud=True):
        pass

class NeRFStudio_Handler(Reconstruction_Model_Handler):
    def __init__(self, log_dir, experiment_name, transform_type=None, render_camera_path="", timestamp=1):
        self.experiment_path = os.path.join(log_dir, experiment_name)
        self.experiment_name = experiment_name

        self.render_camera_path = render_camera_path

        if transform_type is not None:
            self.set_transform_type(transform_type)

    def set_transform_type(self, transform_type):

        self.transform_type = transform_type
        self.transform_path = os.path.join(self.experiment_path, "transforms", self.transform_type, "transforms.json")

        self.setup_directory()

    def setup_directory(self):
        self.nerf_dir = os.path.join(self.experiment_path, "nerfacto")
        if not os.path.exists(self.nerf_dir):
            os.mkdir(self.nerf_dir)

        self.exports_dir =  os.path.join(self.experiment_path, "exports")
        if not os.path.exists(self.exports_dir):
            os.mkdir(self.exports_dir)
            
        self.pointcloud_dir = os.path.join(self.exports_dir, "pointcloud")
        if not os.path.exists(self.pointcloud_dir):
            os.mkdir(self.pointcloud_dir)

        self.render_dir = os.path.join(self.exports_dir, "render")
        if not os.path.exists(self.render_dir):
            os.mkdir(self.render_dir)

        self.timestamp = 1

        self.render_path = os.path.join(self.render_dir, self.transform_type+".mp4")
        self.checkpoint_path = os.path.join(self.nerf_dir, self.transform_type, "nerfacto", str(self.timestamp), "config.yml")
        self.evaluation_path = os.path.join(self.nerf_dir, self.transform_type, "nerfacto", str(self.timestamp), "evaluation.json")

    def train_model(self, transforms_path, output_dir, experiment_name="nerf"):
        return f"ns-train nerfacto --data {transforms_path} --experiment-name {experiment_name} --output-dir {output_dir} --logging.steps-per-log 5000 \
                                 --pipeline.model.background-color random --pipeline.model.near-plane 0.01 --pipeline.model.far-plane 50.0 --viewer.quit-on-train-completion True \
                                 --timestamp {str(self.timestamp)} nerfstudio-data --eval-mode filename"

    def calculate_pointcloud(self, checkpoint, output_dir):
        return f"ns-export pointcloud --load-config {checkpoint} --output-dir {output_dir} --num-points 1000000 --remove-outliers True \
                                  --normal-method open3d --use-bounding-box True --bounding-box-min -1.0 -1.0 -1.0 --bounding-box-max 3.0 3.0 3.0"
    
    def render_model(self, checkpoint, output_path, camera_path=""):
        return f"ns-render camera-path --load-config {checkpoint} --camera-path-filename {camera_path} --output-path {output_path}"

    def evaluate_results(self, checkpoint, output_path):
        return f"ns-eval --load-config {checkpoint} --output-path {output_path}"

    def run_model(self, render=True, evaluate=True, generate_pointcloud=True):

        if not os.path.exists(self.checkpoint_path):
    
            print("Training NeRF: ")
            
            err = os.system(self.train_model(self.transform_path, self.nerf_dir, experiment_name=self.experiment_name))

            if err != 0:
                print("ERROR TRAINING NERF")
                return False
        else:
            print("Skipping NeRF Training")

        if not os.path.exists(self.evaluation_path) and evaluate:
            print("Evaluating Model: ")

            err = os.system(self.evaluate(self.checkpoint_path, self.evaluation_path))

            if err != 0:
                print("ERROR EVALUATING NERF")
        else:
            print("Skipping NeRF Evaluation")

        if not os.path.exists(self.render_path) and render:
            print("Rendering Video: ")

            err = os.system(self.render_model(self.checkpoint_path, self.render_path, camera_path=self.camera_path))

            if err != 0:
                print("ERROR RENDERING NERF")
        else:
            print("Skipping NeRF Rendering")

        if not os.path.exists(os.path.join(self.pointcloud_dir, f"{self.transform_type}.ply")) and generate_pointcloud:
            print("Generating Pointcloud: ")

            os.system(self.calculate_pointcloud(self.checkpoint_path, self.pointcloud_dir))

            os.rename(os.path.join(self.pointcloud_dir, "point_cloud.ply"), os.path.join(self.pointcloud_dir, f"{self.transform_type}.ply"))
        else:
            print("Skipping Pointcloud Generation")


class Gaussian_Handler(Reconstruction_Model_Handler):
    def __init__(self, log_dir, experiment_name, transform_type="", 
                 gaussian_project_dir="gaussian-splatting", img_path="..\\..\\images\\undistorted",
                 render_camera_path=""):

        self.experiment_path = os.path.join(log_dir, experiment_name)
        self.experiment_name = experiment_name

        self.img_path = img_path
        self.gaussian_project_dir = gaussian_project_dir

        if transform_type is not None:
            self.set_transform_type(transform_type)

        self.dir_path = pathlib.Path(__file__).parent.resolve()

    def set_transform_type(self, transform_type):

        self.transform_type = transform_type
        self.transform_path = os.path.join(self.experiment_path, "transforms", self.transform_type)

        self.setup_directory()

    def setup_directory(self):
        self.gaussian_dir = os.path.join(self.experiment_path, "gaussian_splatting")
        if not os.path.exists(self.gaussian_dir):
            os.mkdir(self.gaussian_dir)

        self.exports_dir =  os.path.join(self.experiment_path, "exports")
        if not os.path.exists(self.exports_dir):
            os.mkdir(self.exports_dir)

        self.render_dir = os.path.join(self.exports_dir, "render")
        if not os.path.exists(self.render_dir):
            os.mkdir(self.render_dir)

    def train_model(self, transforms_path, output_dir, experiment_name="gaussian"):
        return f"python {os.path.join(self.dir_path, self.gaussian_project_dir,'train.py')} -s {transforms_path} -i {self.img_path} -r 1 --eval -m {output_dir}"

    def calculate_pointcloud(self, checkpoint, output_dir):
        return ""
    
    def render_model(self, checkpoint, render_camera_path=""):
        return f"python {os.path.join(self.dir_path, self.gaussian_project_dir, 'render.py')} -m {checkpoint}"

    def evaluate_results(self, checkpoint):
        return f"python {os.path.join(self.dir_path, self.gaussian_project_dir, 'metrics.py')} -m {checkpoint}"

    def run_model(self, render=True, evaluate=True, generate_pointcloud=True):
        if not os.path.exists(os.path.join(self.gaussian_dir, "cfg_args")):

            print("Training Gausssian")
            
            err = os.system(self.train_model(self.transform_path, self.gaussian_dir))
            
            if err != 0:
                print("ERROR TRAINING GAUSSIAN")
                return False
        else:
            print("Skipping Gaussian Training")

        if not os.path.exists(os.path.join(self.gaussian_dir, "test")):

            print("Gausssian Comparison")
            
            err = os.system(self.render_model(self.gaussian_dir))
            
            if err != 0:
                print("ERROR GENERATING GAUSSIAN COMPARISON")
                return False
        else:
            print("Skipping Gaussian Image Comparisons")

        if not os.path.exists(os.path.join(self.gaussian_dir, "results.json")):

            print("Gausssian Evaluation")
            
            err = os.system(self.evaluate_results(self.gaussian_dir))
            
            if err != 0:
                print("ERROR GENERATING GAUSSIAN EVLUATION")
                return False
        else:
            print("Skipping Gaussian Evaluation")


def get_reconstruction_handler(reconstruction_type, log_dir, experiment_name, render_camera_path=""):
    if reconstruction_type.lower() == "nerfacto":
        return NeRFStudio_Handler(log_dir, experiment_name, render_camera_path=render_camera_path)
    elif reconstruction_type.lower() == "gaussian_splatting":
        return Gaussian_Handler(log_dir, experiment_name)
        
        """ ADD YOUR METHOD HERE """
        
    else:
        raise Exception(f"Unknown 3D Reconstruction handler type: {reconstruction_type}")


def config_parser():
    parser = configargparse.ArgumentParser()

    parser.add_argument("--config", is_config_file=True, help="Path to configuration file containing a list of configuration arguments")
    parser.add_argument("--log_dir", required=True, type=str)
    parser.add_argument("--experiment_name", required=True, type=str)
    parser.add_argument("--render_camera_path", type=str, default="")

    parser.add_argument("--reconstruction_models", type=str, action="append", default=[])
    parser.add_argument("--reconstruct_transforms", type=str, action="append", default=[])

    return parser.parse_args() 

if __name__ == "__main__":

    args = config_parser()

    for model in args.reconstruction_models:
        current_model = get_reconstruction_handler(model, args.log_dir, args.experiment_name, render_camera_path=args.render_camera_path)

        for transform in args.reconstruct_transforms:
            current_model.set_transform_type(transform)

            current_model.run_model() 



