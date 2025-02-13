import os
import sys
import subprocess
from setuptools import setup, find_packages, Command
from setuptools.command.install import install

install_requires = [
    "pycolmap @ git+https://github.com/rmbrualla/pycolmap@cc7ea4b7301720ac29287dbe450952511b32125e",
    "plyfile",
    "viser",
    "nerfview @ git+https://github.com/hangg7/nerfview.git",
    "imageio[ffmpeg]",
    "numpy<2.0.0",
    "scikit-learn",
    "tqdm",
    "torchmetrics[image]",
    "opencv-python",
    "tyro>=0.8.8",
    "Pillow",
    "tensorboard",
    "tensorly",
    "pyyaml",
    "matplotlib",
    "pandas",
    "fused-ssim @ git+https://github.com/rahul-goel/fused-ssim@1272e21a282342e89537159e4bad508b19b34157",
]

class BuildSubmodule(Command):
    """Custom command to install the submodule located in the 'submodule' folder."""
    description = "Install the submodule package from the 'submodule' folder."
    user_options = []  # No options needed for this command

    def initialize_options(self):
        pass

    def finalize_options(self):
        pass

    def run(self):
        submodule_path = os.path.join(os.path.dirname(__file__), "submodules")
        # Use pip to install the submodule; adjust the arguments as needed.
        subprocess.check_call(["pip", "install", "."], cwd=submodule_path)

class CustomInstall(install):
    """Custom install command that installs the submodule before installing the main project."""
    def run(self):
        # First, install the submodule
        self.run_command("build_submodule")
        # Then proceed with the standard installation of the main project
        install.run(self)

setup(
    name="beta_splatting",  
    version="0.1.0",      
    packages=find_packages(),
    install_requires=install_requires,
    cmdclass={
        "build_submodule": BuildSubmodule,
        "install": CustomInstall,
    },
    description="Python package for differentiable rasterization of deformable beta splatting",
    keywords="beta, splatting, cuda",
    url="https://github.com/RongLiu-Leo/beta-splatting",  
)
