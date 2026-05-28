import importlib
import sys
import subprocess


def _is_installed(package_name):
    try:
        importlib.import_module(package_name)
        return True
    except ImportError:
        return False


def _install_package(package_name, pip_name=None):
    if pip_name is None:
        pip_name = package_name
    try:
        subprocess.check_call(
            [sys.executable, "-m", "pip", "install", pip_name],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
    except subprocess.CalledProcessError:
        try:
            from modules.launch_utils import run_pip
            run_pip(f"install {pip_name}", f"AFR dependency: {pip_name}")
        except Exception:
            print(f"[AFR] Failed to install {pip_name}. Please install manually: pip install {pip_name}")


try:
    from modules import launch
    USE_LAUNCH_MODULE = True
except ImportError:
    USE_LAUNCH_MODULE = False


DEPENDENCIES = {
    "mediapipe": "mediapipe>=0.10.0,<0.11.0",
    "cv2": "opencv-python>=4.6.0",
    "numpy": "numpy>=1.21.0",
}


for import_name, pip_spec in DEPENDENCIES.items():
    if USE_LAUNCH_MODULE:
        if not launch.is_installed(import_name if import_name != "cv2" else "opencv-python"):
            print(f"[AFR] Installing {pip_spec} ...")
            launch.run_pip(f"install {pip_spec}", f"AFR dependency: {pip_spec}")
    else:
        if not _is_installed(import_name):
            print(f"[AFR] Installing {pip_spec} ...")
            _install_package(import_name, pip_spec)

print("[AFR] Advanced Face Refiner - dependencies check complete.")
