import os
from dotenv import load_dotenv

load_dotenv()

def get_env(key: str) -> str:
    value = os.getenv(key)
    if not value or not value.strip():
        raise EnvironmentError(f"Error: Environment variable {key} is not set")
    return value

class Yolo:
    model_yolo = get_env("MODEL_YOLO")
    hardware_yolo = get_env("HARDWARE_YOLO")

class LocalCam:
    model_cam = get_env("CAMERA_INDEX")

class Config:
    yolo = Yolo()
    local_cam = LocalCam()

config_env = Config()