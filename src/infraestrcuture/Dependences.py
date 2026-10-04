from src.infraestrcuture.processing.yolo.Image_Processor_Impl import ImageProcessorImpl
from src.core.config import Env_Config

def image_processing_service(
    model = Env_Config.Yolo.model_yolo,
    hardware = Env_Config.Yolo.hardware_yolo
):
    return ImageProcessorImpl(
        model = model,
        hardware = hardware
    )