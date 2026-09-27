from ultralytics import YOLO
from typing_extensions import List
from src.domain.entity.Deteccion import Deteccion
from src.domain.interface.I_Image_Processor import IImageProcessor

class ImageProcessorImpl(IImageProcessor):
    def __init__(self, model: str, warmup: bool, hardware: str):
        self.model = model
        self.model_instance = None
        self.warmup = warmup
        self.hardware = hardware

    def load_model(self) -> None:
        try:
            self.model_instance = YOLO(self.model)

            self.model_instance.to(self.hardware)

            if self.warmup:
                self.model_instance(self.model_instance[0])

        except Exception as e:
            raise RuntimeError(f"Error al cargar el modelo: {e}")
        
    def predict(self, frame: object, stream: bool = True) -> List[Deteccion]:
        if self.model_instance is None:
            raise RuntimeError("El modelo no ha sido cargado. Llama a load_model() primero.")

        results = self.model_instance.predict(frame, stream=stream, verbose=False)

        list_detections: List[Deteccion] = []

        for result in results:
            in_boxes = result.boxes
            
            for box in in_boxes:
                x1, y1, x2, y2 = box.xyxy[0].tolist()

                confidence = float(box.conf[0].item())
                class_id =  int(box.cls[0].item())

                class_name = self.model_instance.names[class_id]

                deteccion = Deteccion(
                    x = x1,
                    y = y1,
                    width = x2,
                    height = y2,
                    class_name = class_name,
                    confidence = confidence
                )

                list_detections.append(deteccion)

        return list_detections 