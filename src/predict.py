import numpy as np
from PIL import Image
import random
import yaml


config = None


def load_config(config_path="configs/default.yaml"):
    """
    Загружает конфигурацию из файла
    """
    global config
    try:
        with open(config_path, "r") as f:
            config = yaml.safe_load(f)
        print(f"Loaded config: {config_path}")
        print(f"Config content: {config}")
    except Exception as e:
        print(f"Error loading config: {e}")
        config = {"model": {"use_plug": True, "confidence_threshold": 0.5}}


def get_predictor():
    """
    Возвращает функцию предсказания в зависимости от конфига
    """
    global config

    # Если конфиг не загружен, используем значения по умолчанию
    if config is None:
        print("Config not loaded, using default plug")
        return predict_plug

    if config.get("model", {}).get("use_plug", True):
        return predict_plug
    else:
        try:
            from ultralytics import YOLO

            model_path = config.get("model", {}).get("path", "models/best.pt")
            model = YOLO(model_path)
            return lambda image, confidence: predict_image(image, confidence, model)
        except ImportError:
            print("Ultralytics not available, using plug")
            return predict_plug
        except Exception as e:
            print(f"Error loading YOLO model: {e}, using plug")
            return predict_plug


def predict_plug(image: Image.Image, confidence_threshold: float = 0.5):
    """
    Заглушка для предсказания - возвращает случайные bbox
    """
    width, height = image.size

    num_detections = random.randint(0, 5)
    detections = []

    for i in range(num_detections):
        box_size = min(width, height) // 4
        x_min = (width - box_size) // 2 + random.randint(-50, 50)
        y_min = (height - box_size) // 2 + random.randint(-50, 50)
        x_max = x_min + box_size
        y_max = y_min + box_size

        x_min = max(0, x_min)
        y_min = max(0, y_min)
        x_max = min(width, x_max)
        y_max = min(height, y_max)

        confidence = random.uniform(0.3, 0.9)

        if confidence >= confidence_threshold:
            detections.append(
                {
                    "x_min": x_min,
                    "y_min": y_min,
                    "x_max": x_max,
                    "y_max": y_max,
                    "confidence": confidence,
                }
            )

    return detections


def predict_image(image: Image.Image, confidence_threshold: float = 0.5, model=None):
    """
    Предсказание через YOLO
    """
    opencv_image = np.array(image)
    opencv_image = opencv_image[:, :, ::-1].copy()

    results = model(opencv_image, conf=confidence_threshold)

    detections = []
    for result in results:
        if result.boxes is not None:
            for box in result.boxes:
                x_min, y_min, x_max, y_max = (
                    box.xyxy[0].cpu().numpy().astype(int).tolist()
                )
                confidence = box.conf[0].cpu().numpy().item()

                detections.append(
                    {
                        "x_min": x_min,
                        "y_min": y_min,
                        "x_max": x_max,
                        "y_max": y_max,
                        "confidence": confidence,
                    }
                )

    return detections
