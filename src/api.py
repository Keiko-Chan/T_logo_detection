from fastapi import FastAPI, File, UploadFile, HTTPException, Query, Form
from fastapi.responses import JSONResponse, Response
from fastapi.staticfiles import StaticFiles
from PIL import Image, ImageDraw
import io
import logging

from src.predict import load_config, get_predictor
from src.data_models import (
    DetectionResponse,
    ErrorResponse,
    Detection,
    BoundingBox,
    ConfigResponse,
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="T-Bank Logo Detector API",
    description="API для детекции логотипа Т-Банка",
    version="1.0.0",
)

app.mount("/configs", StaticFiles(directory="configs"), name="configs")

predict_function = None
current_config = None


@app.on_event("startup")
async def startup_event():
    """Загрузка конфигурации при старте"""
    global predict_function, current_config
    load_config("configs/default.yaml")

    from src.predict import config as predict_config

    current_config = predict_config

    predict_function = get_predictor()
    logger.info("API started")
    logger.info(f"Current config: {current_config}")


@app.post("/config")
async def set_config(config_file: UploadFile = File(...)):
    """
    Установить конфигурационный файл
    """
    global predict_function, current_config
    
    contents = await config_file.read()
    temp_config_path = "configs/temp_config.yaml"
    
    with open(temp_config_path, "wb") as f:
        f.write(contents)
    
    load_config(temp_config_path)
    
    from src.predict import config as predict_config
    current_config = predict_config
    
    predict_function = get_predictor()

    if current_config and 'model' in current_config:
        model_name = "plug" if current_config['model'].get('use_plug', True) else current_config['model'].get('path', 'real_model')
        confidence = current_config['model'].get('confidence_threshold', 0.5)
        use_plug = current_config['model'].get('use_plug', True)
        
        return ConfigResponse(
            model=model_name,
            confidence=confidence,
            use_plug=use_plug,
            config_file=config_file.filename
        )
    else:
        raise HTTPException(500, "Config not loaded properly")


@app.post(
    "/detect",
    response_model=DetectionResponse,
    responses={400: {"model": ErrorResponse}},
)
async def detect_logo(
    file: UploadFile = File(...),
    return_image: bool = Query(
        False, description="Вернуть изображение с bbox вместо JSON"
    ),
    confidence: float = Query(
        None, description="Порог confidence (переопределяет конфиг)"
    ),
):
    """
    Детекция логотипа
    """
    global current_config

    if file.content_type not in ["image/jpeg", "image/png", "image/bmp", "image/webp"]:
        raise HTTPException(400, "Invalid file format. Support JPEG, PNG, BMP, WEBP.")

    try:
        if current_config is None:
            raise HTTPException(500, "Configuration not loaded")

        contents = await file.read()
        image = Image.open(io.BytesIO(contents)).convert("RGB")

        if confidence is None:
            if current_config and "model" in current_config:
                confidence = current_config["model"].get("confidence_threshold", 0.5)
            else:
                confidence = 0.5

        detections = predict_function(image, confidence)

        response_detections = []
        for det in detections:
            bbox = BoundingBox(
                x_min=det["x_min"],
                y_min=det["y_min"],
                x_max=det["x_max"],
                y_max=det["y_max"],
            )
            response_detections.append(Detection(bbox=bbox))

        if return_image:
            draw_image = image.copy()
            draw = ImageDraw.Draw(draw_image)

            for det in detections:
                bbox = [det["x_min"], det["y_min"], det["x_max"], det["y_max"]]
                conf = det["confidence"]
                draw.rectangle(bbox, outline="red", width=3)
                draw.text((bbox[0], bbox[1] - 20), f"{conf:.2f}", fill="red")

            img_byte_arr = io.BytesIO()
            draw_image.save(img_byte_arr, format="JPEG")
            return Response(content=img_byte_arr.getvalue(), media_type="image/jpeg")

        else:
            return DetectionResponse(detections=response_detections)

    except Exception as e:
        logger.error(f"Error processing image: {str(e)}")
        return JSONResponse(
            status_code=500,
            content={"error": "Internal server error", "detail": str(e)},
        )


@app.get("/health")
async def health_check():
    """Health check endpoint"""
    global current_config
    return {
        "status": "healthy",
        "config_loaded": current_config is not None,
        "predict_function_loaded": predict_function is not None,
    }


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
