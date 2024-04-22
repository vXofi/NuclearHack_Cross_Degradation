from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
import numpy as np
import cv2
from ultralytics import YOLO
import uuid
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"]
)

model = YOLO("C:/Users/Alexander/Desktop/200/yolo8n-sanya.pt")

@app.post("/detect/")
async def detect(file: UploadFile = File(...)):
    contents = await file.read()
    nparr = np.frombuffer(contents, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    # Применение модели для детекции
    results = model(img)

    threshold = 0.25
    cigarette_class = 0
    person_class = 1
    max_conf = 0.0
    smoker_detected = False
    crop = False

    if (cigarette_class in results[0].boxes.cls.cpu().numpy()) and (person_class in results[0].boxes.cls.cpu().numpy()):
        conf = results[0].boxes.conf.cpu().numpy()[cigarette_class == results[0].boxes.cls.cpu().numpy()]
        conf = max(conf)
        if conf >= threshold:
            smoker_detected = True
            max_conf = conf

    new_img = results
    if not smoker_detected:
        crop = True
        person_indices = np.where(results[0].boxes.cls.cpu().numpy() == person_class)[0]
        for index in person_indices:
            box = results[0].boxes[index]
            x1, y1, x2, y2 = [int(coord) for coord in box.xyxy.cpu().numpy().flatten()]
            roi = img[y1:y2, x1:x2]
            res_roi = model(roi)
            if cigarette_class in res_roi[0].boxes.cls.cpu().numpy():
                confs = res_roi[0].boxes.conf.cpu().numpy()[res_roi[0].boxes.cls.cpu().numpy() == cigarette_class]
                if confs.size > 0:
                    conf = max(confs)
                    if conf > max_conf:
                        new_img = res_roi
                        max_conf = conf
                        smoker_detected = True if conf >= threshold else False

    # Сохранение входного и обработанного изображения
    input_image_path = f"temp/{uuid.uuid4()}_input.jpg"
    output_image_path = f"temp/{uuid.uuid4()}_output.jpg"
    cv2.imwrite(input_image_path, img)
    cv2.imwrite(output_image_path, new_img[0].plot())  # результат с рендерингом

    # Преобразование numpy float в стандартный float
    max_conf = float(max_conf * 100)  # Преобразование для JSON сериализации

    return JSONResponse(content={
        "input_image": input_image_path,
        "output_image": output_image_path,
        "smoker_detected": smoker_detected,
        "confidence": max_conf,  # Теперь это обычный float
        "crop": crop
    })

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
