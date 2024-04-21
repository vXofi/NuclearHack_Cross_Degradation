from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
import numpy as np
import cv2
from ultralytics import YOLO

app = FastAPI()
model = YOLO("/Users/veceloe/Downloads/NuclearHack/augmented.pt")

@app.post("/detect/")
async def detect(file: UploadFile = File(...)):
    contents = await file.read()
    nparr = np.frombuffer(contents, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    threshold = 0.4
    crop_threshold = 0.3  # более низкий порог для кропнутых картинок
    cigarette_class = 0
    person_class = 1
    max_conf = 0.0
    smoker_detected = False
    crop = False
    res = model(img)
    if (1 in res[0].boxes.cls.cpu().numpy()):
        conf = res[0].boxes.conf.cpu().numpy()[1 == res[0].boxes.cls.cpu().numpy()]
        conf = max(conf)
        if conf >= threshold:
            smoker_detected = True
            max_conf = conf
    # if (cigarette_class in res[0].boxes.cls.cpu().numpy()) and (person_class in res[0].boxes.cls.cpu().numpy()):
    #     conf = res[0].boxes.conf.cpu().numpy()[cigarette_class == res[0].boxes.cls.cpu().numpy()]
    #     conf = max(conf)
    #     if conf >= threshold:
    #         smoker_detected = True
    #         max_conf = conf
    # if not smoker_detected:  # проверяем кропнутые картинки только если не нашли сигарету на полной
    #     crop = True
    #     person_indices = np.where(res[0].boxes.cls.cpu().numpy() == person_class)[0]
    #     for index in person_indices:
    #         box = res[0].boxes[index]
    #         x1, y1, x2, y2 = [int(coord) for coord in box.xyxy.cpu().numpy().flatten()]
    #         roi = img[y1:y2, x1:x2]
    #         res_roi = model(roi)
    #         if cigarette_class in res_roi[0].boxes.cls.cpu().numpy():
    #             confs = res_roi[0].boxes.conf.cpu().numpy()[res_roi[0].boxes.cls.cpu().numpy() == cigarette_class]
    #             if confs.size > 0:
    #                 conf = max(confs)
    #                 if conf > max_conf:
    #                     max_conf = conf
    #                     smoker_detected = (True if conf >= crop_threshold else False)
    return JSONResponse(content={"smoker_detected": smoker_detected, "confidence": float(max_conf), "crop": crop})

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
