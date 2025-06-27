from fastapi import FastAPI, UploadFile, File
import mediapipe as mp
import cv2
import numpy as np
import uvicorn

app = FastAPI()

mp_pose = mp.solutions.pose
pose = mp_pose.Pose()

@app.post("/detect_pose/")
async def detect_pose(file: UploadFile = File(...)):
    # Читаем изображение из файла
    contents = await file.read()
    nparr = np.frombuffer(contents, np.uint8)
    image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    # Распознавание позы
    results = pose.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    
    if results.pose_landmarks:
        landmarks = []
        for lm in results.pose_landmarks.landmark:
            landmarks.append({
                "x": lm.x,
                "y": lm.y,
                "z": lm.z,
                "visibility": lm.visibility
            })
        return {"landmarks": landmarks}
    else:
        return {"landmarks": []}

if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000)