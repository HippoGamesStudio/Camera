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
    contents = await file.read()
    nparr = np.frombuffer(contents, np.uint8)
    image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    results = pose.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    
    landmarks_response = []
    if results.pose_landmarks:
        for idx, lm in enumerate(results.pose_landmarks.landmark):
            landmarks_response.append({
                "index": idx,
                "x": lm.x,
                "y": lm.y,
                "z": lm.z,
                "visibility": lm.visibility
            })
    return {"landmarks": landmarks_response}

if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000)
