from fastapi import FastAPI, UploadFile, File
from fastapi.responses import JSONResponse
import mediapipe as mp, cv2, numpy as np

app = FastAPI()
mp_hands = mp.solutions.hands

@app.post("/process")
async def process(file: UploadFile = File(...)):
    data = await file.read()
    img = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    with mp_hands.Hands(static_image_mode=True, max_num_hands=1) as hands:
        res = hands.process(img)
    if not res.multi_hand_landmarks:
        return JSONResponse({"hands": []})
    coords = []
    for lm in res.multi_hand_landmarks[0].landmark:
        coords.append({"x": lm.x, "y": lm.y, "z": lm.z})
    return JSONResponse({"hands": [coords]})
