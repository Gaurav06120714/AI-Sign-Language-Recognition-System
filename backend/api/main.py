import os
import sys
import json

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from dotenv import load_dotenv

load_dotenv(dotenv_path=os.path.join(os.path.dirname(__file__), "../.env"))

sys.path.append(os.path.join(os.path.dirname(__file__), "../gesture"))
from feature_extractor import extract_features_from_list
from model_predictor import predict_gesture, get_supported_gestures

app = FastAPI(title="Sign Language Recognition API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://127.0.0.1:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health")
def health():
    gestures = get_supported_gestures()
    return {
        "status": "ok",
        "model_loaded": len(gestures) > 0,
        "supported_gestures": len(gestures),
    }


@app.get("/gestures")
def gestures():
    return {"gestures": get_supported_gestures()}


@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    try:
        while True:
            data = await websocket.receive_text()
            payload = json.loads(data)
            landmarks = payload.get("landmarks")
            features = extract_features_from_list(landmarks)
            result = predict_gesture(features)
            await websocket.send_text(json.dumps(result))
    except WebSocketDisconnect:
        pass
    except Exception as e:
        await websocket.send_text(json.dumps({"error": str(e)}))


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host=os.getenv("HOST", "0.0.0.0"),
                port=int(os.getenv("PORT", "8000")), reload=True)
