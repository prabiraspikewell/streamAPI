import os
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
import cv2
import numpy as np
from datetime import datetime

app = FastAPI()


app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  
    allow_credentials=True,
    allow_methods=["*"],  
    allow_headers=["*"],  
)

os.makedirs("stream-results", exist_ok=True)
# Parameters for the video
video_filename = "stream-results/output_video.mp4"
frame_width = 640
frame_height = 480
fps = 20

# Initialize video writer
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
video_writer = cv2.VideoWriter(video_filename, fourcc, fps, (frame_width, frame_height))

@app.websocket("/ws/stream")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    try:
        count = 1
        while True:
            data = await websocket.receive_bytes()
            np_img = np.frombuffer(data, np.uint8)
            img = cv2.imdecode(np_img, cv2.IMREAD_COLOR)

            if img is not None:
                file_path = os.path.join("stream-results", f"frame{count}.jpg")
                count = count + 1
                save_success = cv2.imwrite(file_path, img)
                
                img_resized = cv2.resize(img, (frame_width, frame_height))
                video_writer.write(img_resized)
                
                if save_success:
                    await websocket.send_json({"message": f"Frame saved"})
                else:
                    print("Failed to save stream")
                    await websocket.send_json({"error": "Failed to save stream"})
            else:
                print("Failed to decode stream content")
                await websocket.send_json({"error": "Failed to decode strem content"})
    except WebSocketDisconnect:
        print("WebSocket connection closed")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
                                                                 