from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
import os
import cv2
import numpy as np

app = FastAPI()

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Adjust to specific origins in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Ensure the images directory exists
os.makedirs("images", exist_ok=True)

@app.post("/process-frame")
async def process_frame(frame: UploadFile = File(...)):
    try:
        # Read the image data
        img_data = await frame.read()
        
        # Convert image data to numpy array
        np_img = np.frombuffer(img_data, np.uint8)
        
        # Decode the image
        img = cv2.imdecode(np_img, cv2.IMREAD_COLOR)

        if img is None:
            return JSONResponse(content={"error": "Failed to decode image"}, status_code=400)

        # Save the image to the 'images' folder
        file_path = os.path.join("images", frame.filename)
        cv2.imwrite(file_path, img)

        return JSONResponse(content={"message": "Frame processed and saved successfully"}, status_code=200)

    except Exception as e:
        return JSONResponse(content={"error": str(e)}, status_code=500)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
