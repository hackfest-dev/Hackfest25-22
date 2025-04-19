import os
from fastapi import FastAPI, WebSocket, HTTPException, File, Form, UploadFile
from fastapi.middleware.cors import CORSMiddleware
import cv2
import numpy as np
import base64
from ultralytics import YOLO
from dotenv import load_dotenv
from translation import translate_text
from pydantic import BaseModel
import asyncio
from concurrent.futures import ThreadPoolExecutor
from depth import get_depth
import mediapipe as mp
from typing import List
from orb import router as orb_router
from orb import match_descriptors
import pickle
from pathlib import Path

# Initialize MediaPipe Face Mesh
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(min_detection_confidence=0.5, min_tracking_confidence=0.5)
orb = cv2.ORB_create(nfeatures=1000)

class TranslationRequest(BaseModel):
    text: str
    target_lang: str

load_dotenv()
SERVER_IP = os.getenv("SERVER_IP")
if not SERVER_IP:
    raise ValueError("SERVER_IP not found in environment variables")

print(f"Server running on IP: {SERVER_IP}")

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(orb_router, prefix="/orb", tags=["orb"])

MODEL_PATH = os.getenv("YOLO_MODEL_PATH", "yolov8n.pt")
print(f"Loading YOLO model from {MODEL_PATH}")
model = YOLO(MODEL_PATH)
print("Model loaded successfully")

depth_executor = ThreadPoolExecutor(max_workers=2, thread_name_prefix="depth")

def load_stored_descriptors():
    """Load all stored descriptors from pkl files"""
    descriptor_dir = Path("descriptors")
    stored_descriptors = {}
    
    if descriptor_dir.exists():
        for pkl_file in descriptor_dir.glob("*.pkl"):
            try:
                with open(pkl_file, 'rb') as f:
                    data = pickle.load(f)
                    stored_descriptors[data['person_name']] = data
                print(f"✅ Loaded descriptors for {data['person_name']}")
            except Exception as e:
                print(f"❌ Error loading {pkl_file.name}: {e}")
    
    return stored_descriptors

async def process_frame_detection(frame, target_lang="en"):
    if frame is None:
        return None, "Invalid frame"
    
    try:
        results = model(frame)[0]
        detected_objects = []
        boxes_info = []
        
        # Load stored descriptors
        saved_descriptors = load_stored_descriptors()
        
        # Process faces
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        face_results = face_mesh.process(frame_rgb)
        
        # Improved box processing with error handling
        for box in results.boxes:
            try:
                class_id = int(box.cls)
                class_name = model.names[class_id]
                
                coords = [int(x) for x in box.xyxy[0].tolist()]
                h, w, _ = frame.shape
                x1, y1, x2, y2 = max(0, coords[0]), max(0, coords[1]), min(w, coords[2]), min(h, coords[3])
                
                if class_name == "person":
                    person_crop = frame[y1:y2, x1:x2]
                    if person_crop.size > 0:
                        gray = cv2.cvtColor(person_crop, cv2.COLOR_BGR2GRAY)
                        keypoints, current_descriptors = orb.detectAndCompute(gray, None)
                        
                        if current_descriptors is not None and len(current_descriptors) > 0:
                            # Ensure descriptors are uint8
                            current_descriptors = np.array(current_descriptors, dtype=np.uint8)
                            if len(current_descriptors.shape) == 1:
                                current_descriptors = current_descriptors.reshape(-1, 32)
                            
                            person_name, match_count = match_descriptors(
                                current_descriptors, 
                                saved_descriptors,
                                min_matches=10
                            )
                            
                            if person_name:
                                print(f"👤 {person_name} detected with {match_count} matches!")
                                class_name = person_name
                            else:
                                print("⚠️ No face match found")
                
                translated_name = translate_text(class_name, target_lang)
                box_info = {"label": translated_name, "box": [x1, y1, x2, y2]}
                detected_objects.append(translated_name)
                boxes_info.append(box_info)
            
            except Exception as single_box_error:
                print(f"❌ Error processing box: {single_box_error}")
                continue
            
        detection_text = ", ".join(set(detected_objects)) if detected_objects else "No objects detected"
        if not detected_objects:
            detection_text = translate_text("No objects detected", target_lang)
            
        return results, detection_text, boxes_info
        
    except Exception as e:
        error_msg = translate_text("Detection error", target_lang)
        print(f" Detection error: {str(e)}")
        return None, error_msg, []

async def process_frame_depth(frame):
    if frame is None:
        return None
    try:
        depth_result = await asyncio.get_event_loop().run_in_executor(
            depth_executor,
            get_depth,
            frame
        )
        if isinstance(depth_result, dict):
            return depth_result
        return {"depth": depth_result, "confidence": 1.0, "method": "default"}
    except Exception as e:
        print(f"❌ Depth error: {str(e)}")
        return None

def match_descriptors(frame_descriptors, saved_descriptors, min_matches=10):
    if not saved_descriptors:
        print("\n⚠️ No saved descriptors found")
        return None, 0
    
    best_match = None
    max_matches = 0
    
    # Ensure frame descriptors are uint8 and correct shape (Nx32)
    frame_descriptors = np.array(frame_descriptors, dtype=np.uint8)
    if frame_descriptors.shape[1] != 32:
        print(f"⚠️ Invalid frame descriptor shape: {frame_descriptors.shape}")
        return None, 0
    
    print(f"\n🔍 Frame descriptors: {frame_descriptors.shape}")
    
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    
    for person_name, person_data in saved_descriptors.items():
        try:
            descriptors_list = person_data.get('descriptors', [])
            
            for descriptor in descriptors_list:
                try:
                    stored_descriptors = np.array(descriptor['descriptors'], dtype=np.uint8)
                    
                    # Skip if wrong shape
                    if stored_descriptors.shape[1] != 32:
                        print(f"⚠️ Invalid stored descriptor shape: {stored_descriptors.shape}")
                        continue
                    
                    matches = bf.match(frame_descriptors, stored_descriptors)
                    good_matches = [m for m in matches if m.distance < 50]
                    num_matches = len(good_matches)
                    
                    if num_matches > max_matches and num_matches >= min_matches:
                        max_matches = num_matches
                        best_match = person_name
                        print(f"✨ Match found: {person_name} with {num_matches} matches")
                        
                except Exception as e:
                    print(f"❌ Error matching descriptors: {str(e)}")
                    continue
                    
        except Exception as e:
            print(f"❌ Error processing person: {str(e)}")
            continue
    
    return best_match, max_matches

@app.websocket("/ws/video")
async def video_stream(websocket: WebSocket):
    await websocket.accept()
    print(f" WebSocket connection established on {SERVER_IP}")
    
    try:
        await websocket.receive_text()
        lang_data = await websocket.receive_json()
        target_lang = lang_data.get("target_lang", "en")
        
        while True:
            data = await websocket.receive_text()
            frame_data = base64.b64decode(data)
            np_arr = np.frombuffer(frame_data, np.uint8)
            frame = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
            
            results, detection_text, boxes_info = await process_frame_detection(frame, target_lang)
            depth_result = await process_frame_depth(frame)
            
            await websocket.send_json({
                "translated_text": detection_text,
                "boxes": boxes_info,
                "depth": depth_result,
                "status": "success"
            })
            
    except Exception as e:
        print(f" Error: {str(e)}")
        await websocket.close()

@app.post("/translate")
async def translate(request: TranslationRequest):
    try:
        translated_text = translate_text(request.text, request.target_lang)
        return {"translated_text": translated_text, "status": "success"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/")
async def root():
    return {"status": "running", "server_ip": SERVER_IP}

@app.get("/depth")
async def get_depth_value():
    """API endpoint to return the estimated depth in cm."""
    distance = get_depth()
    if distance is None:
        return {"error": "Failed to capture depth"}
    return {"estimated_distance_cm": distance}

@app.post("/orb/process-images")
async def process_images(
    name: str = Form(...),
    images: List[UploadFile] = File(...)
):
    try:
        # Create directory for person if it doesn't exist
        person_dir = f"data/faces/{name}"
        os.makedirs(person_dir, exist_ok=True)
        
        all_descriptors = []
        processed_count = 0
        
        # Process each image
        for idx, image in enumerate(images):
            contents = await image.read()
            nparr = np.frombuffer(contents, np.uint8)
            img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            
            if img is None:
                print(f"Failed to decode image {idx}")
                continue
                
            # Extract ORB descriptors
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            keypoints, desc = orb.detectAndCompute(gray, None)
            
            if desc is not None and desc.size > 0:
                # Ensure consistent shape by padding if necessary
                if len(desc) < 500:  # Pad if less than 500 keypoints
                    padding = np.zeros((500 - len(desc), 32), dtype=np.uint8)
                    desc = np.vstack((desc, padding))
                else:
                    desc = desc[:500]  # Take only first 500 keypoints
                    
                all_descriptors.append(desc)
                processed_count += 1
                
            # Save image
            cv2.imwrite(f"{person_dir}/image_{idx}.jpg", img)
            
        if processed_count > 0:
            # Stack all descriptors into a single array
            all_descriptors = np.stack(all_descriptors)
            np.save(f"{person_dir}/descriptors.npy", all_descriptors)
            
            print(f"Saved descriptors shape: {all_descriptors.shape}")
            return {
                "status": "success", 
                "message": f"Processed {processed_count} images for {name}",
                "shape": all_descriptors.shape
            }
        else:
            raise HTTPException(
                status_code=400, 
                detail="No valid descriptors could be extracted from the images"
            )
        
    except Exception as e:
        print(f"Processing error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.on_event("shutdown")
async def shutdown_event():
    depth_executor.shutdown(wait=True)