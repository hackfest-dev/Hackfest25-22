import os
import pickle
from pathlib import Path
from fastapi import APIRouter, File, UploadFile, Form
from typing import List
import cv2
import numpy as np
import mediapipe as mp

router = APIRouter()
mp_face_mesh = mp.solutions.face_mesh

# Update directory name to match your folder
DESCRIPTOR_DIR = Path("descriptors")  # Changed from "descriptor" to "descriptors"

def save_descriptors(person_name: str, descriptors_data: dict):
    """Save descriptors to a pickle file in the descriptors directory"""
    try:
        DESCRIPTOR_DIR.mkdir(exist_ok=True)
        file_path = DESCRIPTOR_DIR / f"{person_name}.pkl"
        
        # Format the data correctly
        data_to_save = {
            'person_name': person_name,
            'descriptors': [{  # Here it uses 'descriptors' not 'variations'
                'image_idx': idx,
                'descriptors': desc['descriptors']
            } for idx, desc in enumerate(descriptors_data.get('descriptors', []))]
        }
        
        with open(file_path, 'wb') as f:
            pickle.dump(data_to_save, f)
            
        print(f"✅ Saved descriptors for {person_name}")
        return True
            
    except Exception as e:
        print(f"❌ Error saving descriptors: {str(e)}")
        return False

def match_descriptors(frame_descriptors, saved_descriptors, min_matches=10):
    """Match frame descriptors with saved descriptors"""
    if not saved_descriptors:
        print("\n⚠️ No saved descriptors found.")
        return None, 0, 0.0

    best_match = None
    max_matches = 0
    confidence_score = 0.0

    # Convert frame descriptors to correct type and shape
    frame_descriptors = np.array(frame_descriptors, dtype=np.uint8)
    print(f"\n[INFO] Frame descriptors shape: {frame_descriptors.shape}, dtype: {frame_descriptors.dtype}")

    # BFMatcher with Hamming distance
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

    for person_name, person_data in saved_descriptors.items():
        try:
            descriptors_list = person_data.get('descriptors', [])
            print(f"\n[INFO] Processing {person_name} ({len(descriptors_list)} variations)")

            for i, desc_data in enumerate(descriptors_list):
                try:
                    stored_descriptors = np.array(desc_data['descriptors'], dtype=np.uint8)
                    print(f"[DEBUG] Comparing with {person_name} - Variation {i}")
                    print(f"        Shape: {stored_descriptors.shape}, dtype: {stored_descriptors.dtype}")

                    # Skip if shapes don't match
                    if frame_descriptors.shape[1] != stored_descriptors.shape[1]:
                        print(f"⚠️ Shape mismatch: {frame_descriptors.shape[1]} vs {stored_descriptors.shape[1]}")
                        continue

                    # Match descriptors
                    matches = bf.match(frame_descriptors, stored_descriptors)
                    good_matches = [m for m in matches if m.distance < 50]
                    num_matches = len(good_matches)
                    curr_confidence = num_matches / len(matches) if matches else 0.0

                    print(f"🧠 Matches: {num_matches} (Confidence: {curr_confidence:.2%})")

                    if num_matches > max_matches and num_matches >= min_matches:
                        max_matches = num_matches
                        best_match = person_name
                        confidence_score = curr_confidence
                        print(f"✨ New best match: {person_name}")

                except Exception as var_error:
                    print(f"❌ Error processing variation {i}: {str(var_error)}")
                    continue

        except Exception as person_error:
            print(f"❌ Error processing {person_name}: {str(person_error)}")
            continue

    if best_match:
        print(f"\n✅ Final match: {best_match}")
        print(f"   Confidence: {confidence_score:.2%}")
        print(f"   Matches: {max_matches}")
    else:
        print("\n❌ No match found")

    return best_match, max_matches, confidence_score

def print_stored_descriptors(descriptor_dir=DESCRIPTOR_DIR):
    """Print information about stored descriptors"""
    descriptor_path = Path(descriptor_dir)
    
    if not descriptor_path.exists():
        print("❌ Descriptor directory not found.")
        return

    pkl_files = list(descriptor_path.glob("*.pkl"))

    if not pkl_files:
        print("⚠️ No .pkl descriptor files found.")
        return

    for pkl_file in pkl_files:
        print(f"\n📂 Reading: {pkl_file.name}")
        try:
            with open(pkl_file, 'rb') as f:
                data = pickle.load(f)
                
                person_name = data.get("person_name", "Unknown")
                descriptors = data.get("descriptors", [])
                
                print(f"👤 Person Name: {person_name}")
                print(f"🧬 Descriptors Found: {len(descriptors)}")
                
                for i, desc_data in enumerate(descriptors):
                    desc_shape = np.array(desc_data['descriptors']).shape
                    print(f"  ➤ Image {desc_data['image_idx'] + 1}: Descriptors shape: {desc_shape}")
                    
        except Exception as e:
            print(f"❌ Error reading {pkl_file.name}: {e}")

@router.post("/process-images")
async def process_images(
    name: str = Form(...),
    images: List[UploadFile] = File(...)
):
    try:
        print(f"\n📸 Processing images for: {name}")
        
        descriptors_data = {
            'person_name': name,
            'descriptors': []
        }
        
        # Configure ORB with consistent settings
        orb = cv2.ORB_create(
            nfeatures=500,  # Limit features for consistency
            scaleFactor=1.2,
            nlevels=8,
            edgeThreshold=31,
            firstLevel=0,
            WTA_K=2,
            patchSize=31
        )
        
        for idx, image_file in enumerate(images):
            contents = await image_file.read()
            nparr = np.frombuffer(contents, np.uint8)
            image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            
            if image is None:
                print(f"⚠️ Failed to decode image {idx}")
                continue
            
            # Convert to grayscale for processing
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            keypoints, descriptors = orb.detectAndCompute(gray, None)
            
            if descriptors is not None:
                # Ensure descriptors are uint8 and correct shape
                descriptors = np.array(descriptors, dtype=np.uint8)
                if descriptors.shape[1] != 32:  # ORB descriptors should be 32 bytes
                    print(f"⚠️ Warning: Unexpected descriptor size: {descriptors.shape}")
                    continue
                
                descriptors_data['descriptors'].append({
                    'image_idx': idx,
                    'descriptors': descriptors.tolist(),
                    'shape': descriptors.shape
                })
                print(f"✨ Image {idx}: Found {len(keypoints)} keypoints, shape: {descriptors.shape}")
        
        if not descriptors_data['descriptors']:
            return {"status": "error", "message": "No valid descriptors extracted"}
            
        # Save descriptors to file
        if save_descriptors(name, descriptors_data):
            # Print stored descriptors info
            print("\n📊 Stored Descriptors Summary:")
            print_stored_descriptors()
            return {
                "status": "success",
                "message": f"Processed {len(images)} images for {name}",
                "descriptors_saved": len(descriptors_data['descriptors'])
            }
        else:
            return {"status": "error", "message": "Failed to save descriptors"}
            
    except Exception as e:
        print(f"❌ Processing error: {str(e)}")
        return {"status": "error", "message": str(e)}

def process_frame_detection(frame, coords, class_name, saved_descriptors, orb):
    """Process frame detection and match descriptors."""
    match_info = {"name": None, "confidence": 0.0}
    
    if class_name == "person":
        person_crop = frame[coords[1]:coords[3], coords[0]:coords[2]]
        if person_crop.size > 0:
            gray = cv2.cvtColor(person_crop, cv2.COLOR_BGR2GRAY)
            keypoints, current_descriptors = orb.detectAndCompute(gray, None)
            if current_descriptors is not None:
                person_name, match_count, confidence_score = match_descriptors(current_descriptors, saved_descriptors)
                if person_name:
                    print(f"👤 Matched: {person_name}")
                    print(f"📊 Match Stats:")
                    print(f"   - Matches: {match_count}")
                    print(f"   - Confidence: {confidence_score:.2%}")
                    if confidence_score > 0.6:
                        class_name = person_name
                        print(f"✅ High confidence match")
                    else:
                        print(f"⚠️ Low confidence match, keeping as generic person")
    
    return class_name, match