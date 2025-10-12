from fastapi import FastAPI, Request, WebSocket, WebSocketDisconnect, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from fastapi.staticfiles import StaticFiles
import cv2
import numpy as np
import json
import asyncio
import mediapipe as mp
from typing import List, Dict
import math
from collections import deque
import os
import pickle
import base64
import time

app = FastAPI()

# Serve static files (HTML, CSS, JS)
app.mount("/static", StaticFiles(directory="."), name="static")

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load MediaPipe models
mp_face_detection = mp.solutions.face_detection
mp_face_mesh = mp.solutions.face_mesh
mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils

# Initialize models
face_detection = mp_face_detection.FaceDetection(model_selection=0, min_detection_confidence=0.5)
face_mesh = mp_face_mesh.FaceMesh(static_image_mode=False, max_num_faces=2, refine_landmarks=True, min_detection_confidence=0.5, min_tracking_confidence=0.5)
pose = mp_pose.Pose(static_image_mode=False, min_detection_confidence=0.5, min_tracking_confidence=0.5)

# Face database for authentication
FACE_DB_PATH = "face_database.pkl"
face_database = {}

def load_face_database():
    """Load face database from file"""
    global face_database
    if os.path.exists(FACE_DB_PATH):
        try:
            with open(FACE_DB_PATH, 'rb') as f:
                face_database = pickle.load(f)
            print(f"Loaded {len(face_database)} faces from database")
        except Exception as e:
            print(f"Error loading face database: {e}")
            face_database = {}

def save_face_database():
    """Save face database to file"""
    try:
        with open(FACE_DB_PATH, 'wb') as f:
            pickle.dump(face_database, f)
        print(f"Saved {len(face_database)} faces to database")
    except Exception as e:
        print(f"Error saving face database: {e}")

def extract_face_embedding(face_landmarks):
    """Extract face embedding from MediaPipe landmarks"""
    if not face_landmarks:
        return None
    
    # Convert landmarks to normalized coordinates
    embedding = []
    for landmark in face_landmarks.landmark:
        embedding.extend([landmark.x, landmark.y, landmark.z])
    
    # Ensure consistent embedding size (468 landmarks * 3 coordinates = 1404)
    embedding_array = np.array(embedding)
    if len(embedding_array) != 1404:
        print(f"Debug: Embedding size mismatch: {len(embedding_array)} != 1404, truncating/padding")
        if len(embedding_array) > 1404:
            embedding_array = embedding_array[:1404]  # Truncate if too long
        else:
            # Pad with zeros if too short
            padding = np.zeros(1404 - len(embedding_array))
            embedding_array = np.concatenate([embedding_array, padding])
    
    return embedding_array

def calculate_face_similarity(embedding1, embedding2):
    """Calculate similarity between two face embeddings"""
    if embedding1 is None or embedding2 is None:
        return 0.0
    
    try:
        # Ensure both embeddings have the same shape
        if len(embedding1) != len(embedding2):
            print(f"Debug: Embedding shape mismatch: {len(embedding1)} vs {len(embedding2)}")
            min_len = min(len(embedding1), len(embedding2))
            embedding1 = embedding1[:min_len]
            embedding2 = embedding2[:min_len]
        
        # Normalize embeddings
        norm1 = np.linalg.norm(embedding1)
        norm2 = np.linalg.norm(embedding2)
        
        if norm1 == 0 or norm2 == 0:
            return 0.0
        
        # Calculate cosine similarity
        similarity = np.dot(embedding1, embedding2) / (norm1 * norm2)
        return float(similarity)
    except Exception as e:
        print(f"Debug: Error calculating face similarity: {e}")
        return 0.0

# Load face database on startup
load_face_database()

# Liveness verification state
liveness_state = {
    "ear_history": deque(maxlen=30),  # Eye Aspect Ratio history
    "mouth_history": deque(maxlen=30),  # Mouth movement history
    "head_pose_history": deque(maxlen=30),  # Head pose history
    "blink_count": 0,
    "last_blink_frame": 0,
    "challenge_active": False,
    "challenge_type": None,
    "challenge_start_time": 0,
    "verification_passed": False
}

def calculate_ear(eye_landmarks):
    """Calculate Eye Aspect Ratio (EAR) for blink detection"""
    # MediaPipe eye landmarks (6 points per eye)
    # Points: outer corner, upper eyelid, inner corner, lower eyelid
    if len(eye_landmarks) < 6:
        return 0
    
    # Vertical distances
    vertical_1 = np.linalg.norm(eye_landmarks[1] - eye_landmarks[5])
    vertical_2 = np.linalg.norm(eye_landmarks[2] - eye_landmarks[4])
    
    # Horizontal distance
    horizontal = np.linalg.norm(eye_landmarks[0] - eye_landmarks[3])
    
    # EAR formula
    ear = (vertical_1 + vertical_2) / (2.0 * horizontal)
    return ear

def calculate_mouth_aspect_ratio(mouth_landmarks):
    """Calculate Mouth Aspect Ratio for mouth movement detection"""
    if len(mouth_landmarks) < 20:
        return 0
    
    # Key mouth points for MAR calculation
    # Vertical distances (upper and lower lip)
    vertical_1 = np.linalg.norm(mouth_landmarks[3] - mouth_landmarks[9])
    vertical_2 = np.linalg.norm(mouth_landmarks[2] - mouth_landmarks[10])
    vertical_3 = np.linalg.norm(mouth_landmarks[4] - mouth_landmarks[8])
    
    # Horizontal distance (mouth width)
    horizontal = np.linalg.norm(mouth_landmarks[0] - mouth_landmarks[6])
    
    # MAR formula
    mar = (vertical_1 + vertical_2 + vertical_3) / (3.0 * horizontal)
    return mar

def calculate_head_pose(face_landmarks):
    """Calculate head pose angles (yaw, pitch, roll)"""
    if face_landmarks is None or len(face_landmarks) < 68:
        return {"yaw": 0, "pitch": 0, "roll": 0}
    
    # Key facial points for pose estimation
    nose_tip = face_landmarks[30]
    chin = face_landmarks[8]
    left_eye = face_landmarks[36]
    right_eye = face_landmarks[45]
    left_mouth = face_landmarks[48]
    right_mouth = face_landmarks[54]
    
    # Calculate angles
    # Yaw (left/right rotation)
    eye_center = (left_eye + right_eye) / 2
    yaw = math.atan2(nose_tip[0] - eye_center[0], nose_tip[1] - eye_center[1])
    
    # Pitch (up/down rotation)
    mouth_center = (left_mouth + right_mouth) / 2
    pitch = math.atan2(nose_tip[1] - mouth_center[1], nose_tip[0] - mouth_center[0])
    
    # Roll (tilt rotation)
    roll = math.atan2(right_eye[1] - left_eye[1], right_eye[0] - left_eye[0])
    
    return {
        "yaw": math.degrees(yaw),
        "pitch": math.degrees(pitch),
        "roll": math.degrees(roll)
    }

def detect_blink(ear, frame_count):
    """Detect blink based on EAR threshold"""
    global liveness_state
    
    liveness_state["ear_history"].append(ear)
    
    if len(liveness_state["ear_history"]) < 3:
        return False
    
    # Blink detection: EAR drops below threshold
    ear_threshold = 0.25
    consecutive_low = 0
    
    for recent_ear in list(liveness_state["ear_history"])[-3:]:
        if recent_ear < ear_threshold:
            consecutive_low += 1
    
    if consecutive_low >= 2 and frame_count - liveness_state["last_blink_frame"] > 10:
        liveness_state["blink_count"] += 1
        liveness_state["last_blink_frame"] = frame_count
        return True
    
    return False

def check_liveness(frame_count):
    """Comprehensive liveness verification"""
    global liveness_state
    
    liveness_checks = {
        "blink_detected": False,
        "mouth_movement": False,
        "head_movement": False,
        "micro_movements": False
    }
    
    # Check for blinks
    if len(liveness_state["ear_history"]) > 0:
        current_ear = liveness_state["ear_history"][-1]
        liveness_checks["blink_detected"] = detect_blink(current_ear, frame_count)
    
    # Check mouth movement
    if len(liveness_state["mouth_history"]) > 5:
        recent_mars = list(liveness_state["mouth_history"])[-5:]
        mar_variance = np.var(recent_mars)
        liveness_checks["mouth_movement"] = mar_variance > 0.001
    
    # Check head movement
    if len(liveness_state["head_pose_history"]) > 5:
        recent_poses = list(liveness_state["head_pose_history"])[-5:]
        yaw_variance = np.var([pose["yaw"] for pose in recent_poses])
        pitch_variance = np.var([pose["pitch"] for pose in recent_poses])
        liveness_checks["head_movement"] = (yaw_variance > 1.0) or (pitch_variance > 1.0)
    
    # Check for micro-movements (consistency check)
    if len(liveness_state["ear_history"]) > 10:
        ear_variance = np.var(list(liveness_state["ear_history"])[-10:])
        liveness_checks["micro_movements"] = ear_variance > 0.0001
    
    return liveness_checks

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    frame_count = 0
    try:
        while True:
            # Receive frame data from frontend
            data = await websocket.receive_bytes()
            frame_count += 1
            
            # Convert bytes to numpy array
            nparr = np.frombuffer(data, np.uint8)
            frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            
            # Convert BGR to RGB for MediaPipe
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            # Detect faces, landmarks, and pose using MediaPipe
            face_detection_results = face_detection.process(rgb_frame)
            face_mesh_results = face_mesh.process(rgb_frame)
            pose_results = pose.process(rgb_frame)
            
            face_data = []
            print(f"Debug: Face detections: {len(face_detection_results.detections) if face_detection_results.detections else 0}")
            print(f"Debug: Face landmarks: {len(face_mesh_results.multi_face_landmarks) if face_mesh_results.multi_face_landmarks else 0}")
            print(f"Debug: Pose landmarks: {len(pose_results.pose_landmarks.landmark) if pose_results.pose_landmarks else 0}")
            
            if face_detection_results.detections and face_mesh_results.multi_face_landmarks:
                for i, detection in enumerate(face_detection_results.detections):
                    # Get face bounding box
                    bbox = detection.location_data.relative_bounding_box
                    h, w, _ = frame.shape
                    x = int(bbox.xmin * w)
                    y = int(bbox.ymin * h)
                    width = int(bbox.width * w)
                    height = int(bbox.height * h)
                    
                    # Get facial landmarks for this face
                    if i < len(face_mesh_results.multi_face_landmarks):
                        landmarks = face_mesh_results.multi_face_landmarks[i]
                        
                        # Extract specific facial features (MediaPipe 468 landmarks)
                        left_eye_points = np.array([
                            [landmarks.landmark[33].x * w, landmarks.landmark[33].y * h],
                            [landmarks.landmark[7].x * w, landmarks.landmark[7].y * h],
                            [landmarks.landmark[163].x * w, landmarks.landmark[163].y * h],
                            [landmarks.landmark[144].x * w, landmarks.landmark[144].y * h],
                            [landmarks.landmark[145].x * w, landmarks.landmark[145].y * h],
                            [landmarks.landmark[153].x * w, landmarks.landmark[153].y * h]
                        ])
                        
                        right_eye_points = np.array([
                            [landmarks.landmark[362].x * w, landmarks.landmark[362].y * h],
                            [landmarks.landmark[382].x * w, landmarks.landmark[382].y * h],
                            [landmarks.landmark[381].x * w, landmarks.landmark[381].y * h],
                            [landmarks.landmark[380].x * w, landmarks.landmark[380].y * h],
                            [landmarks.landmark[374].x * w, landmarks.landmark[374].y * h],
                            [landmarks.landmark[373].x * w, landmarks.landmark[373].y * h]
                        ])
                        
                        mouth_points = np.array([
                            [landmarks.landmark[61].x * w, landmarks.landmark[61].y * h],
                            [landmarks.landmark[84].x * w, landmarks.landmark[84].y * h],
                            [landmarks.landmark[17].x * w, landmarks.landmark[17].y * h],
                            [landmarks.landmark[314].x * w, landmarks.landmark[314].y * h],
                            [landmarks.landmark[405].x * w, landmarks.landmark[405].y * h],
                            [landmarks.landmark[320].x * w, landmarks.landmark[320].y * h],
                            [landmarks.landmark[307].x * w, landmarks.landmark[307].y * h],
                            [landmarks.landmark[375].x * w, landmarks.landmark[375].y * h],
                            [landmarks.landmark[321].x * w, landmarks.landmark[321].y * h],
                            [landmarks.landmark[308].x * w, landmarks.landmark[308].y * h],
                            [landmarks.landmark[324].x * w, landmarks.landmark[324].y * h],
                            [landmarks.landmark[318].x * w, landmarks.landmark[318].y * h],
                            [landmarks.landmark[13].x * w, landmarks.landmark[13].y * h],
                            [landmarks.landmark[82].x * w, landmarks.landmark[82].y * h],
                            [landmarks.landmark[81].x * w, landmarks.landmark[81].y * h],
                            [landmarks.landmark[80].x * w, landmarks.landmark[80].y * h],
                            [landmarks.landmark[78].x * w, landmarks.landmark[78].y * h],
                            [landmarks.landmark[95].x * w, landmarks.landmark[95].y * h],
                            [landmarks.landmark[88].x * w, landmarks.landmark[88].y * h],
                            [landmarks.landmark[178].x * w, landmarks.landmark[178].y * h]
                        ])
                        
                        # Calculate liveness metrics
                        left_ear = calculate_ear(left_eye_points)
                        right_ear = calculate_ear(right_eye_points)
                        avg_ear = (left_ear + right_ear) / 2
                        mar = calculate_mouth_aspect_ratio(mouth_points)
                        
                        # Store in liveness state
                        liveness_state["ear_history"].append(avg_ear)
                        liveness_state["mouth_history"].append(mar)
                        
                        # Calculate head pose (simplified for MediaPipe 468 landmarks)
                        head_pose = {
                            "yaw": 0,  # Simplified for now
                            "pitch": 0,
                            "roll": 0
                        }
                        liveness_state["head_pose_history"].append(head_pose)
                        
                        # Perform liveness verification
                        liveness_checks = check_liveness(frame_count)
                        
                        # Extract face embedding for authentication
                        face_embedding = extract_face_embedding(landmarks)
                        print(f"Debug: Face embedding extracted: {face_embedding is not None}")
                        if face_embedding is not None:
                            print(f"Debug: Embedding shape: {face_embedding.shape}")
                        else:
                            print("Debug: No face embedding generated")
                        
                        landmarks_data = {
                            "left_eye": [
                                {"x": int(point[0]), "y": int(point[1])} for point in left_eye_points
                            ],
                            "right_eye": [
                                {"x": int(landmarks.landmark[362].x * w), "y": int(landmarks.landmark[362].y * h)},
                                {"x": int(landmarks.landmark[382].x * w), "y": int(landmarks.landmark[382].y * h)},
                                {"x": int(landmarks.landmark[381].x * w), "y": int(landmarks.landmark[381].y * h)},
                                {"x": int(landmarks.landmark[380].x * w), "y": int(landmarks.landmark[380].y * h)},
                                {"x": int(landmarks.landmark[374].x * w), "y": int(landmarks.landmark[374].y * h)},
                                {"x": int(landmarks.landmark[373].x * w), "y": int(landmarks.landmark[373].y * h)}
                            ],
                            "nose": [
                                {"x": int(landmarks.landmark[1].x * w), "y": int(landmarks.landmark[1].y * h)},
                                {"x": int(landmarks.landmark[2].x * w), "y": int(landmarks.landmark[2].y * h)},
                                {"x": int(landmarks.landmark[5].x * w), "y": int(landmarks.landmark[5].y * h)},
                                {"x": int(landmarks.landmark[4].x * w), "y": int(landmarks.landmark[4].y * h)},
                                {"x": int(landmarks.landmark[6].x * w), "y": int(landmarks.landmark[6].y * h)},
                                {"x": int(landmarks.landmark[19].x * w), "y": int(landmarks.landmark[19].y * h)},
                                {"x": int(landmarks.landmark[20].x * w), "y": int(landmarks.landmark[20].y * h)},
                                {"x": int(landmarks.landmark[94].x * w), "y": int(landmarks.landmark[94].y * h)},
                                {"x": int(landmarks.landmark[125].x * w), "y": int(landmarks.landmark[125].y * h)}
                            ],
                            "mouth": [
                                {"x": int(landmarks.landmark[61].x * w), "y": int(landmarks.landmark[61].y * h)},
                                {"x": int(landmarks.landmark[84].x * w), "y": int(landmarks.landmark[84].y * h)},
                                {"x": int(landmarks.landmark[17].x * w), "y": int(landmarks.landmark[17].y * h)},
                                {"x": int(landmarks.landmark[314].x * w), "y": int(landmarks.landmark[314].y * h)},
                                {"x": int(landmarks.landmark[405].x * w), "y": int(landmarks.landmark[405].y * h)},
                                {"x": int(landmarks.landmark[320].x * w), "y": int(landmarks.landmark[320].y * h)},
                                {"x": int(landmarks.landmark[307].x * w), "y": int(landmarks.landmark[307].y * h)},
                                {"x": int(landmarks.landmark[375].x * w), "y": int(landmarks.landmark[375].y * h)},
                                {"x": int(landmarks.landmark[321].x * w), "y": int(landmarks.landmark[321].y * h)},
                                {"x": int(landmarks.landmark[308].x * w), "y": int(landmarks.landmark[308].y * h)},
                                {"x": int(landmarks.landmark[324].x * w), "y": int(landmarks.landmark[324].y * h)},
                                {"x": int(landmarks.landmark[318].x * w), "y": int(landmarks.landmark[318].y * h)},
                                {"x": int(landmarks.landmark[13].x * w), "y": int(landmarks.landmark[13].y * h)},
                                {"x": int(landmarks.landmark[82].x * w), "y": int(landmarks.landmark[82].y * h)},
                                {"x": int(landmarks.landmark[81].x * w), "y": int(landmarks.landmark[81].y * h)},
                                {"x": int(landmarks.landmark[80].x * w), "y": int(landmarks.landmark[80].y * h)},
                                {"x": int(landmarks.landmark[78].x * w), "y": int(landmarks.landmark[78].y * h)},
                                {"x": int(landmarks.landmark[95].x * w), "y": int(landmarks.landmark[95].y * h)},
                                {"x": int(landmarks.landmark[88].x * w), "y": int(landmarks.landmark[88].y * h)},
                                {"x": int(landmarks.landmark[178].x * w), "y": int(landmarks.landmark[178].y * h)}
                            ]
                        }
                        
                        # Face verification
                        face_verified = False
                        face_similarity = 0.0
                        if face_embedding is not None and face_database:
                            best_similarity = 0.0
                            for name, stored_face_data in face_database.items():
                                similarity = calculate_face_similarity(face_embedding, stored_face_data["embedding"])
                                if similarity > best_similarity:
                                    best_similarity = similarity
                            face_verified = best_similarity > 0.85
                            face_similarity = best_similarity
                        
                        face_data.append({
                            "x": x,
                            "y": y,
                            "width": width,
                            "height": height,
                            "landmarks": landmarks_data,
                            "embedding": face_embedding.tolist() if face_embedding is not None else None,
                            "verified": face_verified,
                            "similarity": face_similarity
                        })
            
            # If no faces with landmarks, try just face detection
            if not face_data and face_detection_results.detections:
                print("Debug: No landmarks, trying face detection only")
                for detection in face_detection_results.detections:
                    bbox = detection.location_data.relative_bounding_box
                    h, w, _ = frame.shape
                    x = int(bbox.xmin * w)
                    y = int(bbox.ymin * h)
                    width = int(bbox.width * w)
                    height = int(bbox.height * h)
                    
                    # Try to get face mesh for this detection
                    face_embedding = None
                    face_verified = False
                    face_similarity = 0.0
                    
                    if face_mesh_results.multi_face_landmarks:
                        # Try to find the best matching face mesh
                        best_landmarks = None
                        best_distance = float('inf')
                        
                        for landmarks in face_mesh_results.multi_face_landmarks:
                            # Calculate distance between detection center and mesh center
                            mesh_center_x = sum([lm.x for lm in landmarks.landmark]) / len(landmarks.landmark)
                            mesh_center_y = sum([lm.y for lm in landmarks.landmark]) / len(landmarks.landmark)
                            
                            detection_center_x = (bbox.xmin + bbox.width/2)
                            detection_center_y = (bbox.ymin + bbox.height/2)
                            
                            distance = ((mesh_center_x - detection_center_x)**2 + (mesh_center_y - detection_center_y)**2)**0.5
                            
                            if distance < best_distance:
                                best_distance = distance
                                best_landmarks = landmarks
                        
                        if best_landmarks:
                            face_embedding = extract_face_embedding(best_landmarks)
                            print(f"Debug: Extracted embedding from best match: {face_embedding is not None}")
                            
                            if face_embedding is not None and face_database:
                                best_similarity = 0.0
                                for name, stored_face_data in face_database.items():
                                    similarity = calculate_face_similarity(face_embedding, stored_face_data["embedding"])
                                    if similarity > best_similarity:
                                        best_similarity = similarity
                                face_verified = best_similarity > 0.85
                                face_similarity = best_similarity
                    
                    face_data.append({
                        "x": x,
                        "y": y,
                        "width": width,
                        "height": height,
                        "landmarks": None,
                        "embedding": face_embedding.tolist() if face_embedding is not None else None,
                        "verified": face_verified,
                        "similarity": face_similarity
                    })
            
            # Extract pose landmarks
            pose_data = []
            if pose_results.pose_landmarks:
                h, w, _ = frame.shape
                landmarks = pose_results.pose_landmarks.landmark
                
                # Key body parts we want to track
                body_parts = {
                    "left_shoulder": landmarks[11],
                    "right_shoulder": landmarks[12],
                    "left_elbow": landmarks[13],
                    "right_elbow": landmarks[14],
                    "left_wrist": landmarks[15],
                    "right_wrist": landmarks[16],
                    "left_hip": landmarks[23],
                    "right_hip": landmarks[24],
                    "left_knee": landmarks[25],
                    "right_knee": landmarks[26],
                    "left_ankle": landmarks[27],
                    "right_ankle": landmarks[28],
                    "nose": landmarks[0],
                    "left_eye": landmarks[2],
                    "right_eye": landmarks[5],
                    "left_ear": landmarks[7],
                    "right_ear": landmarks[8]
                }
                
                for part_name, landmark in body_parts.items():
                    if landmark.visibility > 0.5:  # Only include visible landmarks
                        pose_data.append({
                            "name": part_name,
                            "x": int(landmark.x * w),
                            "y": int(landmark.y * h),
                            "visibility": landmark.visibility
                        })
            
            # Convert numpy types to Python types for JSON serialization
            liveness_checks = check_liveness(frame_count)
            checks_serializable = {
                "blink_detected": bool(liveness_checks["blink_detected"]),
                "mouth_movement": bool(liveness_checks["mouth_movement"]),
                "head_movement": bool(liveness_checks["head_movement"]),
                "micro_movements": bool(liveness_checks["micro_movements"])
            }
            
            # Send face coordinates, pose landmarks, and liveness data back to frontend
            response_data = {
                "faces": face_data,
                "pose": pose_data,
                "face_count": len(face_data),
                "pose_count": len(pose_data),
                "liveness": {
                    "blink_count": int(liveness_state["blink_count"]),
                    "ear_current": float(liveness_state["ear_history"][-1]) if liveness_state["ear_history"] else 0.0,
                    "mar_current": float(liveness_state["mouth_history"][-1]) if liveness_state["mouth_history"] else 0.0,
                    "head_pose": liveness_state["head_pose_history"][-1] if liveness_state["head_pose_history"] else {"yaw": 0, "pitch": 0, "roll": 0},
                    "checks": checks_serializable,
                    "verification_passed": bool(liveness_state["verification_passed"])
                }
            }
            print(f"Debug: Sending {len(face_data)} faces, {len(pose_data)} pose points, blinks: {liveness_state['blink_count']}")
            if face_data:
                print(f"Debug: First face has embedding: {face_data[0].get('embedding') is not None}")
                if face_data[0].get('embedding') is not None:
                    print(f"Debug: Embedding length: {len(face_data[0]['embedding'])}")
            await websocket.send_text(json.dumps(response_data))
            
    except WebSocketDisconnect:
        print("Client disconnected")

@app.post("/detect-faces")
async def detect_faces_static(file: UploadFile = File(...)):
    # For static image uploads (backup method)
    contents = await file.read()
    nparr = np.frombuffer(contents, np.uint8)
    image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    
    # Convert BGR to RGB for MediaPipe
    rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    # Detect faces using MediaPipe
    face_detection_results = face_detection.process(rgb_image)
    
    face_data = []
    if face_detection_results.detections:
        for detection in face_detection_results.detections:
            bbox = detection.location_data.relative_bounding_box
            h, w, _ = image.shape
            x = int(bbox.xmin * w)
            y = int(bbox.ymin * h)
            width = int(bbox.width * w)
            height = int(bbox.height * h)
            
            face_data.append({
                "x": x,
                "y": y,
                "width": width,
                "height": height
            })
    
    return {"faces": face_data, "count": len(face_data)}

@app.get("/")
async def root():
    from fastapi.responses import FileResponse
    return FileResponse("index.html")

@app.get("/api")
async def api_info():
    return {"message": "Face Detection API"}

@app.post("/register-face")
async def register_face():
    """Register a new face for authentication"""
    try:
        # This will be called from the frontend with face data
        return {"message": "Face registration endpoint ready", "status": "success"}
    except Exception as e:
        return {"message": f"Error: {str(e)}", "status": "error"}

@app.post("/save-face")
async def save_face(request: Request):
    """Save face embedding to database"""
    try:
        data = await request.json()
        print(f"Debug: Received face registration data: {data}")
        face_name = data.get("name", "user")
        face_embedding = data.get("embedding")
        
        print(f"Debug: Face name: {face_name}")
        print(f"Debug: Face embedding type: {type(face_embedding)}")
        print(f"Debug: Face embedding length: {len(face_embedding) if face_embedding else 'None'}")
        
        if face_embedding:
            face_database[face_name] = {
                "embedding": np.array(face_embedding),
                "timestamp": time.time()
            }
            save_face_database()
            print(f"Debug: Successfully saved face '{face_name}'")
            return {"message": f"Face '{face_name}' registered successfully", "status": "success"}
        else:
            print("Debug: No face embedding provided")
            return {"message": "No face embedding provided", "status": "error"}
    except Exception as e:
        print(f"Debug: Error in save_face: {str(e)}")
        return {"message": f"Error saving face: {str(e)}", "status": "error"}

@app.get("/face-database")
async def get_face_database():
    """Get current face database status"""
    return {
        "faces": list(face_database.keys()),
        "count": len(face_database),
        "status": "success"
    }

@app.post("/verify-face")
async def verify_face(request: Request):
    """Verify if detected face matches registered faces"""
    try:
        data = await request.json()
        current_embedding = data.get("embedding")
        
        if not current_embedding or not face_database:
            return {"verified": False, "similarity": 0.0, "message": "No registered faces or invalid embedding"}
        
        current_embedding = np.array(current_embedding)
        best_similarity = 0.0
        best_match = None
        
        for name, face_data in face_database.items():
            similarity = calculate_face_similarity(current_embedding, face_data["embedding"])
            if similarity > best_similarity:
                best_similarity = similarity
                best_match = name
        
        # Threshold for face recognition (adjust as needed)
        threshold = 0.85
        verified = best_similarity > threshold
        
        return {
            "verified": verified,
            "similarity": best_similarity,
            "best_match": best_match,
            "threshold": threshold,
            "message": f"Face {'verified' if verified else 'not recognized'}"
        }
    except Exception as e:
        return {"verified": False, "similarity": 0.0, "message": f"Error: {str(e)}"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)