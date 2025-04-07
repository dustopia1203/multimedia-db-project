from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from motor.motor_asyncio import AsyncIOMotorClient
import librosa
import numpy as np
import os
import tempfile
from datetime import datetime
from typing import List
import uvicorn
from pydantic import BaseModel

app = FastAPI(title="Audio Similarity API")

# Serve the `training-data` folder as static files
app.mount("/training-data", StaticFiles(directory="training-data"), name="training-data")

# CORS middleware setup
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# MongoDB connection setup
MONGODB_URL = "mongodb://localhost:27017/"
client = AsyncIOMotorClient(MONGODB_URL)
db = client.multimedia_db
audio_collection = db.audio_files

class AudioMetadata(BaseModel):
    filename: str
    filepath: str
    duration: float
    sample_rate: int
    similarity_score: float = None

@app.on_event("startup")
async def startup_db_client():
    """Initialize MongoDB client and load training data"""
    app.mongodb_client = AsyncIOMotorClient(MONGODB_URL)
    app.mongodb = app.mongodb_client.multimedia_db

    # Load training data from the 'training-data' folder
    app.training_data = []
    training_folder = "training-data"

    print(f"Loading training data from {training_folder}...")
    
    for filename in os.listdir(training_folder):
        if filename.endswith(".mp3"):
            file_path = os.path.join(training_folder, filename)
            features, sample_rate, duration = await extract_features(file_path)

            print(f"Loaded {filename}: {duration} seconds, {sample_rate} Hz")    

            # Save training data to MongoDB and retrieve the `_id`
            training_file = {
                "filename": filename,
                "filepath": file_path,
                "features": features,
                "duration": duration,
                "sample_rate": sample_rate,
                "timestamp": datetime.utcnow()
            }
            result = await audio_collection.insert_one(training_file)
            training_file["_id"] = str(result.inserted_id)  # Convert ObjectId to string

            app.training_data.append(training_file)

    # Ensure the training data is loaded
    if not app.training_data:
        raise HTTPException(status_code=500, detail="No training data found")
    
    # Print the number of training files loaded
    print(f"Loaded {len(app.training_data)} training files.")

@app.on_event("shutdown")
async def shutdown_db_client():
    """Close MongoDB client"""
    app.mongodb_client.close()

async def extract_features(file_path):
    """Extract robust audio features from the file"""
    y, sr = librosa.load(file_path, sr=22050)
    
    # Extract MFCCs
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=20)
    mfcc_mean = np.mean(mfcc.T, axis=0)
    
    # Extract Spectral Centroid
    spectral_centroid = librosa.feature.spectral_centroid(y=y, sr=sr)
    spectral_centroid_mean = np.mean(spectral_centroid)
    
    # Extract Chroma Features
    chroma = librosa.feature.chroma_stft(y=y, sr=sr)
    chroma_mean = np.mean(chroma.T, axis=0)
    
    # Extract Spectral Contrast
    spectral_contrast = librosa.feature.spectral_contrast(y=y, sr=sr)
    spectral_contrast_mean = np.mean(spectral_contrast.T, axis=0)
    
    # Extract Zero-Crossing Rate
    zero_crossing_rate = librosa.feature.zero_crossing_rate(y)
    zero_crossing_rate_mean = np.mean(zero_crossing_rate)
    
    # Combine all features into a single array
    features = np.concatenate([
        mfcc_mean,
        [spectral_centroid_mean],
        chroma_mean,
        spectral_contrast_mean,
        [zero_crossing_rate_mean]
    ])
    
    return features.tolist(), sr, len(y) / sr

async def calculate_similarity(features1, features2):
    """Calculate cosine similarity between normalized feature vectors"""
    # Normalize the feature vectors
    norm_features1 = features1 / np.linalg.norm(features1)
    norm_features2 = features2 / np.linalg.norm(features2)
    
    # Calculate cosine similarity
    return 1 - np.dot(norm_features1, norm_features2)

@app.post("/upload/", response_model=List[AudioMetadata])
async def upload_audio(file: UploadFile = File(...)):
    """Upload a test MP3 file and find the 3 nearest files from training data"""
    if not file.filename.lower().endswith(".mp3"):
        raise HTTPException(status_code=400, detail="File must be an MP3")
    
    # Save uploaded file temporarily
    temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".mp3")
    try:
        # Write the uploaded file content to the temporary file
        content = await file.read()
        with open(temp_file.name, "wb") as f:
            f.write(content)
            
        # Explicitly close the file to release the lock
        temp_file.close()  
        
        # Extract features from the uploaded test file
        features, sample_rate, duration = await extract_features(temp_file.name)
        
        # Compare with training data
        similarities = []
        for train_file in app.training_data:
            similarity = await calculate_similarity(features, train_file["features"])
            similarities.append({
                "id": train_file["_id"],
                "filename": train_file["filename"],
                "filepath": train_file["filepath"],
                "duration": train_file["duration"],
                "sample_rate": train_file["sample_rate"],
                "similarity_score": similarity
            })
        
        # Sort by similarity (lower score = more similar)
        similarities.sort(key=lambda x: x["similarity_score"])

        # Return top 3 most similar files
        return similarities[:3]  
    
    finally:
        # Ensure the temporary file is deleted
        if os.path.exists(temp_file.name):
            os.unlink(temp_file.name)

if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
