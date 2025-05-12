from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse, JSONResponse
from motor.motor_asyncio import AsyncIOMotorClient
import os
import tempfile
from datetime import datetime
from typing import List, Dict, Any
from bson import ObjectId
import uvicorn
from pydantic import BaseModel
from process_data import extract_features, calculate_similarity

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

class FeatureDetails(BaseModel):
    mfcc_mean: List[float]
    mfcc_var: List[float]
    chroma_mean: List[float]
    zero_crossing_rate: Dict[str, float]
    spectral_centroid: Dict[str, float]
    spectral_rolloff: Dict[str, float]
    spectral_flux: Dict[str, float]
    rms_energy: Dict[str, float]
    spectral_contrast: List[float]

class AudioMetadata(BaseModel):
    id: str
    filename: str
    filepath: str
    duration: float
    sample_rate: int
    similarity_score: float = None
    spectrogram: str = None
    feature_details: FeatureDetails = None

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
            features, feature_details, sample_rate, duration, spectrogram = await extract_features(file_path)

            print(f"Loaded {filename}: {duration} seconds, {sample_rate} Hz")    

            # Save training data to MongoDB and retrieve the `_id`
            training_file = {
                "filename": filename,
                "filepath": file_path,
                "features": features,
                "feature_details": {
                    "mfcc_mean": feature_details["mfcc_mean"],
                    "mfcc_var": feature_details["mfcc_var"],
                    "chroma_mean": feature_details["chroma_mean"],
                    "zero_crossing_rate": {
                        "mean": feature_details["zero_crossing_rate_mean"],
                        "var": feature_details["zero_crossing_rate_var"]
                    },
                    "spectral_centroid": {
                        "mean": feature_details["spectral_centroid_mean"],
                        "var": feature_details["spectral_centroid_var"]
                    },
                    "spectral_rolloff": {
                        "mean": feature_details["rolloff_mean"],
                        "var": feature_details["rolloff_var"]
                    },
                    "spectral_flux": {
                        "mean": feature_details["spectral_flux_mean"],
                        "var": feature_details["spectral_flux_var"]
                    },
                    "rms_energy": {
                        "mean": feature_details["rms_mean"],
                        "var": feature_details["rms_var"]
                    },
                    "spectral_contrast": feature_details["spectral_contrast_mean"].tolist() if hasattr(feature_details["spectral_contrast_mean"], "tolist") else feature_details["spectral_contrast_mean"]
                },
                "duration": duration,
                "sample_rate": sample_rate,
                "spectrogram": spectrogram,
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

@app.post("/upload")
async def upload_audio_file(file: UploadFile = File(...)):
    """Upload an audio file and return its metadata without comparison"""
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
        
        # Extract features from the uploaded file
        features, feature_details, sample_rate, duration, spectrogram = await extract_features(temp_file.name)
        
        # Store the uploaded file with its features
        uploaded_file = {
            "filename": file.filename,
            "features": features,
            "feature_details": {
                "mfcc_mean": feature_details["mfcc_mean"],
                "mfcc_var": feature_details["mfcc_var"],
                "chroma_mean": feature_details["chroma_mean"],
                "zero_crossing_rate": {
                    "mean": feature_details["zero_crossing_rate_mean"],
                    "var": feature_details["zero_crossing_rate_var"]
                },
                "spectral_centroid": {
                    "mean": feature_details["spectral_centroid_mean"],
                    "var": feature_details["spectral_centroid_var"]
                },
                "spectral_rolloff": {
                    "mean": feature_details["rolloff_mean"],
                    "var": feature_details["rolloff_var"]
                },
                "spectral_flux": {
                    "mean": feature_details["spectral_flux_mean"],
                    "var": feature_details["spectral_flux_var"]
                },
                "rms_energy": {
                    "mean": feature_details["rms_mean"],
                    "var": feature_details["rms_var"]
                },
                "spectral_contrast": feature_details["spectral_contrast_mean"].tolist() if hasattr(feature_details["spectral_contrast_mean"], "tolist") else feature_details["spectral_contrast_mean"]
            },
            "duration": duration,
            "sample_rate": sample_rate,
            "spectrogram": spectrogram,
            "timestamp": datetime.utcnow(),
            "is_query": True
        }

        # Return the metadata with ID
        return {
            "duration": duration,
            "sample_rate": sample_rate,
            "spectrogram": spectrogram,
            "feature_details": uploaded_file["feature_details"]
        }
    
    finally:
        # Ensure the temporary file is deleted
        if os.path.exists(temp_file.name):
            os.unlink(temp_file.name)

@app.post("/upload/q", response_model=List[AudioMetadata])
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
        features, feature_details, sample_rate, duration, spectrogram = await extract_features(temp_file.name)
        
        # Store the uploaded file with its features
        uploaded_file = {
            "filename": file.filename,
            "filepath": "uploads/" + file.filename,
            "features": features,
            "feature_details": {
                "mfcc_mean": feature_details["mfcc_mean"],
                "mfcc_var": feature_details["mfcc_var"],
                "chroma_mean": feature_details["chroma_mean"],
                "zero_crossing_rate": {
                    "mean": feature_details["zero_crossing_rate_mean"],
                    "var": feature_details["zero_crossing_rate_var"]
                },
                "spectral_centroid": {
                    "mean": feature_details["spectral_centroid_mean"],
                    "var": feature_details["spectral_centroid_var"]
                },
                "spectral_rolloff": {
                    "mean": feature_details["rolloff_mean"],
                    "var": feature_details["rolloff_var"]
                },
                "spectral_flux": {
                    "mean": feature_details["spectral_flux_mean"],
                    "var": feature_details["spectral_flux_var"]
                },
                "rms_energy": {
                    "mean": feature_details["rms_mean"],
                    "var": feature_details["rms_var"]
                },
                "spectral_contrast": feature_details["spectral_contrast_mean"].tolist() if hasattr(feature_details["spectral_contrast_mean"], "tolist") else feature_details["spectral_contrast_mean"]
            },
            "duration": duration,
            "sample_rate": sample_rate,
            "spectrogram": spectrogram,
            "timestamp": datetime.utcnow(),
            "is_query": True
        }
        
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
                "similarity_score": similarity,
                "spectrogram": train_file["spectrogram"],
                "feature_details": train_file.get("feature_details")
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