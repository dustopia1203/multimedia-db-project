import librosa
import numpy as np
import matplotlib.pyplot as plt
import io
import base64
from matplotlib.figure import Figure
import librosa.display

async def extract_features(file_path):
    """Extract robust audio features from the file, optimized for wind instruments"""
    y, sr = librosa.load(file_path, sr=22050)
    
    # 1. Extract MFCCs (20 coefficients)
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=20)
    mfcc_mean = np.mean(mfcc.T, axis=0)
    mfcc_var = np.var(mfcc.T, axis=0)
    
    # 2. Extract Chroma Features (12 dimensions)
    chroma = librosa.feature.chroma_stft(y=y, sr=sr)
    chroma_mean = np.mean(chroma.T, axis=0)
    
    # 3. Extract Zero-Crossing Rate
    zero_crossing_rate = librosa.feature.zero_crossing_rate(y)
    zero_crossing_rate_mean = np.mean(zero_crossing_rate)
    zero_crossing_rate_var = np.var(zero_crossing_rate)
    
    # 4. Extract Spectral Centroid - represents "brightness" of sound
    spectral_centroid = librosa.feature.spectral_centroid(y=y, sr=sr)
    spectral_centroid_mean = np.mean(spectral_centroid)
    spectral_centroid_var = np.var(spectral_centroid)
    
    # 5. Extract Spectral Roll-off - distinguishes "thin" vs "thick" timbre
    spectral_rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr, roll_percent=0.9)
    rolloff_mean = np.mean(spectral_rolloff)
    rolloff_var = np.var(spectral_rolloff)
    
    # 6. Extract Spectral Flux - captures transients and attack/release
    # Calculate frame-by-frame spectral difference
    hop_length = 512
    stft = np.abs(librosa.stft(y, hop_length=hop_length))
    spectral_flux = np.sum(np.diff(stft, axis=1)**2, axis=0)
    spectral_flux_mean = np.mean(spectral_flux)
    spectral_flux_var = np.var(spectral_flux)
    
    # 7. Extract RMS Energy - helps with loudness characteristics
    rms = librosa.feature.rms(y=y)
    rms_mean = np.mean(rms)
    rms_var = np.var(rms)
    
    # 8. Extract Spectral Contrast - emphasizes differences between peaks and valleys
    spectral_contrast = librosa.feature.spectral_contrast(y=y, sr=sr)
    spectral_contrast_mean = np.mean(spectral_contrast.T, axis=0)
    
    # Collect all individual feature components
    feature_details = {
        "mfcc_mean": mfcc_mean.tolist(),
        "mfcc_var": mfcc_var.tolist(),
        "chroma_mean": chroma_mean.tolist(),
        "zero_crossing_rate_mean": float(zero_crossing_rate_mean),
        "zero_crossing_rate_var": float(zero_crossing_rate_var),
        "spectral_centroid_mean": float(spectral_centroid_mean),
        "spectral_centroid_var": float(spectral_centroid_var),
        "rolloff_mean": float(rolloff_mean),
        "rolloff_var": float(rolloff_var),
        "spectral_flux_mean": float(spectral_flux_mean),
        "spectral_flux_var": float(spectral_flux_var),
        "rms_mean": float(rms_mean),
        "rms_var": float(rms_var),
        "spectral_contrast_mean": spectral_contrast_mean.tolist()
    }
    
    # Combine all features into a single array
    features = np.concatenate([
        mfcc_mean,                  # 20 features - timbre characteristics
        mfcc_var,                   # 20 features - timbre variability  
        chroma_mean,                # 12 features - harmonic content
        [zero_crossing_rate_mean],  # 1 feature - oscillation properties
        [zero_crossing_rate_var],   # 1 feature - stability of oscillation
        [spectral_centroid_mean],   # 1 feature - brightness
        [spectral_centroid_var],    # 1 feature - brightness stability
        [rolloff_mean],             # 1 feature - timbral thickness
        [rolloff_var],              # 1 feature - timbre stability
        [spectral_flux_mean],       # 1 feature - attack characteristics
        [spectral_flux_var],        # 1 feature - attack variation
        [rms_mean],                 # 1 feature - loudness
        [rms_var],                  # 1 feature - dynamic range
        spectral_contrast_mean      # 7 features - detailed spectral shape
    ])
    
    # Generate spectrogram image
    spectrogram = generate_spectrogram(y, sr)
    
    return features.tolist(), feature_details, sr, len(y) / sr, spectrogram

def generate_spectrogram(y, sr):
    """Generate and return a spectrogram as a base64 encoded string"""
    # Create figure with no margins
    fig = Figure(figsize=(10, 4))
    ax = fig.add_subplot(111)
    
    # Compute spectrogram
    D = librosa.amplitude_to_db(np.abs(librosa.stft(y)), ref=np.max)
    
    # Plot spectrogram
    img = librosa.display.specshow(D, sr=sr, x_axis='time', y_axis='log', ax=ax)
    ax.set_title('Power spectrogram')
    fig.colorbar(img, ax=ax, format="%+2.0f dB")
    
    # Save figure to a binary buffer
    buf = io.BytesIO()
    fig.savefig(buf, format='png')
    plt.close(fig)
    
    # Encode the buffer as base64
    buf.seek(0)
    img_str = base64.b64encode(buf.getvalue()).decode('utf-8')
    
    return img_str

async def calculate_similarity(features1, features2):
    """Calculate cosine similarity between normalized feature vectors,
    with proper scaling to prevent any feature group from dominating"""
    # Convert to numpy arrays if they aren't already
    features1 = np.array(features1)
    features2 = np.array(features2)
    
    # Define feature group indices for targeted normalization
    mfcc_mean_indices = range(0, 20)  # First 20 features are MFCC means
    mfcc_var_indices = range(20, 40)  # Next 20 are MFCC variances
    chroma_indices = range(40, 52)    # Next 12 are chroma features
    scalar_indices = range(52, 67)    # Remaining are scalar features
    
    # Initialize normalized feature arrays
    norm_features1 = np.zeros_like(features1, dtype=float)
    norm_features2 = np.zeros_like(features2, dtype=float)
    
    # Apply group-wise normalization with weights
    feature_groups = [
        {"indices": mfcc_mean_indices, "weight": 1.0},
        {"indices": mfcc_var_indices, "weight": 0.8},  # Slightly less weight to variance
        {"indices": chroma_indices, "weight": 1.2},    # Slightly more weight to pitch info
        {"indices": scalar_indices, "weight": 0.9}     # Slightly less weight to scalar features
    ]
    
    for group in feature_groups:
        indices = group["indices"]
        weight = group["weight"]
        
        # Skip empty groups
        if len(indices) == 0:
            continue
        
        # Get the feature group for both vectors
        group1 = features1[indices]
        group2 = features2[indices]
        
        # Calculate range for normalization
        min_val = min(np.min(group1), np.min(group2))
        max_val = max(np.max(group1), np.max(group2))
        
        # Avoid division by zero
        range_val = max_val - min_val
        if range_val < 1e-10:
            range_val = 1.0
        
        # Normalize to [0,1] range
        norm_features1[indices] = weight * (features1[indices] - min_val) / range_val
        norm_features2[indices] = weight * (features2[indices] - min_val) / range_val
    
    # Calculate L2 norm
    norm1 = np.linalg.norm(norm_features1)
    norm2 = np.linalg.norm(norm_features2)
    
    # Avoid division by zero
    if norm1 < 1e-10:
        norm1 = 1.0
    if norm2 < 1e-10:
        norm2 = 1.0
    
    # Normalize vectors
    norm_features1 = norm_features1 / norm1
    norm_features2 = norm_features2 / norm2
    
    # Calculate cosine similarity
    similarity = np.dot(norm_features1, norm_features2)
    
    # Return distance measure (1 - similarity) so smaller = more similar
    return 1 - similarity