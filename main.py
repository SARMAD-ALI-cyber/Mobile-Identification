import os
import numpy as np
import librosa
import joblib
from fastapi.responses import JSONResponse
# from sklearn.mixture import GaussianMixture
# from sklearn.preprocessing import StandardScaler
# from sklearn.model_selection import train_test_split
# from sklearn.svm import SVC
# from sklearn.preprocessing import LabelEncoder
# from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from io import BytesIO

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Your frontend development server
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
    expose_headers=["*"]  # Add this to expose necessary headers
)
# Load trained models and preprocessor
svm_model = joblib.load("svm_modelv2.pkl")
scaler_sgmm = joblib.load("scaler_sgmm.pkl")
label_encoder = joblib.load("label_encoder.pkl")
ubm_model = joblib.load("ubm64.pkl")

# MFCC extraction parameters
N_MFCC = 13
FRAME_SIZE = 0.025  # 25ms
FRAME_STEP = 0.010  # 10ms
N_MIXTURES = 64  # Number of components for UBM
MAX_ITER = 200  # Max iterations for GMM
RANDOM_STATE = 42


# Function to extract MFCC features
def extract_mfcc(y,sr):
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=N_MFCC)
    
    # Compute delta and delta-delta features
    delta_mfcc = librosa.feature.delta(mfcc)
    delta2_mfcc = librosa.feature.delta(mfcc, order=2)
    
    # Stack features
    mfcc_features = np.vstack((mfcc, delta_mfcc, delta2_mfcc)).T
    return mfcc_features


# Function to adapt UBM to create SGMM
def extract_sgmm_features(mfcc_features, ubm):
    responsibilities = ubm.predict_proba(mfcc_features)  # Shape: (num_frames, num_mixtures)
    mean_responsibilities = responsibilities.mean(axis=0).reshape(-1, 1)  # Shape: (num_mixtures, 1)

    # Ensure correct stacking along the second axis
    sgmm_features = np.hstack((ubm.means_, ubm.covariances_, mean_responsibilities))

    return sgmm_features.flatten()

@app.get("/")
def hello():
    return {"Running 100%"}
@app.post("/predict/")
async def predict_audio(file: UploadFile = File(...)):
    try:
        # Read audio file
        audio_bytes = await file.read()
        y, sr = librosa.load(BytesIO(audio_bytes), sr=None)

        # Extract MFCC features
        mfcc_features = extract_mfcc(y, sr)
        print("mfcc_features.shape: ",mfcc_features.shape)
        # Extract SGMM features
        sgmm_features = extract_sgmm_features(mfcc_features, ubm_model)
        print("sgmm_features.shape: ",sgmm_features.shape)

        sgmm_features = sgmm_features.reshape(1, -1)  # Ensure correct shape
        print("sgmm_features.shape: ",sgmm_features.shape)
        # Apply the same scaling as during training
        sgmm_features_scaled = scaler_sgmm.transform(sgmm_features)
        sgmm_features_scaled=sgmm_features_scaled[:,:-1]
        print("sgmm_features_scaled.shape: ",sgmm_features_scaled.shape)
        # Predict device class
        y_pred = svm_model.predict(sgmm_features_scaled)

        # Convert prediction to mobile device name
        predicted_device = label_encoder.inverse_transform(y_pred)[0]

        response = JSONResponse(content={"predicted_device": predicted_device})
        response.headers["Access-Control-Allow-Origin"] = "*"
        return response

    except Exception as e:
        response = JSONResponse(content={"error": str(e)}, status_code=400)
        response.headers["Access-Control-Allow-Origin"] = "*"
        return response
    
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app",host="0.0.0.0",reload=True)