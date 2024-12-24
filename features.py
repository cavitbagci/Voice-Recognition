import librosa
import numpy as np

def extract_features(audio_path):
    try:
        y, sr = librosa.load(audio_path, sr=16000)
        
        mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=40)
        
        mfcc_mean = np.mean(mfcc.T, axis=0)
        return mfcc_mean
    except Exception as e:
        print(f"Hata oluştu: {e}")
        return None
