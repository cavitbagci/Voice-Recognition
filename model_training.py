import os
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from features import extract_features

# Ses dosyalarının bulunduğu klasör
audio_folder = "audio_samples"

features = []
labels = []

for file_name in os.listdir(audio_folder):
    if file_name.endswith(".wav"):
        # Dosya yolunu oluştur
        file_path = os.path.join(audio_folder, file_name)
        
        # Özellikleri çıkar
        feature = extract_features(file_path)
        if feature is not None:
            features.append(feature)
            
            # Etiketi dosya adına göre belirle
        if "cavit" in file_name:
            label = "Cavit"
        elif "fatihterim" in file_name:
                label = "Fatih Terim"
        elif "irmak" in file_name:
                label = "Irmak"
        elif "arda" in file_name:
                label = "Arda"
        elif "zeyden" in file_name:
                label = "Zeyden"
        elif "nevriz" in file_name:
                label = "Nevriz"
        elif "bilal" in file_name:
                label = "Bilal"
        labels.append(label)

X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.2, random_state=42)

model = RandomForestClassifier()
model.fit(X_train, y_train)

# Modeli değerlendir
y_pred = model.predict(X_test)
print("Doğruluk:", accuracy_score(y_test, y_pred))

import joblib
joblib.dump(model, "voice_model.pkl")
print("Model başarıyla kaydedildi.")
