import joblib
from features import extract_features
import speech_recognition as sr

model = joblib.load("voice_model.pkl")

recognizer = sr.Recognizer()

with sr.Microphone() as source:
    print("Konuşun...")
    audio = recognizer.listen(source)

    with open("real_time.wav", "wb") as f:
        f.write(audio.get_wav_data())

    features = extract_features("real_time.wav")
    if features is not None:
        prediction = model.predict([features])
        print("Tahmin edilen kişi:", prediction[0])
    else:
        print("Ses özellikleri çıkarılamadı.")
