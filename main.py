from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import pandas as pd
import numpy as np
import tensorflow as tf
import joblib
import os
import json
from dotenv import load_dotenv
import google.generativeai as genai

# Load environment variables dari file .env
load_dotenv()

# Konfigurasi Gemini API
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
if not GEMINI_API_KEY:
    print("GEMINI_API_KEY tidak ditemukan")
else:
    genai.configure(api_key=GEMINI_API_KEY)

# ==========================================
# INISIALISASI APLIKASI & LOAD DEEP LEARNING
# ==========================================
app = FastAPI(title="AI Health Coach API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], # Mengizinkan akses dari semua frontend (untuk development)
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

try:
    model_dnn = tf.keras.models.load_model('dnn_lifestyle_model.keras')
    scaler_loaded = joblib.load('scaler.pkl')
    training_cols = joblib.load('training_columns.pkl')
    target_classes = joblib.load('class_names.pkl')
    print("Model Deep Learning siap!")
except Exception as e:
    print(f"Error load model: {e}")

# ==========================================
# SKEMA INPUT FRONTEND
# ==========================================
class UserInput(BaseModel):
    # --- Variabel Tambahan Khusus untuk LLM Prompt ---
    Name: str
    Target_Weight_Loss: float # Target turun dalam kg

    # --- Variabel untuk Deep Learning ---
    Height: float 
    Weight: float
    Gender: str
    Age: int

# ==========================================
# ENDPOINT --> PREDIKSI + GENERATE QUEST
# ==========================================
@app.post("/api/generate_quests")
async def generate_quests(user: UserInput):
    try:
        user_bmi = round(user.Weight / (user.Height ** 2), 1)
        
        # Menentukan Kategori BMI Faktual secara statis
        if user_bmi < 18.5:
            bmi_category = "Underweight"
        elif 18.5 <= user_bmi < 25.0:
            bmi_category = "Normal"
        elif 25.0 <= user_bmi < 30.0:
            bmi_category = "Overweight"
        else:
            bmi_category = "Obese"

        # 1. --- FASE DEEP LEARNING (PREDIKSI RISIKO) ---
        user_df = pd.DataFrame([user.model_dump()])
        user_encoded = pd.get_dummies(user_df)

        # MAGIC TRICK: Fitur Name, Height, Weight, dll akan OTOMATIS DIBUANG di sini
        # karena kolom tersebut tidak ada di dalam 'training_cols'. Model DL tetap aman!
        user_aligned = user_encoded.reindex(columns=training_cols, fill_value=0)
        user_scaled = scaler_loaded.transform(user_aligned)
        
        pred_probs = model_dnn.predict(user_scaled, verbose=0)[0]
        pred_index = np.argmax(pred_probs)
        risk_class = target_classes[pred_index]
        
        # 2. --- FASE GENERATIVE AI (LLM GEMINI) ---
        # Pilih model Gemini 1.5 Flash karena paling cepat dan gratis
        model_llm = genai.GenerativeModel('gemini-2.5-flash')
        
        # Merangkai Prompt sesuai instruksi Anda
        system_prompt = f"""
        Kamu adalah AI Health Coach pada aplikasi gamifikasi. Buatkan "Daily Quests" selama 7 hari.
        Setiap hari harus memiliki 1 hingga 3 quest (Pilihan kategori: Nutrisi, Olahraga, atau Kebiasaan).
        
        === DATA PENGGUNA ===
        - Nama: {user.Name}
        - Umur: {user.Age} tahun
        - Gender: {user.Gender}
        - BMI Faktual: {user_bmi} (Kategori manual: {bmi_category})
        - Target Turun Berat Badan: {user.Target_Weight_Loss} Kg

        === ANALISIS AI DEEP LEARNING (SANGAT PENTING) ===
        Berdasarkan ekstraksi metrik profil fisiknya yang spesifik (tingkat obesitas), model Deep Learning kami mengklasifikasikan kondisi pengguna saat ini pada kategori: **{risk_class}**.
        
        ATURAN TINGKAT KESULITAN:
        - Jika "{risk_class}" adalah Normal_Weight / Insufficient_Weight: Berikan quest fisik menengah (push up, lari, dsb).
        - Jika "{risk_class}" adalah Overweight (Level I/II): Berikan quest fisik ringan (jalan kaki 5000 langkah, kurangi porsi ngemil).
        - Jika "{risk_class}" adalah Obesity (Type_I, Type_II, atau Type_III): DILARANG KERAS memberikan olahraga yang melompat atau membebani lutut. Fokus utama pada nutrisi (kontrol porsi, defisit kalori, minum banyak air) dan pergerakan sangat ringan (jalan santai, peregangan duduk).
        
        WAJIB OUTPUT DALAM FORMAT JSON BERIKUT, TANPA TEKS LAINNYA:
        {{
          "schedule": [
            {{
              "day": 1,
              "quests": [
                {{"quest_name": "...", "description": "...", "category": "...", "xp_reward": 50}}
              ]
            }}
          ]
        }}
        """

        # Konfigurasi JSON Mode pada Gemini
        # Ini memastikan Gemini TIDAK akan menjawab dengan bahasa manusia, melainkan murni JSON
        generation_config = genai.GenerationConfig(
            response_mime_type="application/json",
            temperature=0.7 # 0.7 memberikan sedikit kreativitas pada nama quest
        )

        # Memanggil API Gemini (await --> FastAPI bersifat asinkron)
        response = await model_llm.generate_content_async(
            system_prompt,
            generation_config=generation_config
        )
        
        # 3. --- MENGEMBALIKAN RESPONSE KE FRONTEND ---
        # Parse output JSON dari Gemini agar menjadi object Python yang rapi di FastAPI
        llm_quests = json.loads(response.text)

        return {
            "status": "success",
            "ai_analysis": {
                "factual_bmi": user_bmi,
                "factual_category": bmi_category,
                "risk_class": risk_class,
            },
            "gamification_data": llm_quests
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/")
async def root():
    return {"message": "Server API Aktif!"}