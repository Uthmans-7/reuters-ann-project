from fastapi import FastAPI, HTTPException, Depends, status
from fastapi.security import HTTPBasic, HTTPBasicCredentials
from fastapi.responses import HTMLResponse
from pydantic import BaseModel
import numpy as np
import tensorflow as tf
from tensorflow.keras.datasets import reuters
import re
import string
import secrets
import base64

# --- CONFIGURATION ---
app = FastAPI(title="Reuters AI", description="Secure News Classifier")

# 🔐 ONE SINGLE LOGIN FOR EVERYTHING
ADMIN_USER = "admin"
ADMIN_PASS = "password123"

# --- SECURITY SETUP ---
security = HTTPBasic()

def verify_credentials(credentials: HTTPBasicCredentials = Depends(security)):
    """
    This function checks the Username & Password.
    It protects BOTH the UI page AND the Prediction API.
    """
    is_correct_user = secrets.compare_digest(credentials.username, ADMIN_USER)
    is_correct_pass = secrets.compare_digest(credentials.password, ADMIN_PASS)
    
    if not (is_correct_user and is_correct_pass):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect username or password",
            headers={"WWW-Authenticate": "Basic"},
        )
    return credentials

# --- MODEL LOADING ---
print("Loading Model...")
model = tf.keras.models.load_model('reuters_best_model.keras')
word_index = reuters.get_word_index()
reverse_word_index = {k: v + 3 for k, v in word_index.items()}
print("Model Loaded!")

# --- DATA HELPERS ---
TOPICS = {
    0: "COCOA", 1: "GRAIN", 2: "VEG-OIL", 3: "EARNINGS", 4: "ACQ", 5: "WHEAT", 
    6: "COPPER", 7: "HOUSING", 8: "MONEY-SUPPLY", 9: "RUBBER", 10: "SUGAR", 
    11: "TRADE", 12: "RESERVES", 13: "SHIP", 14: "COFFEE", 15: "INTEREST", 
    16: "CRUDE", 17: "LUMBER", 18: "LIVESTOCK", 19: "ALUM", 20: "GOLD", 
    21: "NAT-GAS", 22: "CORN", 23: "IPO", 24: "COTTON", 25: "SUGAR", 
    26: "TIN", 27: "NICKEL", 28: "BARLEY", 29: "STRATEGIC-METAL", 30: "VEG-OIL",
    31: "HOUSING", 32: "MONEY-FX", 33: "RETAIL", 34: "WPI", 35: "CPI", 
    36: "PET-CHEM", 37: "LEAD", 38: "GNP", 39: "PLATINUM", 40: "HEAT", 
    41: "ORANGE", 42: "UNKNOWN", 43: "COFFEE", 44: "CAN", 45: "POTATO"
}

def vectorize_text(text, dimension=10000):
    text = text.lower()
    text = re.sub(f'[{string.punctuation}]', '', text) 
    words = text.split()
    vector = np.zeros((1, dimension))
    for word in words:
        if word in reverse_word_index:
            index = reverse_word_index[word]
            if index < dimension:
                vector[0, index] = 1.
    return vector

class NewsItem(BaseModel):
    text: str

# --- ENDPOINTS ---

@app.get("/", response_class=HTMLResponse, tags=["System"])
async def root():
    return """
    <html>
        <head>
            <title>Reuters AI Classifier</title>
            <style>
                body { font-family: 'Segoe UI', sans-serif; background: #f0f2f5; display: flex; justify-content: center; align-items: center; height: 100vh; margin: 0; }
                .container { background: white; padding: 40px; border-radius: 12px; box-shadow: 0 10px 30px rgba(0,0,0,0.1); text-align: center; max-width: 400px; }
                h1 { color: #1a73e8; }
                p { color: #5f6368; }
                .btn { background: #1a73e8; color: white; padding: 12px 25px; text-decoration: none; border-radius: 6px; font-weight: bold; display: inline-block; margin-top: 20px; transition: 0.3s; }
                .btn:hover { background: #1557b0; }
            </style>
        </head>
        <body>
            <div class="container">
                <h1>📰 Reuters AI System</h1>
                <p>Protected Corporate System.</p>
                <a href="/predict-ui" class="btn">Login to Tool</a>
            </div>
        </body>
    </html>
    """

@app.get("/predict-ui", response_class=HTMLResponse, tags=["UI"])
async def predict_ui(creds: HTTPBasicCredentials = Depends(verify_credentials)):
    """
    PROTECTED PAGE: Browser asks for User/Pass.
    """
    return f"""
    <html>
        <head>
            <title>AI Workspace</title>
            <style>
                body {{ font-family: 'Segoe UI', sans-serif; background-color: #eef2f6; display: flex; justify-content: center; padding-top: 50px; }}
                .card {{ background: white; padding: 40px; border-radius: 15px; box-shadow: 0 5px 20px rgba(0,0,0,0.05); width: 600px; }}
                h2 {{ color: #2c3e50; border-bottom: 2px solid #f0f0f0; padding-bottom: 10px; }}
                textarea {{ width: 100%; height: 150px; padding: 15px; border: 2px solid #e0e0e0; border-radius: 8px; margin: 20px 0; font-family: inherit; resize: vertical; }}
                button {{ background: #27ae60; color: white; border: none; padding: 15px; width: 100%; border-radius: 8px; font-size: 16px; cursor: pointer; font-weight: bold; }}
                button:hover {{ background: #219150; }}
                #result {{ margin-top: 25px; padding: 20px; border-radius: 8px; display: none; }}
                .success {{ background: #d4edda; color: #155724; border: 1px solid #c3e6cb; }}
                .user-badge {{ float: right; font-size: 12px; color: #7f8c8d; background: #f0f0f0; padding: 5px 10px; border-radius: 20px; }}
            </style>
        </head>
        <body>
            <div class="card">
                <span class="user-badge">👤 User: {creds.username}</span>
                <h2>🚀 News Analyzer Tool</h2>
                <p>Paste your article text below:</p>
                <textarea id="newsInput" placeholder="Type news here..."></textarea>
                <button onclick="analyze()">Analyze Topic</button>
                
                <div id="result" class="success">
                    <h3 id="category" style="margin: 0; font-size: 24px;"></h3>
                    <p>Confidence: <span id="confidence"></span></p>
                </div>
            </div>

            <script>
                // AUTOMATION: Use the credentials the user JUST entered
                const USER = "{creds.username}";
                const PASS = "{creds.password}";

                async function analyze() {{
                    const text = document.getElementById('newsInput').value;
                    const resDiv = document.getElementById('result');
                    
                    // Create the "Basic Auth" header automatically
                    const authString = btoa(USER + ":" + PASS);

                    const response = await fetch('/predict', {{
                        method: 'POST',
                        headers: {{ 
                            'Content-Type': 'application/json',
                            'Authorization': 'Basic ' + authString
                        }},
                        body: JSON.stringify({{ text: text }})
                    }});
                    
                    const data = await response.json();
                    if (response.ok) {{
                        document.getElementById('category').innerText = "Topic: " + data.prediction;
                        document.getElementById('confidence').innerText = data.confidence;
                        resDiv.style.display = 'block';
                    }} else {{
                        alert("Error: " + data.detail);
                    }}
                }}
            </script>
        </body>
    </html>
    """

@app.post("/predict")
async def predict_category(item: NewsItem, creds: HTTPBasicCredentials = Depends(verify_credentials)):
    """
    PROTECTED API: Also requires User/Pass.
    """
    try:
        processed_data = vectorize_text(item.text)
        prediction_probs = model.predict(processed_data, verbose=0)
        class_id = int(np.argmax(prediction_probs))
        confidence = float(np.max(prediction_probs))
        label = TOPICS.get(class_id, "Unknown")
        return {"prediction": label, "class_id": class_id, "confidence": f"{confidence:.2%}"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
async def health():
    return {"status": "ok"}