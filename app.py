# app.py
from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from PIL import Image
import io, json, os
import torch
from torchvision import transforms

# === Configuración ===
MODELS_DIR = os.getenv("MODELS_DIR", "models")
MODEL_PATH = os.path.join(MODELS_DIR, "model.pt")
LABELS_PATH = os.path.join(MODELS_DIR, "labels.json")

SCORE_BY_CLASS = {
    "carton": 10,
    "metal_aluminio": 9,
    "papel": 7,
    "plastico_hdpe": 8,
    "plastico_ldpe": 6,
    "plastico_pet": 8,
    "plastico_pp": 5,
    "plastico_ps": 5,
    "vidrio": 5,
    "no_reciclable": 0,
}
MIN_CONFIDENCE = 0.60  # umbral para otorgar puntos completos

# === Carga modelo/labels ===
if not (os.path.exists(MODEL_PATH) and os.path.exists(LABELS_PATH)):
    raise RuntimeError("Faltan model.pt o labels.json en /models")

model = torch.jit.load(MODEL_PATH, map_location="cpu")
model.eval()

with open(LABELS_PATH, "r", encoding="utf-8") as f:
    LABELS = json.load(f)

IMG_SIZE = 224
MEAN = [0.485, 0.456, 0.406]  # Imagenet
STD  = [0.229, 0.224, 0.225]

tfm = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(MEAN, STD),
])

app = FastAPI(title="ReciclaDUOC Classifier", version="1.0.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # en prod: restringe a tu dominio
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class Prediction(BaseModel):
    className: str
    probability: float

class ClassifyResponse(BaseModel):
    topClass: str
    confidence: float  # 0-100
    score: int
    awarded: bool
    predictions: list[Prediction]

@app.get("/health")
def health():
    return {"status": "ok", "labels": LABELS, "num_classes": len(LABELS)}

@app.post("/classify", response_model=ClassifyResponse)
async def classify(file: UploadFile = File(...)):
    # Leer bytes y abrir imagen
    img_bytes = await file.read()
    img = Image.open(io.BytesIO(img_bytes)).convert("RGB")

    # Preprocesar
    x = tfm(img).unsqueeze(0)  # [1,3,H,W]

    with torch.no_grad():
        logits = model(x)  # [1, num_classes]
        probs = torch.softmax(logits, dim=1)[0].tolist()

    ranked = sorted(
        [{"className": LABELS[i], "probability": float(p)} for i, p in enumerate(probs)],
        key=lambda r: r["probability"], reverse=True
    )

    top = ranked[0]

    conf = top["probability"]  # 0-1
    class_name = top["className"]
    score = SCORE_BY_CLASS.get(class_name, 0)

    # regla simple: premio completo si confianza >= MIN_CONFIDENCE
    awarded = conf >= MIN_CONFIDENCE

    return {
        "topClass": class_name,
        "confidence": round(conf * 100, 1),
        "score": int(score if awarded else round(score * 0.5)),  # 50% si dudosa
        "awarded": awarded,
        "predictions": ranked[:5],
    }
