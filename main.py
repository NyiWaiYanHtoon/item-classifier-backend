import os
import io
import re
from typing import List
from contextlib import nullcontext

from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from PIL import Image
import torch
import numpy as np
import open_clip
from pymongo import MongoClient
from dotenv import load_dotenv

# Load .env
load_dotenv()

# ========== Config ==========
MONGODB_URI = os.environ.get("MONGODB_URI", "mongodb://localhost:27017")
MONGODB_DB = os.environ.get("MONGODB_DB", "visual_search")
MODEL_NAME = os.environ.get("MODEL_NAME", "hf-hub:Marqo/marqo-ecommerce-embeddings-L")
ALLOWED_ORIGINS = os.environ.get("ALLOWED_ORIGINS", "*")

# Connect to MongoDB
client = MongoClient(MONGODB_URI)
db = client[MONGODB_DB]
items_collection = db["items"]

# FastAPI app + CORS
app = FastAPI(title="Image -> Category -> DB matches (OpenCLIP)")

origins = [o.strip() for o in ALLOWED_ORIGINS.split(",")] if ALLOWED_ORIGINS != "*" else ["*"]
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Device
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

# ========== Model loading (OpenCLIP) ==========
print("Loading OpenCLIP model and transforms (this can take a while)...")
try:
    # The user-provided MODEL_NAME (example: 'hf-hub:Marqo/marqo-ecommerce-embeddings-L')
    # open_clip.create_model_and_transforms accepts a model identifier. Depending on the open_clip version,
    # MODEL_NAME might need to be like 'ViT-H-14' + pretrained argument. The Marqo repo should be loadable
    # via the 'hf-hub:' prefix if open_clip supports it. If this fails, see the fallback message printed below.
    model, _, preprocess = open_clip.create_model_and_transforms(MODEL_NAME)
    # get tokenizer for that model; fallback to model name if needed
    try:
        tokenizer = open_clip.get_tokenizer(MODEL_NAME)
    except Exception:
        # Try to get tokenizer by model's name (safe fallback)
        tokenizer = open_clip.get_tokenizer(model.name)
except Exception as e:
    # Provide a helpful error message
    raise RuntimeError(
        f"Failed to load OpenCLIP model with create_model_and_transforms('{MODEL_NAME}'). "
        "Make sure you installed open_clip properly and MODEL_NAME is valid for open_clip. "
        "Original error: " + str(e)
    )

# Move model to device
model = model.to(device)
model.eval()

# ========== Category list and precompute text embeddings ==========
ECOMMERCE_ITEMS = [
    "laptop", "notebook", "microwave", "toothbrush", "plates", "glasses",
    "bicycle helmet", "perfume", "book", "hair straightener"
]

# Tokenize text list and compute text embeddings once
print("Tokenizing and encoding text categories...")
# tokenizer(...) usually returns a torch.LongTensor ready for model.encode_text
text_tokens = tokenizer(ECOMMERCE_ITEMS)  # may be tensor or dict depending on tokenizer impl
# Ensure it's a tensor and moved to device
if hasattr(text_tokens, "to"):
    text_tokens = text_tokens.to(device)
elif isinstance(text_tokens, (list, tuple)):
    # Some tokenizers return lists of tensors or lists of token ids - convert to tensor if possible
    try:
        text_tokens = torch.tensor(text_tokens).to(device)
    except Exception:
        raise RuntimeError("Tokenizer output format not recognized. Inspect open_clip.get_tokenizer output.")
else:
    # If it's a dict with input_ids
    if isinstance(text_tokens, dict) and "input_ids" in text_tokens:
        text_tokens = text_tokens["input_ids"].to(device)

with torch.no_grad():
    # Some open_clip models require different method names; encode_text should be available.
    text_features = model.encode_text(text_tokens)
    text_features = text_features / text_features.norm(dim=-1, keepdim=True)  # normalize
    # move to CPU for cheaper storage & safe reuse
    text_features_cpu = text_features.cpu()

class ItemOut(BaseModel):
    name: str
    price: float
    categories: List[str]
    image_url: str = None
    _id: str = None

# ========== Helper: safe autocast context depending on device ==========
def amp_scope():
    # Use CUDA autocast only when using cuda; on CPU use nullcontext
    if device == "cuda":
        return torch.cuda.amp.autocast()
    else:
        return nullcontext()

# ========== API endpoints ==========
@app.post("/classify")
async def classify_image(file: UploadFile = File(...), top_k: int = 5):
    # Read uploaded file bytes
    contents = await file.read()
    try:
        image = Image.open(io.BytesIO(contents)).convert("RGB")
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Could not read image: {e}")

    # Preprocess image with the transform returned by open_clip.create_model_and_transforms
    try:
        image_input = preprocess(image).unsqueeze(0).to(device)  # shape [1, C, H, W]
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Preprocessing failed: {e}")

    with torch.no_grad(), amp_scope():
        image_features = model.encode_image(image_input)
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)

        # similarity between image_features (1, D) and text_features_cpu (N, D)
        # move text features to device for computation
        text_feat_dev = text_features_cpu.to(device)
        logits = (100.0 * image_features @ text_feat_dev.T).softmax(dim=-1)
        scores = logits.cpu().numpy()[0]  # numpy array of length len(ECOMMERCE_ITEMS)

    # top-1
    top_idx = int(scores.argmax())
    top_category = ECOMMERCE_ITEMS[top_idx]
    confidence = float(scores[top_idx])

    # Query MongoDB using case-insensitive regex to match category in document categories
    regex = re.compile(f"^{re.escape(top_category)}$", re.IGNORECASE)
    cursor = items_collection.find({"categories": {"$in": [regex]}}).limit(int(top_k))
    matches = []
    for doc in cursor:
        matches.append({
            "name": doc.get("name"),
            "price": doc.get("price"),
            "categories": doc.get("categories"),
            "image_url": doc.get("image_url", None),
            "_id": str(doc.get("_id"))
        })

    return {
        "top_category": top_category,
        "confidence": confidence,
        "matches": matches
    }

@app.get("/items")
def list_items(limit: int = 50):
    docs = items_collection.find().limit(limit)
    out = []
    for d in docs:
        out.append({
            "name": d.get("name"),
            "price": d.get("price"),
            "categories": d.get("categories"),
            "image_url": d.get("image_url", None),
            "_id": str(d.get("_id"))
        })
    return {"items": out}
