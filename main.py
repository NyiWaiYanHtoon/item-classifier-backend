import os
import io
import re
import base64
from typing import List
from contextlib import nullcontext

from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from PIL import Image, ImageEnhance
import torch
import numpy as np
import open_clip
from pymongo import MongoClient
from dotenv import load_dotenv

# image processing libs
import cv2
from skimage import exposure, filters, color, util

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
app = FastAPI(title="Image -> Category -> DB matches (OpenCLIP + preprocessing)")

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
    model, _, preprocess = open_clip.create_model_and_transforms(MODEL_NAME)
    try:
        tokenizer = open_clip.get_tokenizer(MODEL_NAME)
    except Exception:
        tokenizer = open_clip.get_tokenizer(model.name)
except Exception as e:
    raise RuntimeError(
        f"Failed to load OpenCLIP model with create_model_and_transforms('{MODEL_NAME}'). "
        "Make sure open-clip is installed and MODEL_NAME is valid. Original error: " + str(e)
    )

model = model.to(device)
model.eval()

# ========== Category list and precompute text embeddings ==========
ECOMMERCE_ITEMS = [
    "laptop", "notebook", "microwave", "toothbrush", "plates", "glasses",
    "bicycle helmet", "perfume", "book", "hair straightener"
]

print("Tokenizing and encoding text categories...")
text_tokens = tokenizer(ECOMMERCE_ITEMS)
if hasattr(text_tokens, "to"):
    text_tokens = text_tokens.to(device)
elif isinstance(text_tokens, (list, tuple)):
    try:
        text_tokens = torch.tensor(text_tokens).to(device)
    except Exception:
        raise RuntimeError("Tokenizer output format not recognized.")
else:
    if isinstance(text_tokens, dict) and "input_ids" in text_tokens:
        text_tokens = text_tokens["input_ids"].to(device)

with torch.no_grad():
    text_features = model.encode_text(text_tokens)
    text_features = text_features / text_features.norm(dim=-1, keepdim=True)
    text_features_cpu = text_features.cpu()

class ItemOut(BaseModel):
    name: str
    price: float
    categories: List[str]
    image_url: str = None
    _id: str = None

# ========== Helper functions ==========

def pil_to_cv2(img_pil: Image.Image):
    """Convert PIL image to OpenCV BGR"""
    arr = np.array(img_pil)
    if arr.ndim == 2:
        return arr
    return cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)

def cv2_to_pil(img_cv2):
    """Convert OpenCV BGR/gray to PIL RGB"""
    if img_cv2.ndim == 2:
        return Image.fromarray(img_cv2)
    rgb = cv2.cvtColor(img_cv2, cv2.COLOR_BGR2RGB)
    return Image.fromarray(rgb)

def pil_to_data_url(img_pil: Image.Image, fmt="JPEG", quality=85):
    """Encode PIL.Image to base64 data URL (thumbnail-sized)"""
    # create a thumbnail to reduce payload
    max_size = (800, 800)
    img_copy = img_pil.copy()
    img_copy.thumbnail(max_size, Image.Resampling.LANCZOS)
    buffer = io.BytesIO()
    img_copy.save(buffer, format=fmt, quality=quality)
    b64 = base64.b64encode(buffer.getvalue()).decode("utf-8")
    return f"data:image/{fmt.lower()};base64,{b64}"

# ----- Image processing methods -----

def histogram_equalization_color(pil_img: Image.Image):
    img = np.array(pil_img)
    # convert to YCrCb or HSV to equalize luminance
    img_yuv = cv2.cvtColor(img, cv2.COLOR_RGB2YUV)
    img_yuv[:,:,0] = cv2.equalizeHist(img_yuv[:,:,0])
    eq = cv2.cvtColor(img_yuv, cv2.COLOR_YUV2RGB)
    return Image.fromarray(eq)

def brightness_contrast(pil_img: Image.Image, brightness: float = 1.2, contrast: float = 1.2):
    # brightness: factor (1.0 no change), contrast: factor
    img = pil_img
    enhancer_b = ImageEnhance.Brightness(img)
    img_b = enhancer_b.enhance(brightness)
    enhancer_c = ImageEnhance.Contrast(img_b)
    img_bc = enhancer_c.enhance(contrast)
    return img_bc

def mean_filter(pil_img: Image.Image, ksize=5):
    cv = pil_to_cv2(pil_img)
    blurred = cv2.blur(cv, (ksize, ksize))
    return cv2_to_pil(blurred)

def median_filter(pil_img: Image.Image, ksize=5):
    cv = pil_to_cv2(pil_img)
    if cv.ndim == 3:
        median = cv2.medianBlur(cv, ksize)
    else:
        median = cv2.medianBlur(cv, ksize)
    return cv2_to_pil(median)

def laplacian_sharpen(pil_img: Image.Image):
    cv = pil_to_cv2(pil_img)
    gray = cv2.cvtColor(cv, cv2.COLOR_BGR2GRAY)
    lap = cv2.Laplacian(gray, cv2.CV_64F)
    lap = cv2.normalize(lap, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    # combine as overlay
    lap_bgr = cv2.cvtColor(lap, cv2.COLOR_GRAY2BGR)
    sharpened = cv2.addWeighted(cv, 1.0, lap_bgr, 0.5, 0)
    return cv2_to_pil(sharpened)

def low_pass_filter(pil_img: Image.Image, ksize=9):
    cv = pil_to_cv2(pil_img)
    low = cv2.GaussianBlur(cv, (ksize, ksize), 0)
    return cv2_to_pil(low)

def high_pass_filter(pil_img: Image.Image, ksize=9):
    cv = pil_to_cv2(pil_img)
    low = cv2.GaussianBlur(cv, (ksize, ksize), 0)
    high = cv2.subtract(cv, low)
    # shift to displayable range
    high = cv2.normalize(high, None, 0, 255, cv2.NORM_MINMAX)
    return cv2_to_pil(high.astype(np.uint8))

def sobel_edges(pil_img: Image.Image):
    cv = pil_to_cv2(pil_img)
    gray = cv2.cvtColor(cv, cv2.COLOR_BGR2GRAY)
    sx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
    sy = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
    sob = np.hypot(sx, sy)
    sob = np.uint8(255 * sob / np.max(sob + 1e-8))
    return cv2_to_pil(sob)

def canny_edges(pil_img: Image.Image):
    cv = pil_to_cv2(pil_img)
    gray = cv2.cvtColor(cv, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 100, 200)
    return cv2_to_pil(edges)

def otsu_segmentation(pil_img: Image.Image):
    cv = pil_to_cv2(pil_img)
    gray = cv2.cvtColor(cv, cv2.COLOR_BGR2GRAY)
    thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
    colored = cv2.cvtColor(thresh, cv2.COLOR_GRAY2BGR)
    return cv2_to_pil(colored)

def color_kmeans_segmentation(pil_img: Image.Image, K=3):
    cv = pil_to_cv2(pil_img)
    Z = cv.reshape((-1,3))
    Z = np.float32(Z)
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
    ret,label,center=cv2.kmeans(Z, K, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)
    center = np.uint8(center)
    res = center[label.flatten()]
    segmented = res.reshape((cv.shape))
    return cv2_to_pil(segmented)

def morphological_ops(pil_img: Image.Image):
    cv = pil_to_cv2(pil_img)
    gray = cv2.cvtColor(cv, cv2.COLOR_BGR2GRAY)
    kernel = np.ones((5,5),np.uint8)
    dilation = cv2.dilate(gray, kernel, iterations = 1)
    erosion = cv2.erode(gray, kernel, iterations = 1)
    opening = cv2.morphologyEx(gray, cv2.MORPH_OPEN, kernel)
    closing = cv2.morphologyEx(gray, cv2.MORPH_CLOSE, kernel)
    # stack results horizontally for display
    imgs = [gray, dilation, erosion, opening, closing]
    imgs_rgb = [cv2.cvtColor(i, cv2.COLOR_GRAY2BGR) for i in imgs]
    combined = np.hstack([cv2.resize(i, (cv.shape[1]//5, cv.shape[0]//5)) for i in imgs_rgb])
    # to keep scale readable, instead create a 2x3 montage
    h, w = cv.shape[:2]
    thumb_h, thumb_w = h//3, w//3
    thumbs = [cv2.resize(i, (thumb_w, thumb_h)) for i in imgs_rgb]
    # build a 2x3 canvas
    canvas = np.zeros((thumb_h*2, thumb_w*3, 3), dtype=np.uint8)
    # place thumbs
    coords = [(0,0),(0,thumb_w),(0,2*thumb_w),(thumb_h,0),(thumb_h,thumb_w)]
    for t, (r,c) in zip(thumbs, coords):
        canvas[r:r+thumb_h, c:c+thumb_w] = t
    return cv2_to_pil(canvas)

def affine_transform(pil_img: Image.Image):
    cv = pil_to_cv2(pil_img)
    rows,cols = cv.shape[:2]
    # translation
    M_trans = np.float32([[1,0,cols*0.05],[0,1,rows*0.05]])
    trans = cv2.warpAffine(cv, M_trans, (cols, rows))
    # rotation
    M_rot = cv2.getRotationMatrix2D((cols/2,rows/2), 15, 1)
    rot = cv2.warpAffine(cv, M_rot, (cols, rows))
    # scaling
    scaled = cv2.resize(cv, None, fx=1.2, fy=1.2, interpolation=cv2.INTER_LINEAR)
    # combine as montage
    # resize scaled to original size center crop or pad
    scaled_resized = cv2.resize(scaled, (cols, rows))
    combined = np.hstack([trans, rot, scaled_resized])
    return cv2_to_pil(combined)

def interpolation_compare(pil_img: Image.Image):
    cv = pil_to_cv2(pil_img)
    rows,cols = cv.shape[:2]
    # downscale then upscale using different interpolations
    small = cv2.resize(cv, (cols//4, rows//4), interpolation=cv2.INTER_AREA)
    nn = cv2.resize(small, (cols, rows), interpolation=cv2.INTER_NEAREST)
    bilinear = cv2.resize(small, (cols, rows), interpolation=cv2.INTER_LINEAR)
    bicubic = cv2.resize(small, (cols, rows), interpolation=cv2.INTER_CUBIC)
    combined = np.hstack([nn, bilinear, bicubic])
    return cv2_to_pil(combined)

# list of pipelines: name and function
PIPELINES = [
    ("original", lambda x: x.copy()),
    ("histogram_equalization", histogram_equalization_color),
    ("brightness_contrast", lambda x: brightness_contrast(x, brightness=1.2, contrast=1.2)),
    ("mean_filter", mean_filter),
    ("median_filter", median_filter),
    ("laplacian_sharpen", laplacian_sharpen),
    ("low_pass_filter", low_pass_filter),
    ("high_pass_filter", high_pass_filter),
    ("sobel_edges", sobel_edges),
    ("canny_edges", canny_edges),
    ("otsu_segmentation", otsu_segmentation),
    ("color_kmeans_segmentation", color_kmeans_segmentation),
    ("morphological_ops", morphological_ops),
    ("affine_transform", affine_transform),
    ("interpolation_compare", interpolation_compare),
]

# ========== Helper: safe autocast context depending on device ==========
def amp_scope():
    if device == "cuda":
        return torch.cuda.amp.autocast()
    else:
        return nullcontext()

# ========== API endpoints ==========
@app.post("/classify")
async def classify_image(file: UploadFile = File(...), top_k: int = 5):
    contents = await file.read()
    try:
        original = Image.open(io.BytesIO(contents)).convert("RGB")
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Could not read image: {e}")

    results = []

    for name, func in PIPELINES:
        try:
            processed = func(original)
        except Exception as e:
            # If a pipeline fails, continue but report the error for that entry
            results.append({
                "method": name,
                "error": f"processing failed: {e}"
            })
            continue

        # Preprocess -> model input
        try:
            image_input = preprocess(processed).unsqueeze(0).to(device)
        except Exception as e:
            results.append({
                "method": name,
                "error": f"preprocess failed: {e}"
            })
            continue

        with torch.no_grad(), amp_scope():
            image_features = model.encode_image(image_input)
            image_features = image_features / image_features.norm(dim=-1, keepdim=True)
            text_feat_dev = text_features_cpu.to(device)
            logits = (100.0 * image_features @ text_feat_dev.T).softmax(dim=-1)
            scores = logits.cpu().numpy()[0]

        top_idx = int(scores.argmax())
        top_category = ECOMMERCE_ITEMS[top_idx]
        confidence = float(scores[top_idx])

        # MongoDB query
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

        # Encode processed image to base64 data URL
        try:
            data_url = pil_to_data_url(processed, fmt="JPEG", quality=80)
        except Exception:
            # fallback to original thumbnail
            data_url = pil_to_data_url(original, fmt="JPEG", quality=70)

        results.append({
            "method": name,
            "image_data": data_url,
            "top_category": top_category,
            "confidence": confidence,
            "matches": matches
        })

    return {"results": results}

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
