import os
from pymongo import MongoClient
from dotenv import load_dotenv

load_dotenv()

MONGODB_URI = os.environ.get("MONGODB_URI", "mongodb://localhost:27017")
MONGODB_DB = os.environ.get("MONGODB_DB", "visual_search")

print(MONGODB_URI, MONGODB_DB, "@@")

client = MongoClient(MONGODB_URI)
db = client[MONGODB_DB]
items_collection = db["items"]

sample_items = [
    {"name": "MET Rivale Road Helmet", "price": 199.99, "categories": ["bicycle helmet"], "image_url": "https://media.met-helmets.com/app/uploads/2023/09/met-rivale-mips-road-cycling-helmet-BL3.jpg"},
    {"name": "Giro Cycling Helmet", "price": 149.99, "categories": ["bicycle helmet"], "image_url": "https://images.example.com/giro-helmet.jpg"},
    {"name": "Electric Microwave 900W", "price": 89.5, "categories": ["microwave"], "image_url": "https://images.example.com/microwave.jpg"},
    {"name": "Classic Leather Notebook", "price": 12.0, "categories": ["notebook"], "image_url": "https://images.example.com/notebook.jpg"},
    {"name": "Electric Toothbrush Pro", "price": 49.99, "categories": ["toothbrush"], "image_url": "https://images.example.com/toothbrush.jpg"},
    {"name": "Porcelain Plates Set (4)", "price": 29.99, "categories": ["plates"], "image_url": "https://images.example.com/plates.jpg"},
    {"name": "Elegant Perfume - 50ml", "price": 59.0, "categories": ["perfume"], "image_url": "https://images.example.com/perfume.jpg"},
    {"name": "Hair Straightener Pro", "price": 39.99, "categories": ["hair straightener"], "image_url": "https://images.example.com/straightener.jpg"},
    {"name": "Reading Glasses", "price": 15.99, "categories": ["glasses"], "image_url": "https://images.example.com/glasses.jpg"},
    {"name": "Bestselling Novel", "price": 9.99, "categories": ["book"], "image_url": "https://images.example.com/book.jpg"}
]

# Insert or upsert by name (avoid duplicates)
for item in sample_items:
    items_collection.update_one({"name": item["name"]}, {"$set": item}, upsert=True)

print(f"Seeded {len(sample_items)} items into {MONGODB_DB}.items")
