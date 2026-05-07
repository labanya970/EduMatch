from sentence_transformers import SentenceTransformer
from pymongo import MongoClient

model = SentenceTransformer("all-MiniLM-L6-v2")

client = MongoClient("mongodb://localhost:27017/")
db = client["recommenderDB"]

courses = list(db["courses"].find())

for course in courses:
    text = f"""
    Course Name: {course.get('name', '')}
    Category: {course.get('category', '')}
    Tags: {' '.join(course.get('tags', []))}
    Description: {course.get('description', '')}
    """

    embedding = model.encode(text).tolist()

    db["courses"].update_one(
        {"_id": course["_id"]},
        {"$set": {"embedding": embedding}}
    )

print("✅ Course embeddings updated successfully!")