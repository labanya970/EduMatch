from motor.motor_asyncio import AsyncIOMotorClient
from app.config import MONGO_URI, DB_NAME

client = AsyncIOMotorClient(MONGO_URI)
db = client[DB_NAME]

student_collection = db["students"]
course_collection = db["courses"]
interaction_collection = db["interactions"]
user_collection = db["users"]