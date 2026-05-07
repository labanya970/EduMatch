from beanie import Document
from typing import List

class Student(Document):
    student_id: str
    name: str
    embedding: List[float]

    class Settings:
        name = "students"