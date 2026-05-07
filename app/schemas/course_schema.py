from pydantic import BaseModel, validator
from typing import Optional, List


class CourseSchema(BaseModel):
    id: str
    name: str
    description: Optional[str] = None
    level: int
    rating: float = 0
    totalEnrolled: int = 0
    createdAt: str
    embedding: Optional[List[float]] = None   # <-- ADD THIS

    @validator("rating", pre=True)
    def convert_rating(cls, v):
        try:
            if v is None:
                return 0.0
            val = float(v)
            if val != val:
                return 0.0
            return val
        except:
            return 0.0