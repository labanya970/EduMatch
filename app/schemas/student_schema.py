from pydantic import BaseModel, Field, validator
from typing import List, Optional


class EnrolledCourse(BaseModel):
    courseId: str
    courseName: Optional[str] = None
    description: Optional[str] = None
    percentageMarks: float = 0
    isComplete: bool = False
    rating: float = 0
    complete: Optional[bool] = None
    @validator("percentageMarks", "rating", pre=True)
    def convert_to_float(cls, v):
        try:
            return float(v) if v is not None else 0.0
        except:
            return 0.0


class StudentSchema(BaseModel):
    studentId: str
    name: Optional[str] = ""
    description: Optional[str] = ""
    skills: Optional[List[str]] = Field(default_factory=list)
    enrolledCourses: Optional[List[EnrolledCourse]] = Field(default_factory=list)

    embedding: Optional[List[float]] = None

    # NEW FIELD
    notInterestedCourses: Optional[List[str]] = None

    class Config:
        extra = "ignore"