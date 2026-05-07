from pydantic import BaseModel, Field
from typing import List, Optional


class jobSchema(BaseModel):
    totalRecommendations: Optional[int] = 10
    position: Optional[str] = ""
    description: Optional[str] = ""
    skills: Optional[List[str]] = Field(default_factory=list)
    experience: Optional[str] = ""

    class Config:
        extra = "ignore"