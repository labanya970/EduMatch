from fastapi import HTTPException

from fastapi import APIRouter
from typing import List
from app.services.course_service import CourseService
from app.schemas.course_schema import CourseSchema


router = APIRouter()

@router.post("/postCourses")
async def bulk_courses(courses: List[CourseSchema]):
    try:
        result = await CourseService.upsert_courses(courses)

        return {
            "status": "success",
            "data": result
        }

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=str(e)
        )