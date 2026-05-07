from fastapi import APIRouter, HTTPException
from typing import List
from app.schemas.student_schema import StudentSchema
from app.services.student_services import StudentService

router = APIRouter()

@router.post("/postStudents")
async def bulk_students(students: List[StudentSchema]):
    try:
        result = await StudentService.upsert_students(students)

        return {
            "status": "success",
            "data": result
        }

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=str(e)
        )