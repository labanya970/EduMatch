from fastapi import APIRouter

router = APIRouter()

@router.get("/recommend/{student_id}")
async def recommend(student_id: str):
    return await router.recommender.recommend(student_id)