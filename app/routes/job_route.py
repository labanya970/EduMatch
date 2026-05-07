from fastapi import APIRouter, HTTPException

from app.schemas.job_schema import jobSchema
from app.services.job_recommender import JobStudentRecommender

router = APIRouter()

job_recommender = JobStudentRecommender()


@router.post("/recommendJobsTostudents")
async def recommend_jobs_to_students(
        job: jobSchema
):
    try:

        recommended_students = await job_recommender.recommend_students(
            job=job,
            top_n=int(job.totalRecommendations)
        )

        return recommended_students

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=str(e)
        )