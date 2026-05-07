from typing import List
from app.db.mongo import course_collection
from app.schemas.course_schema import CourseSchema
from app.services.course_embedding import get_course_embedding


class CourseService:

    @staticmethod
    async def upsert_courses(courses: List[CourseSchema]):
        results = []

        for course in courses:
            course_dict = course.dict()

            course_id = course_dict["id"]

            # 🔥 generate embedding
            embedding = get_course_embedding(course)
            course_dict["embedding"] = embedding

            # map to Mongo _id
            course_dict["_id"] = course_id

            result = await course_collection.update_one(
                {"_id": course_id},
                {"$set": course_dict},
                upsert=True
            )

            if result.matched_count:
                results.append(f"Updated {course_id}")
            else:
                results.append(f"Inserted {course_id}")

        return results