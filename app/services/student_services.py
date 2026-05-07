from typing import List
from app.db.mongo import student_collection
from app.schemas.student_schema import StudentSchema
from app.services.embedding import get_student_embedding


class StudentService:

    @staticmethod
    async def upsert_students(students: List[StudentSchema]):
        results = []

        for student in students:
            data = student.dict()

            # 🔥 normalize nulls
            data["skills"] = data.get("skills") or []
            data["enrolledCourses"] = data.get("enrolledCourses") or []

            # 🔥 map studentId → _id
            student_id = data["studentId"]
            data["_id"] = student_id

            # 🔥 generate embedding safely
            try:
                embedding = get_student_embedding(data)
                data["embedding"] = embedding
            except Exception as e:
                print(f"Embedding failed for {student_id}: {e}")
                data["embedding"] = []

            try:
                result = await student_collection.update_one(
                    {"_id": student_id},
                    {"$set": data},
                    upsert=True
                )

                if result.matched_count:
                    results.append(f"Updated {student_id}")
                else:
                    results.append(f"Inserted {student_id}")

            except Exception as e:
                print(f"DB error for {student_id}: {e}")
                results.append(f"Failed {student_id}")

        return results