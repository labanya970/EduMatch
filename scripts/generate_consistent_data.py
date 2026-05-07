import asyncio
import numpy as np
from bson import ObjectId

from app.db.mongo import student_collection, course_collection

from app.services.embedding import get_student_embedding
from app.services.course_embedding import get_course_embedding


# -----------------------------
# REGENERATE COURSE EMBEDDINGS
# -----------------------------
async def update_courses():
    print("Updating course embeddings...")

    count = 0

    async for course in course_collection.find():
        emb = get_course_embedding(course)

        await course_collection.update_one(
            {"_id": course["_id"]},
            {"$set": {"embedding": emb}}
        )

        count += 1
        if count % 50 == 0:
            print(f"{count} courses updated...")

    print(f"Done. Total courses updated: {count}")


# -----------------------------
# BUILD NOT INTERESTED EMBEDDING (FROM COURSE IDS)
# -----------------------------
async def build_not_interested_embedding(not_interested_ids):

    embeddings = []

    for cid in not_interested_ids:
        try:
            course = await course_collection.find_one({"_id": ObjectId(cid)})
        except:
            course = await course_collection.find_one({"_id": cid})

        if course and course.get("embedding"):
            embeddings.append(course["embedding"])

    if embeddings:
        return np.mean(np.array(embeddings), axis=0).tolist()

    return []


# -----------------------------
# REGENERATE STUDENT EMBEDDINGS
# -----------------------------
async def update_students():
    print("Updating student embeddings...")

    count = 0

    async for student in student_collection.find():

        # -----------------------------
        # MAIN STUDENT EMBEDDING
        # -----------------------------
        student_emb = get_student_embedding(student)

        # -----------------------------
        # NOT INTERESTED EMBEDDING (FIXED)
        # -----------------------------
        not_interested_ids = student.get("notInterestedCourses") or []

        not_interested_emb = await build_not_interested_embedding(
            not_interested_ids
        )

        await student_collection.update_one(
            {"_id": student["_id"]},
            {
                "$set": {
                    "embedding": student_emb,
                    "not_interested_embedding": not_interested_emb
                }
            }
        )

        count += 1
        if count % 50 == 0:
            print(f"{count} students updated...")

    print(f"Done. Total students updated: {count}")


# -----------------------------
# MAIN
# -----------------------------
async def main():
    await update_courses()
    await update_students()


if __name__ == "__main__":
    asyncio.run(main())