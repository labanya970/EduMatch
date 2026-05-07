from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

from app.db.mongo import student_collection
from app.config import MODEL_NAME


class JobStudentRecommender:

    def __init__(self):
        self.embedder = SentenceTransformer(MODEL_NAME)

    # -----------------------------
    # BUILD JOB TEXT
    # -----------------------------
    def build_job_text(self, job):

        position = job.get("position", "")
        description = job.get("description", "")
        skills = " ".join(job.get("skills", []))
        experience = job.get("experience", "")

        return f"""
        job role {position}.
        required skills {skills}.
        responsibilities {description}.
        experience required {experience}.
        """

    # -----------------------------
    # RECOMMEND STUDENTS
    # -----------------------------
    async def recommend_students(self, job):

        top_n = int(job.get("totalRecommendations", 10))

        # -----------------------------
        # JOB EMBEDDING
        # -----------------------------
        job_text = self.build_job_text(job)

        job_embedding = self.embedder.encode(
            job_text,
            normalize_embeddings=True
        ).reshape(1, -1)

        # -----------------------------
        # LOAD STUDENTS
        # -----------------------------
        students = await student_collection.find().to_list(length=None)

        results = []

        required_skills = set(
            s.lower() for s in job.get("skills", [])
        )

        # -----------------------------
        # SCORE STUDENTS
        # -----------------------------
        for student in students:

            student_embedding = student.get("embedding")

            if not student_embedding:
                continue

            student_embedding = np.array(
                student_embedding
            ).reshape(1, -1)

            # -----------------------------
            # SEMANTIC SIMILARITY
            # -----------------------------
            sim = cosine_similarity(
                job_embedding,
                student_embedding
            )[0][0]

            # -----------------------------
            # SKILL OVERLAP
            # -----------------------------
            student_skills = set(
                s.lower() for s in student.get("skills", [])
            )

            overlap = len(
                required_skills.intersection(student_skills)
            )

            if required_skills:
                overlap_score = overlap / len(required_skills)
            else:
                overlap_score = 0

            # -----------------------------
            # EXPERIENCE SCORE
            # -----------------------------
            experience_score = 0

            for c in student.get("enrolledCourses", []):

                rating = float(c.get("rating", 0))
                progress = float(c.get("percentageMarks", 0))
                complete = c.get("isComplete", False)

                score = (
                    0.5 * (rating / 5) +
                    0.3 * (progress / 100) +
                    (0.2 if complete else 0)
                )

                experience_score += score

            # normalize
            experience_score = min(experience_score / 5, 1)

            # -----------------------------
            # FINAL SCORE
            # -----------------------------
            final_score = (
                0.6 * sim +
                0.25 * overlap_score +
                0.15 * experience_score
            )

            results.append({
                "student_id": str(student["_id"]),
                "name": student.get("name"),
                "score": float(final_score),
                "skills": student.get("skills", []),
                "semantic_similarity": float(sim),
                "skill_overlap_score": float(overlap_score),
                "experience_score": float(experience_score)
            })

        # -----------------------------
        # SORT RESULTS
        # -----------------------------
        results.sort(
            key=lambda x: x["score"],
            reverse=True
        )

        return results[:top_n]