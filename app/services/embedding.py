from sentence_transformers import SentenceTransformer
from app.config import MODEL_NAME
import numpy as np

model = SentenceTransformer(MODEL_NAME)


# =========================================================
# STUDENT EMBEDDING
# =========================================================
def build_student_text(student):
    if isinstance(student, dict):
        skills_list = student.get("skills") or []
        enrolled = student.get("enrolledCourses") or []
        description = student.get("description") or ""
    else:
        skills_list = student.skills or []
        enrolled = student.enrolledCourses or []
        description = student.description or ""

    # skills
    skills = " ".join([str(s).lower() for s in skills_list if s])

    # completed courses
    completed_courses = [
        str(c.get("courseName") or "").lower()
        for c in enrolled
        if isinstance(c, dict) and c.get("isComplete")
    ]
    completed_text = " ".join([c for c in completed_courses if c])

    # fallback
    if not skills and not description:
        return "student interested in learning technology skills"

    return f"""
    student skills {skills}.
    interested in {skills}.
    learning goals {description.lower()}.
    completed topics {completed_text}.
    """


def get_student_embedding(student):
    text = build_student_text(student)

    if not text.strip():
        text = "student profile"

    emb = model.encode(text, normalize_embeddings=True)

    if emb is None or len(emb) == 0:
        return [0.0] * 384

    return emb.tolist()


# =========================================================
# NOT INTERESTED EMBEDDING (UPDATED FOR COURSE IDS)
# =========================================================
def get_not_interested_embedding_from_courses(course_embeddings):
    """
    course_embeddings: list of embeddings (already fetched from DB)
    """

    if not course_embeddings:
        return []

    arr = np.array(course_embeddings)

    # mean vector (fast + stable)
    mean_emb = np.mean(arr, axis=0)

    return mean_emb.tolist()


# =========================================================
# OPTIONAL: ENROLLED AGGREGATION HELPER
# =========================================================
def aggregate_enrolled_embeddings(course_embeddings, indices, method="max"):
    """
    course_embeddings: np.array of all course vectors
    indices: list of indices for enrolled courses
    method: "mean" or "max"
    """

    if not indices:
        return None

    selected = course_embeddings[indices]

    if method == "mean":
        return np.mean(selected, axis=0)

    if method == "max":
        return selected  # used for max similarity

    return None