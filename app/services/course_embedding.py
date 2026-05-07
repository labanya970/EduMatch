from sentence_transformers import SentenceTransformer
from app.config import MODEL_NAME

model = SentenceTransformer(MODEL_NAME)


# =========================================================
# BUILD COURSE TEXT (IMPROVED)
# =========================================================
def build_course_text(course):
    if isinstance(course, dict):
        name = course.get("name", "")
        description = course.get("description", "")
        level = course.get("level", "")
    else:
        name = course.name
        description = course.description or ""
        level = course.level

    name = str(name).lower()
    description = str(description).lower()

    # -----------------------------
    # CONVERT LEVEL TO SEMANTIC TEXT
    # -----------------------------
    level_map = {
        1: "beginner",
        2: "intermediate",
        3: "advanced"
    }
    level_text = level_map.get(level, str(level).lower())

    # -----------------------------
    # STRUCTURED + NATURAL TEXT
    # -----------------------------
    return f"""
    this course teaches {name}.
    topics covered include {description}.
    skills you will gain from this course include {description}.
    this is a {level_text} level course.
    """


# =========================================================
# COURSE EMBEDDING
# =========================================================
def get_course_embedding(course):
    text = build_course_text(course)

    if not text.strip():
        text = "general technology course"

    embedding = model.encode(
        text,
        normalize_embeddings=True
    )

    if embedding is None or len(embedding) == 0:
        return [0.0] * 384

    return embedding.tolist()