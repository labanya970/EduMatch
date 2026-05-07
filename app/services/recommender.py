import numpy as np
import pandas as pd
from datetime import datetime, timezone
from sklearn.metrics.pairwise import cosine_similarity
import lightgbm as lgb
from sentence_transformers import SentenceTransformer

from app.db.mongo import course_collection, student_collection
from app.config import MODEL_NAME


def compute_time_decay(created_at_str: str, half_life_days: float = 180.0) -> float:
    """
    Returns a decay weight in (0, 1] based on how old the course is.
    Newer courses score closer to 1.0; older courses decay toward 0.
    half_life_days=180 means a course created 6 months ago has weight ~0.5.
    """
    if not created_at_str:
        return 0.5  # neutral fallback for missing dates

    try:
        # Handle both ISO format with and without timezone
        created_at_str = created_at_str.replace("Z", "+00:00")
        created_at = datetime.fromisoformat(created_at_str)

        # Make both timezone-aware
        if created_at.tzinfo is None:
            created_at = created_at.replace(tzinfo=timezone.utc)

        now = datetime.now(timezone.utc)
        age_days = max((now - created_at).days, 0)

        # Exponential decay: weight = 2^(-age / half_life)
        return float(2 ** (-age_days / half_life_days))

    except Exception:
        return 0.5  # neutral fallback for unparseable dates


class HybridCourseRecommender:

    def __init__(self):
        self.courses = []
        self.students = []
        self.course_embeddings = None
        self.course_static_features = None
        self.model = None
        self.embedder = SentenceTransformer(MODEL_NAME)
        self.course_id_to_index = {}

    # -----------------------------
    # INITIALIZE
    # -----------------------------
    async def initialize(self):
        await self.load_data()
        self.students = await student_collection.find().to_list(length=None)
        interactions = await self.build_interactions()
        self.train_model(interactions)
        print(
            f"Initialization done. "
            f"Model state: {self.model!r}, "
            f"Courses loaded: {len(self.courses)}, "
            f"Students loaded: {len(self.students)}"
        )

    # -----------------------------
    # LOAD COURSES
    # -----------------------------
    async def load_data(self):
        raw_courses = await course_collection.find().to_list(length=None)

        valid_courses = []
        embeddings = []
        static_features = []

        for c in raw_courses:
            emb = c.get("embedding")
            if emb and len(emb) > 0:
                valid_courses.append(c)
                embeddings.append(emb)

                popularity = np.log1p(c.get("totalEnrolled", 0)) * 0.3
                rating = c.get("rating", 0)
                time_decay = compute_time_decay(c.get("createdAt", ""))

                static_features.append([popularity, rating, time_decay])

        self.courses = valid_courses
        self.course_embeddings = np.array(embeddings)
        self.course_static_features = np.array(static_features)

        self.course_id_to_index = {
            str(c["_id"]): idx
            for idx, c in enumerate(self.courses)
        }

        print(f"Loaded {len(self.courses)} courses with valid embeddings.")

    # -----------------------------
    # BUILD INTERACTIONS
    # -----------------------------
    async def build_interactions(self):
        interactions = []

        for student in self.students:
            user_id = str(student["_id"])

            for c in student.get("enrolledCourses", []) or []:
                course_id = str(c.get("courseId"))

                rating = float(c.get("rating", 0))
                progress = float(c.get("percentageMarks", 0))
                is_complete = c.get("isComplete", False)

                score = (
                    0.6 * rating +
                    0.3 * (progress / 20) +
                    (1.0 if is_complete else 0.0)
                )

                label = 3 if score >= 4 else 2 if score >= 2 else 1 if score > 0 else 0

                # Apply time decay to the label weight using the course's createdAt
                idx = self.course_id_to_index.get(course_id)
                if idx is not None:
                    decay = self.course_static_features[idx][2]  # time_decay column
                    # Scale label upward for newer courses, downward for older
                    # Clamp to valid label range [0, 3]
                    decayed_label = min(3, round(label * decay + label * (1 - decay) * 0.5))
                else:
                    decayed_label = label

                interactions.append({
                    "user_id": user_id,
                    "course_id": course_id,
                    "label": decayed_label
                })

        return interactions

    # -----------------------------
    # ENSURE STUDENT EMBEDDINGS
    # -----------------------------
    def _ensure_student_embeddings(self, student):

        if "skill_embedding" not in student:
            student["skill_embedding"] = self.embedder.encode(
                " ".join(student.get("skills", [])),
                normalize_embeddings=True
            ).reshape(1, -1)

        if "desc_embedding" not in student:
            student["desc_embedding"] = self.embedder.encode(
                student.get("description", ""),
                normalize_embeddings=True
            ).reshape(1, -1)

        if not student.get("embedding"):
            text = student.get("description", "") + " " + " ".join(student.get("skills", []))
            student["embedding"] = self.embedder.encode(
                text,
                normalize_embeddings=True
            ).tolist()

    # -----------------------------
    # FEATURE ENGINEERING
    # -----------------------------
    def build_features(self, student, idx):

        course = self.courses[idx]
        course_emb = self.course_embeddings[idx].reshape(1, -1)

        skill_sim = cosine_similarity(student["skill_embedding"], course_emb)[0][0]
        desc_sim = cosine_similarity(student["desc_embedding"], course_emb)[0][0]

        skill_sim = (skill_sim + 1) / 2
        desc_sim = (desc_sim + 1) / 2

        content = (0.85 * skill_sim + 0.15 * desc_sim) ** 2

        student_skills = set(s.lower() for s in student.get("skills", []))
        course_tags = set(t.lower() for t in course.get("tags", []))
        tag_sim = 1.0 if student_skills & course_tags else 0.0

        popularity, rating, time_decay = self.course_static_features[idx]

        return [content, skill_sim, desc_sim, tag_sim, popularity, rating, time_decay]

    # -----------------------------
    # TRAIN MODEL
    # -----------------------------
    def train_model(self, interactions):

        if len(interactions) < 50:
            print(
                f"Insufficient interactions ({len(interactions)}) for LightGBM training. "
                "Falling back to pure similarity matching."
            )
            self.model = "fallback_mode"
            return

        X, y, group = [], [], []
        df = pd.DataFrame(interactions)

        for user_id, group_df in df.groupby("user_id"):

            student = next(
                (s for s in self.students if str(s["_id"]) == user_id),
                None
            )
            if not student:
                continue

            self._ensure_student_embeddings(student)

            count = 0

            for _, row in group_df.iterrows():

                idx = self.course_id_to_index.get(str(row["course_id"]))
                if idx is None:
                    continue

                X.append(self.build_features(student, idx))
                y.append(row["label"])
                count += 1

                for _ in range(3):
                    neg_idx = np.random.randint(len(self.courses))
                    X.append(self.build_features(student, neg_idx))
                    y.append(0)
                    count += 1

            if count > 0:
                group.append(count)

        if not X:
            print("No feature rows built — falling back to similarity matching.")
            self.model = "fallback_mode"
            return

        X = pd.DataFrame(X, columns=[
            "content", "skill_sim", "desc_sim",
            "tag", "popularity", "rating", "time_decay"
        ])

        self.model = lgb.LGBMRanker(
            objective="lambdarank",
            n_estimators=100,
            learning_rate=0.05,
            label_gain=[0, 1, 3, 7]
        )

        self.model.fit(X, y, group=group)
        print("LightGBM ranker trained successfully.")

    # -----------------------------
    # RECOMMEND
    # -----------------------------
    async def recommend(self, student_id: str, top_n=5):

        student = await student_collection.find_one({"studentId": student_id})

        if not student:
            print(f"Student not found for id: {student_id}")
            return []

        if not self.model:
            print("Model is not initialized yet.")
            return []

        self._ensure_student_embeddings(student)

        student_emb = np.array(student["embedding"]).reshape(1, -1)
        similarities = cosine_similarity(student_emb, self.course_embeddings)[0]

        all_sorted_idx = np.argsort(similarities)[::-1]

        # Courses to hard-skip entirely
        enrolled_ids = {
            str(c.get("courseId"))
            for c in (student.get("enrolledCourses") or [])
        }

        not_interested_ids = {
            str(cid)
            for cid in (student.get("notInterestedCourses") or [])
        }

        skip_ids = enrolled_ids | not_interested_ids

        enrolled_indices = [
            self.course_id_to_index[cid]
            for cid in enrolled_ids
            if cid in self.course_id_to_index
        ]

        enrolled_embs = (
            self.course_embeddings[enrolled_indices]
            if enrolled_indices else None
        )

        X = []
        valid_idx = []

        for idx in all_sorted_idx:
            course = self.courses[idx]
            cid = str(course["_id"])

            if cid in skip_ids:
                continue

            X.append(self.build_features(student, idx))
            valid_idx.append(idx)

        if not X:
            return []

        if self.model == "fallback_mode":
            scores = [similarities[idx] for idx in valid_idx]
        else:
            X_df = pd.DataFrame(X, columns=[
                "content", "skill_sim", "desc_sim",
                "tag", "popularity", "rating", "time_decay"
            ])
            scores = self.model.predict(X_df)
            scores = np.maximum(scores, 0)

        student_skills = set(s.lower() for s in student.get("skills", []))

        candidates = []

        for idx, score in zip(valid_idx, scores):

            sim = similarities[idx]
            course = self.courses[idx]
            cid = str(course["_id"])
            cname = course.get("name", "").lower()
            time_decay = self.course_static_features[idx][2]

            if any(skill in cname for skill in student_skills):
                score *= 1.4

            if enrolled_embs is not None:
                course_emb = self.course_embeddings[idx].reshape(1, -1)
                enrolled_sim = cosine_similarity(enrolled_embs, course_emb).max()
                enrolled_sim = (enrolled_sim + 1) / 2
            else:
                enrolled_sim = 0

            # time_decay boosts newer courses in the final ranking
            final_score = (
                0.20 * score +
                0.45 * sim +
                0.20 * enrolled_sim +
                0.15 * time_decay
            )

            candidates.append((cid, course, final_score))

        candidates.sort(key=lambda x: x[2], reverse=True)

        seen = set()
        final = []

        for cid, course, score in candidates:
            if cid not in seen:
                final.append((course, score))
                seen.add(cid)
            if len(final) == top_n:
                break

        return [
            {
                "course_id": str(c["_id"]),
                "name": c["name"],
                "score": float(s)
            }
            for c, s in final
        ]


# --- GLOBAL SINGLETON ---
# Starts as None. main.py assigns the real instance during lifespan startup.
recommender_instance: "HybridCourseRecommender | None" = None