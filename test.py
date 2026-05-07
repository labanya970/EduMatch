import asyncio

from app.services.job_recommender import JobStudentRecommender


# -----------------------------
# TEST JOBS
# -----------------------------
jobs = [

    # Backend Internship
    {
        "totalRecommendations": 5,
        "position": "Software Engineer Intern",
        "description": (
            "Work on building scalable backend services using Spring Boot "
            "and contribute to frontend features in React. "
            "Collaborate with senior engineers on production systems."
        ),
        "skills": [
            "Java",
            "Spring Boot",
            "React",
            "REST APIs",
            "Git"
        ],
        "experience": "Fresher / 0-1 years"
    },

    # Machine Learning Internship
    {
        "totalRecommendations": 5,
        "position": "Machine Learning Intern",
        "description": (
            "Build machine learning models, analyze datasets, "
            "and work on AI systems using Python and TensorFlow."
        ),
        "skills": [
            "Python",
            "Machine Learning",
            "Deep Learning",
            "TensorFlow",
            "Data Analysis"
        ],
        "experience": "0-1 years"
    },

    # Frontend Internship
    {
        "totalRecommendations": 5,
        "position": "Frontend Developer Intern",
        "description": (
            "Develop responsive web applications using React and JavaScript. "
            "Improve UI/UX and frontend performance."
        ),
        "skills": [
            "React",
            "JavaScript",
            "HTML",
            "CSS",
            "UI Design"
        ],
        "experience": "Fresher"
    },

    # DevOps Internship
    {
        "totalRecommendations": 5,
        "position": "DevOps Engineer Intern",
        "description": (
            "Assist in cloud deployment automation, CI/CD pipelines, "
            "containerization, and infrastructure management."
        ),
        "skills": [
            "Docker",
            "Kubernetes",
            "AWS",
            "CI/CD",
            "Linux"
        ],
        "experience": "0-1 years"
    },

    # Cybersecurity Internship
    {
        "totalRecommendations": 5,
        "position": "Cyber Security Intern",
        "description": (
            "Work on network security, penetration testing, "
            "ethical hacking, and cyber threat analysis."
        ),
        "skills": [
            "Cyber Security",
            "Networking",
            "Ethical Hacking",
            "Cryptography"
        ],
        "experience": "Fresher"
    }
]


# -----------------------------
# TEST FUNCTION
# -----------------------------
async def test_job(job, recommender):

    print("\n" + "=" * 70)
    print(f"JOB ROLE: {job['position']}")
    print("=" * 70)

    print(f"\nDescription:\n{job['description']}\n")

    print("Required Skills:")
    for skill in job["skills"]:
        print(f"- {skill}")

    print(f"\nExperience: {job['experience']}")

    results = await recommender.recommend_students(job)

    print("\nTop Recommended Students:\n")

    if not results:
        print("No matching students found")
        return

    for i, student in enumerate(results, 1):

        print(f"{i}. {student['name']}")
        print(f"   Student ID : {student['student_id']}")
        print(f"   Score      : {student['score']:.4f}")
        print(f"   Skills     : {student['skills']}")
        print()


# -----------------------------
# MAIN
# -----------------------------
async def main():

    recommender = JobStudentRecommender()

    print("\nInitializing Job Recommender...\n")

    for job in jobs:
        await test_job(job, recommender)


# -----------------------------
# ENTRY POINT
# -----------------------------
if __name__ == "__main__":
    asyncio.run(main())