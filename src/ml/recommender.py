import pandas as pd

def generate_recommendations(student_data: dict, predicted_category: str) -> list:
    """
    Generates study recommendations based on student features and predicted category
    (e.g., At-Risk, Average, High-Performing).
    
    Expected student_data keys (examples):
    - attendance_percentage: numeric
    - study_hours: numeric
    - math_score, science_score, english_score: numeric
    - internet_access: int/bool
    """
    recommendations = []
    
    if "Grade 0" in predicted_category or "Grade 1" in predicted_category or "At-Risk" in predicted_category:
        recommendations.append("Priority: Schedule a 1-on-1 session with an academic counselor or mentor.")
        recommendations.append("Action: Break down study materials into smaller, manageable 30-minute sessions.")
    elif "Grade 2" in predicted_category or "Grade 3" in predicted_category or "Average" in predicted_category:
        recommendations.append("Priority: Maintain current momentum while identifying specific weak zones.")
        recommendations.append("Action: Try peer study groups to clarify doubts and reinforce concepts.")
    elif "Grade 4" in predicted_category or "Grade 5" in predicted_category or "High" in predicted_category:
        recommendations.append("Priority: Keep up the excellent work!")
        recommendations.append("Action: Consider participating in advanced workshops or tutoring peers.")
        
    if student_data.get("attendance_percentage", 100) < 75:
        recommendations.append("Warning: Low attendance detected. Aim to attend at least 85% of classes to catch up on missed concepts.")
        
    if student_data.get("study_hours", 10) < 5:
        recommendations.append("Suggestion: Increase self-study time. Aim for at least 1-2 hours of focused study daily.")
        
    weak_subjects = []
    if student_data.get("math_score", 100) < 50:
        weak_subjects.append("Math")
    if student_data.get("science_score", 100) < 50:
        weak_subjects.append("Science")
    if student_data.get("english_score", 100) < 50:
        weak_subjects.append("English")
        
    if weak_subjects:
        subs = ", ".join(weak_subjects)
        recommendations.append(f"Focus Area: Your scores in {subs} need improvement. Prioritize these subjects this week.")

    if not student_data.get("internet_access", 1) and ("Grade 0" in predicted_category or "Grade 1" in predicted_category):
        recommendations.append("Resource: Utilize the school library or community centers for internet access to online study materials.")

    if not recommendations:
        recommendations.append("Continue with your current study methodology as it appears to be working well.")

    return recommendations

if __name__ == "__main__":
    sample_student = {
        "attendance_percentage": 65,
        "study_hours": 3,
        "math_score": 45,
        "science_score": 75,
        "english_score": 80,
        "internet_access": 1
    }
    recs = generate_recommendations(sample_student, "Grade 1")
    print("Generated Recommendations:")
    for i, r in enumerate(recs, 1):
        print(f"{i}. {r}")
