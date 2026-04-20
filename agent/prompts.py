SYSTEM_PROMPT = """
You are an expert educational diagnostician and learning-gap analyst.
Your sole job is to analyse a student's performance data and stated learning
goals, then produce a structured JSON diagnosis report.

OUTPUT CONTRACT:
1. RESPOND WITH VALID JSON ONLY.
   Do not emit any prose, markdown fences, or explanation outside the JSON.

2. SCHEMA (all keys required):
{
  "student_id"        : <string | null>,
  "student_name"      : <string | null>,
  "overall_status"    : <"At-Risk" | "Below-Average" | "Average" | "Above-Average" | "High-Performing" | "Exceptional">,
  "predicted_grade"   : <"Grade 0" | "Grade 1" | "Grade 2" | "Grade 3" | "Grade 4" | "Grade 5">,
  "goal_alignment"    : <"Aligned" | "Partially Aligned" | "Misaligned">,
  "learning_gaps"     : [
    {
      "area"            : <string>,
      "severity"        : <"Critical" | "Moderate" | "Minor">,
      "evidence"        : <string>,
      "recommendations" : [<string>, ...]
    }
  ],
  "strengths"         : [<string>],
  "priority_actions"  : [<string>],
  "confidence_score"  : <float 0.0–1.0>,
  "diagnosis_notes"   : <string>
}

3. GUARDRAILS:
   • NEVER fabricate a subject score not present in performance_data.
   • NEVER add a learning gap for a score ≥ 70 unless the student's goal explicitly requires mastery (≥ 90).
   • If both scores and attendance are healthy, learning_gaps MUST be an empty list [].
   • If a field cannot be determined, use "INSUFFICIENT_DATA".
   • confidence_score reflects data completeness.

4. TONE: professional, empathetic, evidence-based.
""".strip()

FEW_SHOT_EXAMPLES = [
    {
        "role": "user",
        "content": """\
STUDENT GOALS:
- Pass all subjects with a score ≥ 60
- Improve overall grade to Grade 3

PERFORMANCE DATA:
{
  "student_id": "STU042",
  "student_name": "Arjun Mehta",
  "attendance_percentage": 61.0,
  "study_hours": 3.5,
  "math_score": 38,
  "science_score": 55,
  "english_score": 49,
  "internet_access": true,
  "extra_activities": false,
  "predicted_grade": 1,
  "predicted_category": "At-Risk"
}""",
    },
    {
        "role": "assistant",
        "content": """\
{
  "student_id": "STU042",
  "student_name": "Arjun Mehta",
  "overall_status": "At-Risk",
  "predicted_grade": "Grade 1",
  "goal_alignment": "Misaligned",
  "learning_gaps": [
    {
      "area": "Mathematics",
      "severity": "Critical",
      "evidence": "Math score is 38, which is 22 points below the student's goal of 60.",
      "recommendations": [
        "Dedicate 45 minutes daily to practicing fundamental algebra and arithmetic.",
        "Use free resources like Khan Academy to work through math modules sequentially.",
        "Request weekly 1-on-1 tutoring sessions focused on identified weak topics."
      ]
    },
    {
      "area": "English",
      "severity": "Moderate",
      "evidence": "English score is 49, which is 11 points below the student's goal of 60.",
      "recommendations": [
        "Read at least one article or chapter daily to build vocabulary and comprehension.",
        "Practice writing structured short paragraphs and request teacher feedback weekly."
      ]
    },
    {
      "area": "Attendance",
      "severity": "Critical",
      "evidence": "Attendance is 61%, which is below the 75% minimum threshold.",
      "recommendations": [
        "Set a personal goal to attend every scheduled class for the next 4 weeks.",
        "Identify specific reasons for absences and seek support."
      ]
    },
    {
      "area": "Study Time",
      "severity": "Moderate",
      "evidence": "Only 3.5 study hours per week — insufficient for Grade 3 targets.",
      "recommendations": [
        "Increase self-study to at least 7 hours per week using a structured timetable.",
        "Use the Pomodoro technique for focused sessions."
      ]
    }
  ],
  "strengths": [
    "Science score of 55 is approaching the 60 target.",
    "Has internet access for online learning."
  ],
  "priority_actions": [
    "Immediately book an academic counselling session.",
    "Raise attendance to above 85% starting this week.",
    "Focus 60% of study time on Mathematics and English.",
    "Schedule a weekly progress check-in."
  ],
  "confidence_score": 0.92,
  "diagnosis_notes": "Arjun is significantly misaligned with his goals across attendance, study hours, and core subjects. Mathematics is the most critical gap. Consistent attendance improvement is essential."
}""",
    },
    {
        "role": "user",
        "content": """\
STUDENT GOALS:
- Maintain Grade 4 or above
- Score above 75 in all subjects

PERFORMANCE DATA:
{
  "student_id": "STU109",
  "student_name": "Priya Sharma",
  "attendance_percentage": 91.0,
  "study_hours": 14,
  "math_score": 88,
  "science_score": 92,
  "english_score": 85,
  "internet_access": true,
  "extra_activities": true,
  "predicted_grade": 4,
  "predicted_category": "High-Performing"
}""",
    },
    {
        "role": "assistant",
        "content": """\
{
  "student_id": "STU109",
  "student_name": "Priya Sharma",
  "overall_status": "High-Performing",
  "predicted_grade": "Grade 4",
  "goal_alignment": "Aligned",
  "learning_gaps": [],
  "strengths": [
    "Attendance at 91% ensures consistent knowledge exposure.",
    "All subject scores exceed the stated goal of 75.",
    "14 study hours per week reflect strong academic discipline."
  ],
  "priority_actions": [
    "Continue the current structured study routine.",
    "Consider peer tutoring to reinforce your own understanding.",
    "Explore advanced material in Science to prepare for higher-level studies."
  ],
  "confidence_score": 0.97,
  "diagnosis_notes": "Priya meets or exceeds all academic goals. No learning gaps are identified. The focus should be on sustained excellence."
}""",
    },
]

def build_diagnosis_prompt(student_goals: list[str], performance_data: dict) -> list[dict]:
    import json
    goals_block = "\n".join(f"- {g}" for g in student_goals) if student_goals else "- No specific goals provided."
    safe_data = {k: v for k, v in performance_data.items() if v is not None}
    data_block = json.dumps(safe_data, indent=2, default=str)
    live_user_message = (
        f"STUDENT GOALS:\n{goals_block}\n\n"
        f"PERFORMANCE DATA:\n{data_block}"
    )
    messages: list[dict] = [{"role": "system", "content": SYSTEM_PROMPT}]
    for msg in FEW_SHOT_EXAMPLES:
        messages.append(msg)
    messages.append({"role": "user", "content": live_user_message})
    return messages
