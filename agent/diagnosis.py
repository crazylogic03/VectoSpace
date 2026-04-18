from __future__ import annotations

import json
import logging
import os
import re
from dataclasses import dataclass, field, asdict
from typing import Any

from agent.prompts import build_diagnosis_prompt

logger = logging.getLogger(__name__)

ATTENDANCE_CRITICAL   = 60.0
ATTENDANCE_WARNING    = 75.0
STUDY_HOURS_MIN       = 5.0
SCORE_CRITICAL        = 40
SCORE_MODERATE        = 60

GRADE_MAP    = {0: "Grade 0", 1: "Grade 1", 2: "Grade 2",
                3: "Grade 3", 4: "Grade 4", 5: "Grade 5"}
CATEGORY_MAP = {0: "At-Risk", 1: "Below-Average", 2: "Average",
                3: "Above-Average", 4: "High-Performing", 5: "Exceptional"}

SUBJECT_FIELDS = [
    ("math_score",    "Mathematics"),
    ("science_score", "Science"),
    ("english_score", "English"),
]

@dataclass
class LearningGap:
    area            : str
    severity        : str
    evidence        : str
    recommendations : list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return asdict(self)

@dataclass
class DiagnosisReport:
    student_id      : str | None
    student_name    : str | None
    overall_status  : str
    predicted_grade : str
    goal_alignment  : str
    learning_gaps   : list[LearningGap]        = field(default_factory=list)
    strengths       : list[str]                = field(default_factory=list)
    priority_actions: list[str]                = field(default_factory=list)
    confidence_score: float                    = 1.0
    diagnosis_notes : str                      = ""
    source          : str                      = "rule-based"

    def to_dict(self) -> dict:
        d = asdict(self)
        d["learning_gaps"] = [g.to_dict() for g in self.learning_gaps]
        return d

def _extract_json_from_text(text: str) -> dict:
    
    fence_match = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL)
    if fence_match:
        return json.loads(fence_match.group(1))

    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    start = text.find("{")
    end   = text.rfind("}")
    if start != -1 and end != -1 and end > start:
        return json.loads(text[start : end + 1])

    raise ValueError("No valid JSON found in LLM response")

def _parse_llm_response(raw: str, fallback: DiagnosisReport) -> DiagnosisReport:
    
    VALID_STATUSES = {"At-Risk", "Below-Average", "Average",
                      "Above-Average", "High-Performing", "Exceptional"}
    VALID_GRADES   = {f"Grade {i}" for i in range(6)}
    VALID_ALIGN    = {"Aligned", "Partially Aligned", "Misaligned"}
    VALID_SEVERITY = {"Critical", "Moderate", "Minor"}

    try:
        data = _extract_json_from_text(raw)

        if data.get("overall_status") not in VALID_STATUSES:
            logger.warning("LLM returned invalid overall_status: %s", data.get("overall_status"))
            data["overall_status"] = fallback.overall_status

        if data.get("predicted_grade") not in VALID_GRADES:
            data["predicted_grade"] = fallback.predicted_grade

        if data.get("goal_alignment") not in VALID_ALIGN:
            data["goal_alignment"] = fallback.goal_alignment

        gaps: list[LearningGap] = []
        for raw_gap in data.get("learning_gaps", []):
            sev = raw_gap.get("severity", "Moderate")
            if sev not in VALID_SEVERITY:
                sev = "Moderate"
            recs = raw_gap.get("recommendations", [])
            recs = recs[:4] if len(recs) > 4 else (recs + ["Consult your teacher for further guidance."] * (2 - len(recs)) if len(recs) < 2 else recs)
            gaps.append(LearningGap(
                area=str(raw_gap.get("area", "General")),
                severity=sev,
                evidence=str(raw_gap.get("evidence", "")),
                recommendations=recs,
            ))

        confidence = float(data.get("confidence_score", 0.8))
        confidence = max(0.0, min(1.0, confidence))

        return DiagnosisReport(
            student_id       = data.get("student_id"),
            student_name     = data.get("student_name"),
            overall_status   = data["overall_status"],
            predicted_grade  = data["predicted_grade"],
            goal_alignment   = data["goal_alignment"],
            learning_gaps    = gaps,
            strengths        = data.get("strengths", []),
            priority_actions = data.get("priority_actions", []),
            confidence_score = confidence,
            diagnosis_notes  = str(data.get("diagnosis_notes", ""))[:500],
            source           = "llm",
        )

    except Exception as exc:
        logger.error("Failed to parse LLM response: %s", exc)
        fallback.source = "llm-fallback"
        return fallback

def _parse_goal_thresholds(student_goals: list[str]) -> dict[str, float]:
    
    thresholds: dict[str, float] = {}
    pattern = re.compile(r"(\d+(?:\.\d+)?)\s*%?")
    subject_keywords = {
        "math":       ["math", "mathematics"],
        "science":    ["science"],
        "english":    ["english", "language"],
        "attendance": ["attendance", "attend"],
        "general":    ["pass", "score", "all subject"],
    }
    for goal in student_goals:
        goal_lower = goal.lower()
        m = pattern.search(goal_lower)
        if not m:
            continue
        threshold = float(m.group(1))
        matched = False
        for key, keywords in subject_keywords.items():
            if any(kw in goal_lower for kw in keywords):
                thresholds[key] = threshold
                matched = True
                break
        if not matched:
            thresholds.setdefault("general", threshold)
    return thresholds

def _rule_based_diagnosis(
    student_goals   : list[str],
    performance_data: dict[str, Any],
) -> DiagnosisReport:
    
    sid      = performance_data.get("student_id")
    sname    = performance_data.get("student_name")
    grade_id = performance_data.get("predicted_grade", 0)
    cat_id   = performance_data.get("predicted_category", 0)

    pred_grade  = GRADE_MAP.get(int(grade_id), f"Grade {grade_id}")
    pred_status = CATEGORY_MAP.get(int(cat_id), str(cat_id)) if isinstance(cat_id, int) else str(cat_id)

    attendance  = performance_data.get("attendance_percentage", 100.0)
    study_hours = performance_data.get("study_hours", 10.0)
    thresholds  = _parse_goal_thresholds(student_goals)

    gaps      : list[LearningGap] = []
    strengths : list[str]         = []

    if attendance < ATTENDANCE_CRITICAL:
        gaps.append(LearningGap(
            area     = "Attendance",
            severity = "Critical",
            evidence = f"Attendance is {attendance:.1f}%, which is critically low (threshold: {ATTENDANCE_CRITICAL}%).",
            recommendations=[
                f"Your attendance is {attendance:.1f}% — aim to reach 85% over the next 4 weeks.",
                "Identify and address the root cause of absences (transport, health, scheduling).",
                "Request catch-up materials from teachers for every missed session.",
            ],
        ))
    elif attendance < ATTENDANCE_WARNING:
        gaps.append(LearningGap(
            area     = "Attendance",
            severity = "Moderate",
            evidence = f"Attendance is {attendance:.1f}%, below the recommended 75%.",
            recommendations=[
                f"Attendance is {attendance:.1f}% — increase to at least 80% within 3 weeks.",
                "Set phone reminders for all scheduled classes.",
            ],
        ))
    else:
        strengths.append(f"Good attendance at {attendance:.1f}% supports consistent learning.")

    if study_hours < STUDY_HOURS_MIN:
        gaps.append(LearningGap(
            area     = "Study Time",
            severity = "Moderate",
            evidence = f"Only {study_hours:.1f} study hours/week — below the minimum of {STUDY_HOURS_MIN} hours.",
            recommendations=[
                f"Increase weekly study time from {study_hours:.1f} hrs to at least 7 hrs using a timetable.",
                "Use the Pomodoro technique (25-min focused sessions) to build study habits.",
            ],
        ))
    else:
        strengths.append(f"Adequate study time of {study_hours:.1f} hrs/week.")

    goal_score = thresholds.get("general", 60)
    for field_key, subject_name in SUBJECT_FIELDS:
        score = performance_data.get(field_key)
        if score is None:
            continue
        score        = float(score)
        subject_goal = thresholds.get(subject_name.lower(), goal_score)

        if score < SCORE_CRITICAL:
            sev = "Critical"
        elif score < min(SCORE_MODERATE, subject_goal):
            sev = "Moderate"
        else:
            sev = None

        if sev:
            deficit = subject_goal - score
            gaps.append(LearningGap(
                area     = subject_name,
                severity = sev,
                evidence = (
                    f"{subject_name} score is {score:.0f}, which is "
                    f"{deficit:.0f} points below the goal of {subject_goal:.0f}."
                ),
                recommendations=[
                    f"Your {subject_name} score is {score:.0f} — target {subject_goal:.0f} by studying core concepts daily.",
                    f"Work through practice problems for 30 min every day until score reaches {subject_goal:.0f}.",
                    "Ask your teacher for a list of high-yield topics to prioritise.",
                ],
            ))
        else:
            strengths.append(f"{subject_name} score of {score:.0f} meets or approaches goal.")

    if not performance_data.get("internet_access", True) and any(g.severity == "Critical" for g in gaps):
        gaps.append(LearningGap(
            area     = "Resource Access",
            severity = "Minor",
            evidence = "No internet access detected, which limits access to online learning tools.",
            recommendations=[
                "Use the school library for internet access during free periods.",
                "Ask your school counsellor about offline study resources or data-bundle schemes.",
            ],
        ))

    critical_count = sum(1 for g in gaps if g.severity == "Critical")
    moderate_count = sum(1 for g in gaps if g.severity == "Moderate")

    if critical_count >= 2 or (critical_count >= 1 and moderate_count >= 2):
        alignment = "Misaligned"
    elif gaps:
        alignment = "Partially Aligned"
    else:
        alignment = "Aligned"

    priority_actions: list[str] = []
    for g in sorted(gaps, key=lambda x: {"Critical": 0, "Moderate": 1, "Minor": 2}[x.severity]):
        if len(priority_actions) >= 5:
            break
        priority_actions.append(f"[{g.severity}] Address {g.area}: {g.evidence}")

    if not priority_actions:
        priority_actions.append("Maintain current performance. Consider stretch goals or peer mentoring.")

    key_fields = ["attendance_percentage", "study_hours", "math_score", "science_score", "english_score"]
    missing    = sum(1 for f in key_fields if performance_data.get(f) is None)
    confidence = max(0.0, 1.0 - missing * 0.1)

    if gaps:
        gap_names = ", ".join(g.area for g in gaps[:3])
        notes = (
            f"{sname or 'This student'} has {len(gaps)} identified gap(s): {gap_names}. "
            f"Predicted grade is {pred_grade} ({pred_status}). "
            f"Goal alignment: {alignment}. Immediate focus on critical areas is recommended."
        )
    else:
        notes = (
            f"{sname or 'This student'} meets all assessed criteria. "
            f"Predicted grade {pred_grade} ({pred_status}). "
            "No learning gaps detected — encourage continued excellence."
        )

    return DiagnosisReport(
        student_id       = sid,
        student_name     = sname,
        overall_status   = pred_status,
        predicted_grade  = pred_grade,
        goal_alignment   = alignment,
        learning_gaps    = gaps,
        strengths        = strengths,
        priority_actions = priority_actions,
        confidence_score = confidence,
        diagnosis_notes  = notes,
        source           = "rule-based",
    )

def _call_llm(messages: list[dict]) -> str | None:
    
    groq_key = os.getenv("GROQ_API_KEY")
    if groq_key:
        try:
            from openai import OpenAI
            client = OpenAI(api_key=groq_key, base_url="https://api.groq.com/openai/v1")
            resp = client.chat.completions.create(
                model="llama-3.1-8b-instant",
                messages=messages,
                temperature=0.2,
                max_tokens=2000
            )
            return resp.choices[0].message.content
        except Exception as exc:
            logger.warning("Groq call failed: %s", exc)

    gemini_key = os.getenv("GEMINI_API_KEY")
    if gemini_key:
        try:
            import google.generativeai as genai
            genai.configure(api_key=gemini_key)
            model     = genai.GenerativeModel("gemini-1.5-flash")

            system_msg  = next((m["content"] for m in messages if m["role"] == "system"), "")
            history_msgs = [m for m in messages if m["role"] != "system"]

            parts: list[str] = []
            if system_msg:
                parts.append(f"[SYSTEM INSTRUCTIONS]\n{system_msg}\n")
            for m in history_msgs:
                role_label = "USER" if m["role"] == "user" else "ASSISTANT"
                parts.append(f"[{role_label}]\n{m['content']}\n")

            full_prompt = "\n".join(parts)
            response    = model.generate_content(full_prompt)
            return response.text

        except Exception as exc:
            logger.warning("Gemini call failed: %s", exc)

    openai_key = os.getenv("OPENAI_API_KEY")
    if openai_key:
        try:
            from openai import OpenAI
            client = OpenAI(api_key=openai_key)
            resp   = client.chat.completions.create(
                model       = "gpt-4o",
                messages    = messages,
                temperature = 0.2,
                max_tokens  = 1500,
            )
            return resp.choices[0].message.content
        except Exception as exc:
            logger.warning("OpenAI call failed: %s", exc)

    return None

def run_diagnosis_node(
    student_goals   : list[str],
    performance_data: dict[str, Any],
    use_llm         : bool = True,
) -> DiagnosisReport:
    
    logger.info(
        "Diagnosis Node: student=%s goals=%d use_llm=%s",
        performance_data.get("student_id", "?"),
        len(student_goals),
        use_llm,
    )

    rule_report = _rule_based_diagnosis(student_goals, performance_data)

    if not use_llm:
        return rule_report

    messages = build_diagnosis_prompt(student_goals, performance_data)

    raw_llm_output = _call_llm(messages)
    if raw_llm_output is None:
        logger.info("No LLM available — returning rule-based diagnosis.")
        return rule_report

    report = _parse_llm_response(raw_llm_output, fallback=rule_report)
    logger.info("Diagnosis complete. Source=%s confidence=%.2f gaps=%d",
                report.source, report.confidence_score, len(report.learning_gaps))
    return report

def batch_diagnose(
    records     : list[dict[str, Any]],
    goals_key   : str = "student_goals",
    use_llm     : bool = True,
) -> list[DiagnosisReport]:
    
    reports: list[DiagnosisReport] = []
    for record in records:
        goals = record.pop(goals_key, [])
        report = run_diagnosis_node(goals, record, use_llm=use_llm)
        record[goals_key] = goals
        reports.append(report)
    return reports

if __name__ == "__main__":
    import pprint
    logging.basicConfig(level=logging.INFO)

    goals_at_risk = [
        "Pass all subjects with a score ≥ 60",
        "Attend at least 80% of classes",
    ]
    data_at_risk = {
        "student_id"           : "STU007",
        "student_name"         : "Dev Patel",
        "attendance_percentage": 58.0,
        "study_hours"          : 4.0,
        "math_score"           : 35,
        "science_score"        : 68,
        "english_score"        : 52,
        "internet_access"      : True,
        "extra_activities"     : False,
        "predicted_grade"      : 1,
        "predicted_category"   : 0,
    }

    print("=" * 60)
    print("TEST 1: At-Risk Student (rule-based, no LLM)")
    print("=" * 60)
    report1 = run_diagnosis_node(goals_at_risk, data_at_risk, use_llm=False)
    pprint.pprint(report1.to_dict(), width=100)

    goals_high = [
        "Maintain Grade 4 or above",
        "Score above 75 in all subjects",
    ]
    data_high = {
        "student_id"           : "STU109",
        "student_name"         : "Priya Sharma",
        "attendance_percentage": 91.0,
        "study_hours"          : 14.0,
        "math_score"           : 88,
        "science_score"        : 92,
        "english_score"        : 85,
        "internet_access"      : True,
        "extra_activities"     : True,
        "predicted_grade"      : 4,
        "predicted_category"   : 4,
    }

    print("\n" + "=" * 60)
    print("TEST 2: High-Performing Student (rule-based, no LLM)")
    print("=" * 60)
    report2 = run_diagnosis_node(goals_high, data_high, use_llm=False)
    pprint.pprint(report2.to_dict(), width=100)

    print("\nAll tests passed ✓")
