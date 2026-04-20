import json
import logging
import os
from dataclasses import dataclass, field
from typing import Any, Dict

logger = logging.getLogger(__name__)

try:
    from openai import OpenAI
except ImportError:
    OpenAI = None

SYSTEM_PROMPT = """
You are an expert AI Study Planner. Your goal is to create a structured, professional-grade study plan for a student based on their learning gaps and available resources.

OUTPUT CONTRACT:
1. Respond with VALID JSON ONLY.
2. The JSON must have the following keys:
   - "executive_summary": A high-level overview of the plan.
   - "identified_learning_gaps": A list of PLAIN STRINGS (e.g., ["Algebra Foundations", "Attendance Habits"]). Do NOT return objects or dictionaries here.
   - "recommended_resources": A list of PLAIN TEXT Resource Titles (e.g., ["Khan Academy", "Purdue OWL"]). Do NOT return objects or dictionaries here.
   - "multi_step_study_plan": A list of steps, each with "step" (int), "title", and "activities" (list of strings).
   - "weekly_goals": A dictionary (e.g., {"week_1": "...", "week_2": "..."}).

Keep recommendations actionable, realistic, and highly tailored to the provided data.
"""

def extract_json_from_text(text: str) -> dict:
    import re
    fence_match = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL)
    if fence_match:
        return json.loads(fence_match.group(1))
    
    start = text.find("{")
    end = text.rfind("}")
    if start != -1 and end != -1 and end > start:
        return json.loads(text[start : end + 1])
    try:
        return json.loads(text)
    except Exception as e:
        raise ValueError(f"Could not parse JSON. Output: {text}") from e

def _call_llm_for_plan(gaps: str, resources: str) -> dict | None:
    groq_key = os.getenv("GROQ_API_KEY")
    openai_key = os.getenv("OPENAI_API_KEY")
    gemini_key = os.getenv("GEMINI_API_KEY")

    if OpenAI and groq_key:
        try:
            client = OpenAI(api_key=groq_key, base_url="https://api.groq.com/openai/v1")
            messages = [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": f"GAPS:\n{gaps}\n\nRESOURCES:\n{resources}"}
            ]
            
            resp = client.chat.completions.create(
                model="llama-3.1-8b-instant",
                messages=messages,
                temperature=0.2,
                max_tokens=2000
            )
            raw = resp.choices[0].message.content
            return extract_json_from_text(raw)
        except Exception as e:
            logger.error("Groq Plan generation failed: %s", e)

    if OpenAI and openai_key:
        try:
            client = OpenAI(api_key=openai_key)
            messages = [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": f"GAPS:\n{gaps}\n\nRESOURCES:\n{resources}"}
            ]
            
            resp = client.chat.completions.create(
                model="gpt-4o",
                messages=messages,
                temperature=0.2,
                max_tokens=2000
            )
            raw = resp.choices[0].message.content
            return extract_json_from_text(raw)
        except Exception as e:
            logger.error("OpenAI Plan generation failed: %s", e)
    
    if gemini_key:
        try:
            import google.generativeai as genai
            genai.configure(api_key=gemini_key)
            model = genai.GenerativeModel("gemini-1.5-flash")
            full_prompt = SYSTEM_PROMPT + f"\n\nGAPS:\n{gaps}\n\nRESOURCES:\n{resources}"
            resp = model.generate_content(full_prompt)
            return extract_json_from_text(resp.text)
        except Exception as e:
            logger.error("Gemini Plan generation failed: %s", e)

    return None

def fallback_plan(gaps_data: Any, resources_data: Any) -> dict:
    
    return {
        "executive_summary": "Auto-generated baseline study plan focusing on primary weak areas.",
        "identified_learning_gaps": ["Fallback gap based on missing data"],
        "recommended_resources": ["Study guides", "Online tutorials"],
        "multi_step_study_plan": [
            {"step": 1, "title": "Review Basics", "activities": ["Read textbook chapters related to weak subjects."]},
            {"step": 2, "title": "Practice Problems", "activities": ["Complete 10 practice problems daily."]}
        ],
        "weekly_goals": {
            "week_1": "Establish a consistent 2-hour daily study routine.",
            "week_2": "Complete the first module of recommended resources.",
            "week_3": "Take a practice quiz to verify knowledge.",
            "week_4": "Incorporate advanced practice questions."
        }
    }

def run_planner_node(state: dict) -> dict:
    
    gaps_raw = state.get("learning_gaps", "No gaps identified.")
    res_raw = state.get("resources", [])
    
    if isinstance(gaps_raw, dict):
        gaps_str = json.dumps(gaps_raw.get("learning_gaps", []), indent=2)
    else:
        gaps_str = str(gaps_raw)[:1000]
        
    res_str = json.dumps(res_raw[:3], indent=2) if isinstance(res_raw, list) else str(res_raw)[:1000]

    plan_data = _call_llm_for_plan(gaps_str, res_str)
    
    if plan_data is None:
        plan_data = fallback_plan(gaps_raw, res_raw)
        
    markdown_plan = f"# Study Plan Executive Summary\n\n{plan_data.get('executive_summary', '')}\n\n"
    markdown_plan += "## Identified Learning Gaps\n"
    for gap in plan_data.get('identified_learning_gaps', []):
        markdown_plan += f"- {gap}\n"
    
    markdown_plan += "\n## Recommended Resources\n"
    for res in plan_data.get('recommended_resources', []):
        markdown_plan += f"- {res}\n"
        
    markdown_plan += "\n## Multi-Step Study Plan\n"
    for step in plan_data.get('multi_step_study_plan', []):
        markdown_plan += f"### Step {step.get('step', '?')}: {step.get('title', 'Phase')}\n"
        for act in step.get('activities', []):
            markdown_plan += f"- {act}\n"
            
    markdown_plan += "\n## Weekly Goals\n"
    weekly = plan_data.get('weekly_goals', {})
    for week, goal in weekly.items():
        markdown_plan += f"- **{week.replace('_', ' ').capitalize()}**: {goal}\n"

    return {
        "study_plan": markdown_plan,
        "final_report_raw": plan_data
    }
