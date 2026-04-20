import json
import logging
import os
import re
from typing import List, Dict, Any, Optional

logger = logging.getLogger(__name__)

try:
    from openai import OpenAI
except ImportError:
    OpenAI = None

QUIZ_SYSTEM_PROMPT = """
You are an expert AI Quiz Designer. Your goal is to create a personalized practice assessment for a student based on their learning gaps and Grade level.

OUTPUT CONTRACT:
1. Respond with a VALID JSON ARRAY ONLY.
2. Each object in the array must represent a question and have the following keys:
   - "question": The actual question text.
   - "options": A list of 4 multiple-choice options.
   - "answer": The string value of the correct option as it appears in the options list.
   - "explanation": A helpful 1-2 sentence explanation of why the answer is correct and how it relates to the gap.
3. Generate exactly 5 questions per quiz.
"""

def extract_json_array(text: str) -> List[Dict[str, Any]]:
    
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    
    match = re.search(r"(\[.*\])", text, re.DOTALL)
    if match:
        try:
            return json.loads(match.group(1))
        except json.JSONDecodeError:
            pass
            
    start = text.find("[")
    end = text.rfind("]")
    if start != -1 and end != -1 and end > start:
        try:
            return json.loads(text[start : end + 1])
        except json.JSONDecodeError:
            pass
            
    raise ValueError(f"Could not parse valid JSON array from response: {text[:200]}...")

def generate_personalized_quiz(gaps: List[Dict], resources: List[Dict], grade: str) -> List[Dict[str, Any]]:
    
    groq_key = os.getenv("GROQ_API_KEY")
    openai_key = os.getenv("OPENAI_API_KEY")
    gemini_key = os.getenv("GEMINI_API_KEY")

    gaps_context = json.dumps(gaps, indent=2)
    safe_resources = resources if resources is not None else []
    res_context = json.dumps([{"title": r.get("title"), "summary": r.get("summary")} for r in safe_resources[:3]], indent=2)
    user_content = f"GRADE: {grade}\nGAPS: {gaps_context}\nRESOURCES: {res_context}"
    
    messages = [
        {"role": "system", "content": QUIZ_SYSTEM_PROMPT},
        {"role": "user", "content": user_content}
    ]

    if OpenAI and groq_key:
        try:
            client = OpenAI(api_key=groq_key, base_url="https://api.groq.com/openai/v1")
            resp = client.chat.completions.create(
                model="llama-3.1-8b-instant",
                messages=messages,
                temperature=0.3,
                max_tokens=2000
            )
            if resp and resp.choices and len(resp.choices) > 0:
                raw_output = resp.choices[0].message.content
                if raw_output:
                    return extract_json_array(raw_output)
            logger.warning("Groq response was empty or malformed.")
        except Exception as e:
            logger.error(f"Groq Quiz generation failed: {e}")

    if OpenAI and openai_key:
        try:
            client = OpenAI(api_key=openai_key)
            resp = client.chat.completions.create(
                model="gpt-4o",
                messages=messages,
                temperature=0.3,
                max_tokens=2000
            )
            raw_output = resp.choices[0].message.content
            return extract_json_array(raw_output)
        except Exception as e:
            logger.error(f"OpenAI Quiz generation failed: {e}")

    if gemini_key:
        try:
            import google.generativeai as genai
            genai.configure(api_key=gemini_key)
            model = genai.GenerativeModel("gemini-1.5-flash")
            full_prompt = f"{QUIZ_SYSTEM_PROMPT}\n\n{user_content}"
            resp = model.generate_content(full_prompt)
            return extract_json_array(resp.text)
        except Exception as e:
            logger.error(f"Gemini Quiz generation failed: {e}")

    logger.warning("All LLM attempts for Quiz generation failed. Using fallback quiz.")
    return get_fallback_quiz()

def get_fallback_quiz() -> List[Dict[str, Any]]:
    
    return [
        {
            "question": "Which of the following is the most effective way to improve learning consistency?",
            "options": ["Cramming before exams", "Consistent daily study hours", "Skipping complex topics", "Only studying on weekends"],
            "answer": "Consistent daily study hours",
            "explanation": "Research shows that spaced repetition and daily habits lead to better long-term retention than cramming."
        },
        {
            "question": "If you identify a 'Critical' gap in a subject, what should be your first priority?",
            "options": ["Ignore it and focus on strengths", "Wait for the next exam", "Focus on fundamental concepts and seek help", "Change the subject"],
            "answer": "Focus on fundamental concepts and seek help",
            "explanation": "Critical gaps often stem from missing foundational knowledge; addressing these first prevents cumulative learning loss."
        }
    ]
