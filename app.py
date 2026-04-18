import streamlit as st
import pandas as pd
import numpy as np
import pickle
import os
import sys
import json
from dotenv import load_dotenv

load_dotenv()

ROOT = os.path.dirname(__file__)
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "src", "ml"))
from recommender import generate_recommendations
from agent.diagnosis import run_diagnosis_node, DiagnosisReport
from agent.planner import run_planner_node
from src.agent.graph import final_report_node
from extensions.pdf_export import PDFExportExtension
from extensions.quiz_generator import generate_personalized_quiz
from src.ml.utils import preprocess_raw_data
import altair as alt

try:
    from rag.retriever import retrieve_resources_for_gaps
    RAG_AVAILABLE = True
except ImportError:
    RAG_AVAILABLE = False

MODEL_DIR = os.path.join(ROOT, "model_artifacts")

GRADE_MAP    = {0: "Grade 0", 1: "Grade 1", 2: "Grade 2",
                3: "Grade 3", 4: "Grade 4", 5: "Grade 5"}
CATEGORY_MAP = {0: "At-Risk",       1: "Below-Average", 2: "Average",
                3: "Above-Average", 4: "High-Performing", 5: "Exceptional"}

SEVERITY_COLOR = {
    "Critical": "#ef4444",
    "Moderate": "#f97316",
    "Minor":    "#eab308",
}
SEVERITY_BG = {
    "Critical": "#2d1117",
    "Moderate": "#2d1a0a",
    "Minor":    "#1f1c08",
}

STATUS_COLOR = {
    "At-Risk":         "#ef4444",
    "Below-Average":   "#f97316",
    "Average":         "#facc15",
    "Above-Average":   "#34d399",
    "High-Performing": "#22d3ee",
    "Exceptional":     "#a78bfa",
}
ALIGNMENT_COLOR = {
    "Misaligned":        "#ef4444",
    "Partially Aligned": "#f97316",
    "Aligned":           "#22c55e",
}


st.set_page_config(
    page_title="VectoSpace",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

html, body, [class*="css"] { font-family: 'Inter', sans-serif; }

.vs-card {
    background: #111827;
    border: 1px solid #1f2937;
    border-radius: 14px;
    padding: 1.25rem 1.5rem;
    margin-bottom: 1rem;
}

.gap-card {
    border-radius: 12px;
    padding: 1rem 1.25rem;
    margin-bottom: 0.75rem;
    border-left: 4px solid;
}
.gap-card h4 { margin: 0 0 .35rem 0; font-size: 1rem; font-weight: 600; }
.gap-card p  { margin: 0 0 .5rem 0; font-size: .875rem; opacity: .85; }
.gap-card ul { margin: 0; padding-left: 1.15rem; font-size: .85rem; }
.gap-card ul li { margin-bottom: .2rem; }

.pill {
    display: inline-block;
    padding: .2rem .7rem;
    border-radius: 999px;
    font-size: .78rem;
    font-weight: 600;
    letter-spacing: .03em;
    margin-right: .4rem;
}

.metric-box {
    background: #1f2937;
    border-radius: 12px;
    padding: 1rem;
    text-align: center;
}
.metric-box .val { font-size: 1.8rem; font-weight: 700; }
.metric-box .lbl { font-size: .78rem; opacity: .6; margin-top: .15rem; }

.vs-divider {
    border: none;
    border-top: 1px solid #374151;
    margin: 1.5rem 0;
}

.source-llm      { color: #a78bfa; }
.source-rule     { color: #34d399; }
.source-fallback { color: #f97316; }

.conf-track {
    background: #1f2937;
    border-radius: 999px;
    height: 8px;
    width: 100%;
    overflow: hidden;
}
.conf-fill {
    height: 100%;
    border-radius: 999px;
    background: linear-gradient(90deg, #6366f1, #22d3ee);
}

.resource-card {
    background: #111827;
    border: 1px solid #1f2937;
    border-radius: 12px;
    padding: 1rem 1.25rem;
    margin-bottom: .75rem;
    transition: border-color .2s;
}
.resource-card:hover {
    border-color: #374151;
}
.rc-title { font-size: 1rem; font-weight: 600; color: #f3f4f6; margin-bottom: .15rem; }
.rc-url   { font-size: .75rem; color: #6366f1; text-decoration: none; display: block; margin-bottom: .4rem; }
.rc-summary { font-size: .85rem; color: #9ca3af; line-height: 1.45; }

.rc-gap-pill {
    display: inline-block;
    height: 5px;
    border-radius: 999px;
    background: linear-gradient(90deg, #6366f1, #22d3ee);
}
.rc-gap-group {
    margin-bottom: 1.1rem;
}
.rc-gap-label {
    font-size: .8rem;
    font-weight: 600;
    letter-spacing: .05em;
    text-transform: uppercase;
    opacity: .5;
    margin-bottom: .4rem;
}

.rag-status {
    display: inline-flex;
    align-items: center;
    gap: .3rem;
    font-size: .78rem;
    font-weight: 600;
    padding: .2rem .65rem;
    border-radius: 999px;
    border: 1px solid;
}
.rag-on  { color: #34d399; background: #05190f; border-color: #34d39955; }
.rag-off { color: #64748b; background: #1e293b; border-color: #33415555; }
</style>
""", unsafe_allow_html=True)

with st.sidebar:
    st.markdown("## VectoSpace")
    st.caption("Student Performance Intelligence")
    st.markdown("---")

    st.markdown("### Diagnosis Mode")
    use_llm = st.toggle(
        "Enable LLM Diagnosis",
        value=True
    )
    if use_llm:
        st.info("LLM mode enabled.", icon="✨")
    else:
        st.info("Rule-based mode enabled.", icon="⚡")

    st.markdown("---")

    st.markdown("### Resource Retrieval (RAG)")
    if RAG_AVAILABLE:
        use_rag = st.toggle(
            "Enable RAG",
            value=True
        )
        rag_use_llm_summary = st.toggle(
            "LLM Summaries",
            value=True
        )
        rag_top_k = st.slider(
            "Resources per gap",
            min_value=1, max_value=5, value=3,
            help="Number of resources retrieved per learning gap.",
        )
        if use_rag:
            st.markdown('<span class="rag-status rag-on">🟢 RAG Active</span>', unsafe_allow_html=True)
        else:
            st.markdown('<span class="rag-status rag-off">⚪ RAG Disabled</span>', unsafe_allow_html=True)
    else:
        use_rag = False
        rag_use_llm_summary = False
        rag_top_k = 3
        st.warning("⚠️ RAG package not found. Run `pip install -r requirements.txt` to enable resource retrieval.", icon="📦")

    st.markdown("---")

    st.markdown("### Student Learning Goals")
    st.caption("One goal per line. These are fed into the Diagnosis Node along with each student's data.")
    goals_raw = st.text_area(
        label="Goals (one per line)",
        value="Pass all subjects with a score ≥ 60\nAttend at least 80% of classes\nImprove study hours to 7+ per week",
        height=160,
        label_visibility="collapsed",
        key="goals_input",
    )
    student_goals = [g.strip() for g in goals_raw.splitlines() if g.strip()]
    st.markdown(f"`{len(student_goals)}` goal(s) configured")

    st.markdown("---")
    st.markdown("### About")
    st.markdown(
        "Upload a student CSV file to get **ML-powered grade predictions**, "
        "**learning gap diagnosis**, **RAG-powered resource retrieval**, "
        "and **personalised study recommendations**."
    )

st.markdown("""
<div style="padding:1.5rem 0 1rem 0;">
  <h1 style="margin:0;font-size:2rem;font-weight:700;">
    🎓 Student Performance Predictor
    <span style="font-size:1rem;font-weight:400;opacity:.5;margin-left:.5rem;">+ Diagnosis · RAG Engine</span>
  </h1>
  <p style="opacity:.55;margin:.3rem 0 0 0;">
    Upload student data · Get grade predictions · Identify learning gaps · Retrieve learning resources · Generate personalised actions
  </p>
</div>
""", unsafe_allow_html=True)

@st.cache_resource(show_spinner="Loading ML model…")
def load_model():
    with open(os.path.join(MODEL_DIR, "random_forest.pkl"), "rb") as f:
        model = pickle.load(f)
    with open(os.path.join(MODEL_DIR, "scaler.pkl"), "rb") as f:
        scaler = pickle.load(f)
    with open(os.path.join(MODEL_DIR, "scale_cols.pkl"), "rb") as f:
        scale_cols = pickle.load(f)
    return model, scaler, scale_cols




def _status_pill(label: str, color: str) -> str:
    return f'<span class="pill" style="background:{color}22;color:{color};border:1px solid {color}55;">{label}</span>'



_DOMAIN_ICON = {
    "Mathematics": "📐",
    "Science":     "🔬",
    "English":     "📖",
    "Attendance":  "🗓️",
    "Study Time":  "⏱️",
}

def _render_resources(resources: list[dict]):
    """Render RAG-retrieved resources grouped by learning gap."""
    if not resources:
        st.info("No resources retrieved — either no gaps were identified or RAG is disabled.", icon="📭")
        return

    from collections import defaultdict
    grouped: dict = defaultdict(list)
    for r in resources:
        grouped[r.get("gap", "General")].append(r)

    for gap_area, items in grouped.items():
        sev      = items[0].get("severity", "Moderate")
        sev_col  = SEVERITY_COLOR.get(sev, "#64748b")
        area_icon = _DOMAIN_ICON.get(gap_area, "📚")

        st.markdown(
            f'<div class="rc-gap-label" style="color:{sev_col};">{area_icon} {gap_area} '
            f'<span style="opacity:.6;font-weight:400;">({sev})</span></div>',
            unsafe_allow_html=True,
        )

        for r in items:
            score     = float(r.get("score", 0))
            score_pct = min(int(score * 100), 100)
            score_bar = (
                f'<span class="rc-score-bar">'  
                f'<span class="rc-score-fill" style="width:{score_pct}px;"></span>'
                f'{score:.3f}</span>'
            )

            sev_pill = (
                f'<span class="pill" style="background:{sev_col}22;color:{sev_col};'
                f'border:1px solid {sev_col}44;font-size:.7rem;">{sev}</span>'
            )

            title_html = (
                f'<a href="{r["url"]}" target="_blank" rel="noopener noreferrer">'
                f'{r["title"]}</a>'
            )

            st.markdown(
                f"""
                <div class="resource-card">
                  <div class="rc-header">
                    <p class="rc-title">{title_html}</p>
                    <div style="white-space:nowrap;">{sev_pill}</div>
                  </div>
                  <p class="rc-summary">{r['summary']}</p>
                  <div class="rc-meta">
                    {score_bar}
                    <span style="font-size:.73rem;color:
                  </div>
                </div>
                """,
                unsafe_allow_html=True,
            )

        st.markdown("<div style='height:.5rem'></div>", unsafe_allow_html=True)


def generate_chat_response(history: list, diagnosis_context: str, plan_context: str):
    """Call Groq LLM (8B) to generate a streaming chat response."""
    groq_key = os.getenv("GROQ_API_KEY")
    if not groq_key:
         return None
         
    try:
        from openai import OpenAI
        client = OpenAI(api_key=groq_key, base_url="https://api.groq.com/openai/v1")
        
        system_msg = (
            "You are a helpful AI Education Coach. You are chatting with a student "
            "about their learning diagnosis and study plan.\n\n"
            f"STUDENT DIAGNOSIS:\n{diagnosis_context}\n\n"
            f"STUDY PLAN:\n{plan_context}\n\n"
            "Keep your responses concise, encouraging, and focused on helping the student "
            "achieve their goals. Use formatting (bolding, bullet points) where appropriate."
        )
        
        full_history = [{"role": "system", "content": system_msg}] + history
        
        return client.chat.completions.create(
            model="llama-3.1-8b-instant",
            messages=full_history,
            temperature=0.4,
            max_tokens=800,
            stream=True
        )
    except Exception as e:
        print(f"Chat API error: {e}")
        return None



def _render_diagnosis(report: DiagnosisReport, row_raw: pd.Series, resources: list):
    stable_id = f"{report.student_id}_{report.student_name}".replace(" ", "_")
    status_col  = STATUS_COLOR.get(report.overall_status, "#64748b")
    align_col   = ALIGNMENT_COLOR.get(report.goal_alignment, "#64748b")
    source_cls  = {"llm": "source-llm", "rule-based": "source-rule"}.get(report.source, "source-fallback")
    source_lbl = {"llm": "LLM-powered", "rule-based": "Rule-based", "llm-fallback": "LLM fallback"}.get(report.source, report.source)

    st.markdown(
        f"""
        <div class="vs-card">
          <div style="display:flex;justify-content:space-between;align-items:center;flex-wrap:wrap;gap:.5rem;">
            <div>
              <span style="font-size:1.2rem;font-weight:700;">{report.student_name or "Student"}</span>
              <span style="opacity:.45;font-size:.85rem;margin-left:.5rem;">#{report.student_id or "—"}</span>
            </div>
            <div>
              {_status_pill(report.overall_status, status_col)}
              {_status_pill(report.predicted_grade, "#374151")}
              {_status_pill(report.goal_alignment, align_col)}
              <span class="pill {source_cls}" style="background:{status_col}11;">{source_lbl}</span>
            </div>
          </div>
          <hr class="vs-divider" style="margin:.8rem 0;">
          <p style="font-size:.9rem;opacity:.75;margin:0;">{report.diagnosis_notes}</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    pct = int(report.confidence_score * 100)
    st.markdown(
        f"""
        <div style="display:flex;align-items:center;gap:.75rem;margin-bottom:1rem;">
          <span style="font-size:.8rem;opacity:.55;white-space:nowrap;">Diagnosis confidence</span>
          <div class="conf-track" style="flex:1;">
            <div class="conf-fill" style="width:{pct}%;"></div>
          </div>
          <span style="font-size:.85rem;font-weight:600;">{pct}%</span>
        </div>
        """,
        unsafe_allow_html=True,
    )

    tab_gaps, tab_strengths, tab_actions, tab_resources, tab_plan, tab_quiz, tab_chat, tab_raw = st.tabs(
        ["Gaps", "Strengths", "Actions", "Resources", "Study Plan", "Practice Quiz", "Chat", "Raw"]
    )

    with tab_gaps:
        if not report.learning_gaps:
            st.success("No learning gaps identified — this student meets all assessed criteria.")
        else:
            for gap in sorted(report.learning_gaps,
                               key=lambda g: {"Critical": 0, "Moderate": 1, "Minor": 2}[g.severity]):
                sev_col = SEVERITY_COLOR.get(gap.severity, "#64748b")
                sev_bg  = SEVERITY_BG.get(gap.severity, "#111827")
                recs_html = "".join(f"<li>{r}</li>" for r in gap.recommendations)
                st.markdown(
                    f"""
                    <div class="gap-card"
                         style="background:{sev_bg};border-left-color:{sev_col};">
                      <h4 style="color:{sev_col};">
                        {gap.area}
                        <span class="pill" style="font-size:.7rem;background:{sev_col}22;
                              color:{sev_col};border:1px solid {sev_col}44;
                              margin-left:.5rem;">{gap.severity}</span>
                      </h4>
                      <p>{gap.evidence}</p>
                      <ul>{recs_html}</ul>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

    with tab_strengths:
        if not report.strengths:
            st.info("No explicit strengths recorded.")
        else:
            for s in report.strengths:
                st.markdown(f"- {s}")

    with tab_actions:
        for i, action in enumerate(report.priority_actions, 1):
            st.markdown(
                f"""
                <div style="display:flex;gap:.75rem;align-items:flex-start;margin-bottom:.6rem;">
                  <span style="background:
                               width:1.5rem;height:1.5rem;display:flex;align-items:center;
                               justify-content:center;font-size:.75rem;font-weight:700;
                               flex-shrink:0;">{i}</span>
                  <span style="font-size:.88rem;padding-top:.15rem;">{action}</span>
                </div>
                """,
                unsafe_allow_html=True,
            )

    with tab_resources:
        resources = st.session_state.get(f"_resources_{stable_id}", None)
        if resources:
            _render_resources(resources)

    with tab_plan:
        plan_md = st.session_state.get(f"_plan_md_{stable_id}", None)
        plan_report = st.session_state.get(f"_plan_json_{stable_id}", None)
        
        if plan_md:
            st.markdown(plan_md)
            if plan_report:
                with st.expander("Show raw planner data"):
                    st.json(plan_report)
                
                exporter = PDFExportExtension()
                pdf_path, md_path = exporter.export_study_plan(
                    student_id=report.student_id or "Student", 
                    student_name=report.student_name or "N/A", 
                    study_plan_md=plan_md, 
                    final_report_json=plan_report
                )
                
                with open(pdf_path, "rb") as pdf_file:
                    st.download_button(
                        label="⬇️ Download Study Plan (PDF)",
                        data=pdf_file,
                        file_name=os.path.basename(pdf_path),
                        mime="application/pdf",
                        key=f"dl_pdf_plan_{stable_id}",
                    )
                    
    quiz_key = f"_quiz_data_{stable_id}"
    quiz_ans_key = f"_quiz_answers_{stable_id}"
    quiz_submitted_key = f"_quiz_submitted_{stable_id}"

    with tab_quiz:
        st.markdown(f"### Interactive Practice Quiz")
        
        if quiz_key not in st.session_state:
            st.info("No quiz generated yet.")
            if st.button("Generate Personalized Quiz", key=f"btn_gen_{report.student_id}"):
                with st.spinner("Designing assessment..."):
                    from extensions.quiz_generator import generate_personalized_quiz
                    quiz = generate_personalized_quiz(
                        [g.to_dict() for g in report.learning_gaps],
                        resources,
                        report.predicted_grade
                    )
                    if quiz:
                        st.session_state[quiz_key] = quiz
                        st.session_state[quiz_ans_key] = {}
                        st.session_state[quiz_submitted_key] = False
                        st.rerun()
                    else:
                        st.error("Failed to generate quiz. Please check your API key.")
        else:
            quiz = st.session_state[quiz_key]
            submitted = st.session_state[quiz_submitted_key]
            
            for i, q in enumerate(quiz):
                st.markdown(f"**Q{i+1}: {q['question']}**")
                
                current_selection = st.session_state[quiz_ans_key].get(i)
                choice = st.radio(
                    f"Select answer for Q{i+1}:", 
                    q['options'], 
                    index=q['options'].index(current_selection) if current_selection in q['options'] else None,
                    key=f"q_{i}_{stable_id}",
                    disabled=submitted,
                    label_visibility="collapsed"
                )
                
                if choice:
                    st.session_state[quiz_ans_key][i] = choice
                
                if submitted:
                    is_correct = choice == q['answer']
                    color = "green" if is_correct else "red"
                    icon = "✅" if is_correct else "❌"
                    st.markdown(f"<p style='color:{color};font-weight:bold;'>{icon} Result: {choice}</p>", unsafe_allow_html=True)
                    if not is_correct:
                        st.markdown(f"**Correct Answer:** {q['answer']}")
                    st.success(f"**Explanation:** {q['explanation']}")
                
                st.markdown("---")
            
            if not submitted:
                if st.button("🏁 Submit Quiz", key=f"submit_q_{report.student_id}"):
                    st.session_state[quiz_submitted_key] = True
                    st.rerun()
            else:
                score = sum(1 for i, q in enumerate(quiz) if st.session_state[quiz_ans_key].get(i) == q['answer'])
                st.metric("Final Score", f"{score}/{len(quiz)}")
                if st.button("🔄 Retake Quiz", key=f"retake_q_{report.student_id}"):
                    del st.session_state[quiz_key]
                    st.rerun()

    with tab_chat:
        st.markdown("### AI Education Coach")
        chat_key = f"_chat_history_{stable_id}"
        if chat_key not in st.session_state:
            st.session_state[chat_key] = [{"role": "assistant", "content": "Hello! I am your AI coach. How can I help you regarding this learning diagnosis and plan?"}]
            
        for msg in st.session_state[chat_key]:
            with st.chat_message(msg["role"]):
                st.write(msg["content"])
                
        if user_input := st.chat_input("Ask about the diagnosis or study plan..."):
            st.session_state[chat_key].append({"role": "user", "content": user_input})
            with st.chat_message("user"):
                st.write(user_input)
            
            with st.chat_message("assistant"):
                diagnosis_ctx = report.diagnosis_notes
                plan_ctx = st.session_state.get(f"_plan_md_{stable_id}", "")
                
                response_placeholder = st.empty()
                full_response = ""
                
                stream = generate_chat_response(
                    history=st.session_state[chat_key],
                    diagnosis_context=diagnosis_ctx,
                    plan_context=plan_ctx
                )
                
                if stream:
                    for chunk in stream:
                        content = chunk.choices[0].delta.content
                        if content:
                            full_response += content
                            response_placeholder.markdown(full_response + "▌")
                    response_placeholder.markdown(full_response)
                else:
                    full_response = "I'm sorry, I'm having trouble connecting to the brain. Please check your API keys."
                    response_placeholder.error(full_response)
                
                st.session_state[chat_key].append({"role": "assistant", "content": full_response})

    with tab_raw:
        report_dict = report.to_dict()
        
        from src.agent.schema import FinalReport
        final_pydantic_report = st.session_state.get(f"_final_report_obj_{stable_id}")
        
        st.markdown("### Raw Output")
        st.json(report_dict, expanded=False)
        
        if final_pydantic_report:
            st.markdown("#### Plan Metadata")
            st.json(final_pydantic_report.model_dump(), expanded=False)
        st.download_button(
            label="⬇️ Download Diagnosis JSON",
            data=json.dumps(report_dict, indent=2, default=str),
            file_name=f"diagnosis_{report.student_id or 'student'}.json",
            mime="application/json",
            key=f"dl_json_{stable_id}",
        )


uploaded_file = st.file_uploader(
    "📂 Upload Student Data (CSV)",
    type=["csv"],
    help="CSV should include columns like attendance_percentage, study_hours, math_score, science_score, english_score, internet_access, etc.",
)
st.caption("ℹ️ **Handling Missing Data**: If your CSV is missing any expected feature columns, VectoSpace will automatically pad them with a default value of `0` to ensure the model remains stable.")

if uploaded_file is None:
    st.markdown("""
    <div style="text-align:center;padding:3rem 0;opacity:.5;">
      <div style="font-size:3rem;">📋</div>
      <p style="font-size:1rem;margin:.5rem 0 0 0;">Upload a student CSV file using the button above to get started.</p>
    </div>
    """, unsafe_allow_html=True)
    st.stop()

raw_df      = pd.read_csv(uploaded_file)
original_df = raw_df.copy()
model, scaler, scale_cols = load_model()

has_strings  = raw_df.select_dtypes(exclude="number").shape[1] > 0
processed_df = preprocess_raw_data(raw_df, scaler, scale_cols) if has_strings else raw_df.copy()
if "final_grade" in processed_df.columns:
    processed_df.drop(columns=["final_grade"], inplace=True)

input_df = processed_df.copy()
for col in model.feature_names_in_:
    if col not in input_df.columns:
        input_df[col] = 0
input_df = input_df[model.feature_names_in_]

predictions  = model.predict(input_df)

results_df = original_df.copy()
results_df["Predicted Grade"]  = [GRADE_MAP.get(p, f"Grade {p}") for p in predictions]
results_df["Classification"]   = [CATEGORY_MAP.get(p, "Unknown") for p in predictions]
results_df["_pred_int"]        = predictions

tab_overview, tab_data, tab_search = st.tabs(
    ["📊 Overview", "📄 Full Dataset", "🔍 Student Diagnosis"]
)

with tab_overview:
    st.markdown("### Performance Overview")

    total      = len(results_df)
    grade_cts  = pd.Series(predictions).value_counts()
    top_grade  = GRADE_MAP.get(int(grade_cts.idxmax()), "N/A")
    top_cat    = CATEGORY_MAP.get(int(grade_cts.idxmax()), "N/A")
    at_risk_n  = int((predictions <= 1).sum())

    m1, m2, m3, m4 = st.columns(4)
    with m1:
        st.metric("Total Students",     total)
    with m2:
        st.metric("Most Common Grade",  top_grade)
    with m3:
        st.metric("Most Common Status", top_cat)
    with m4:
        st.metric("⚠️ At-Risk Students", at_risk_n,
                  delta=f"{at_risk_n/total*100:.1f}% of class",
                  delta_color="inverse")

    st.markdown("<hr class='vs-divider'>", unsafe_allow_html=True)

    col1, col2 = st.columns(2)
    with col1:
        st.markdown("**Grade Distribution**")
        grade_dist = results_df["Predicted Grade"].value_counts().reset_index()
        grade_dist.columns = ["Grade", "Count"]
        
        chart1 = alt.Chart(grade_dist).mark_bar(cornerRadiusTopLeft=3, cornerRadiusTopRight=3).encode(
            x=alt.X('Grade:N', sort=['Grade 0', 'Grade 1', 'Grade 2', 'Grade 3', 'Grade 4', 'Grade 5']),
            y='Count:Q',
            color=alt.value("#6366f1"),
            tooltip=['Grade', 'Count']
        ).properties(height=300)
        st.altair_chart(chart1, width="stretch")

    with col2:
        st.markdown("**Classification Distribution**")
        cat_dist = results_df["Classification"].value_counts().reset_index()
        cat_dist.columns = ["Status", "Count"]
        
        chart2 = alt.Chart(cat_dist).mark_bar(cornerRadiusTopLeft=3, cornerRadiusTopRight=3).encode(
            x=alt.X('Status:N', sort=['At-Risk', 'Below-Average', 'Average', 'Above-Average', 'High-Performing', 'Exceptional']),
            y='Count:Q',
            color=alt.value("#f43f5e"),
            tooltip=['Status', 'Count']
        ).properties(height=300)
        st.altair_chart(chart2, width="stretch")

    st.markdown("<hr class='vs-divider'>", unsafe_allow_html=True)
    st.markdown("### Subject Scores & Resource Mapping")
    if student_goals:
        for g in student_goals:
            st.markdown(f"- {g}")
    else:
        st.warning("No goals configured. Add goals in the sidebar to power the Diagnosis Node.")

    st.markdown("<hr class='vs-divider'>", unsafe_allow_html=True)
    st.markdown("### Analysis Log")
    export_df = results_df.drop(columns=["_pred_int"], errors="ignore")
    st.download_button(
        label     = "⬇️ Download All Predictions as CSV",
        data      = export_df.to_csv(index=False),
        file_name = "predictions.csv",
        mime      = "text/csv",
    )

with tab_data:
    st.markdown("## Interactive Student Analysis")
    display_df = results_df.drop(columns=["_pred_int"], errors="ignore")
    st.dataframe(display_df, width="stretch")

with tab_search:
    st.markdown("### Select Student for Diagnosis")
    st.caption(
        "Search by student name, ID, or row number. "
        "The Diagnosis Node will analyse their performance against your configured goals."
    )

    if not student_goals:
        st.warning("⚠️ No learning goals are set. Please add goals in the sidebar before running diagnosis.")

    search_query = st.text_input(
        "Search student",
        placeholder="e.g. John, STU001, or row number 5",
        label_visibility="collapsed",
    )

    if not search_query.strip():
        st.info("👆 Enter a name, ID, or row number to look up a student.")
        st.stop()

    query   = search_query.strip()
    matched = pd.DataFrame()

    if query.isdigit():
        row_num = int(query) - 1
        if 0 <= row_num < len(results_df):
            matched = results_df.iloc[[row_num]]

    if matched.empty:
        q_low = query.lower()
        for col in ["student_id", "student_name", "name", "id"]:
            if col in results_df.columns:
                mask    = results_df[col].astype(str).str.lower().str.contains(q_low, na=False)
                matched = pd.concat([matched, results_df[mask]])
        matched = matched.drop_duplicates()

    if matched.empty:
        st.warning(
            f"No student found for **\"{query}\"**. "
            f"Try a different name, ID, or a row number between 1 and {len(results_df)}."
        )
        st.stop()

    st.success(f"Found **{len(matched)}** student(s)")
    st.dataframe(
        matched.drop(columns=["_pred_int"], errors="ignore"),
        width="stretch",
    )

    st.markdown("---")
    st.markdown("### Learning Diagnosis Report")

    if not student_goals:
        st.error(
            "Please configure learning goals in the sidebar to enable diagnosis."
        )
        st.stop()

    for idx in matched.index:
        row      = results_df.loc[idx]
        pred_int = int(row["_pred_int"])

        perf_data = {}
        for field in [
            "student_id", "student_name",
            "attendance_percentage", "study_hours",
            "math_score", "science_score", "english_score",
            "internet_access", "extra_activities",
        ]:
            val = row.get(field)
            if val is not None and not (isinstance(val, float) and np.isnan(val)):
                perf_data[field] = val

        perf_data["predicted_grade"]    = pred_int
        perf_data["predicted_category"] = pred_int

        sname = str(row.get("student_name", row.get("name", f"Student {idx + 1}")))
        sid   = str(row.get("student_id",   row.get("id", "—")))

        cache_config = {
            "goals": student_goals,
            "use_llm": use_llm,
            "use_rag": use_rag,
            "rag_use_llm_summary": rag_use_llm_summary,
            "rag_top_k": rag_top_k,
            "perf": perf_data
        }
        import hashlib
        config_hash = hashlib.md5(json.dumps(cache_config, sort_keys=True, default=str).encode()).hexdigest()
        
        stable_id = f"{sid}_{sname}".replace(" ", "_")
        cache_key = f"_diag_cache_{stable_id}"
        
        is_cached = cache_key in st.session_state and st.session_state[cache_key].get("hash") == config_hash
        
        with st.expander(
            f"{sname} (#{sid}) — {row['Predicted Grade']} · {row['Classification']}",
            expanded=True,
        ):
            if not is_cached:
                with st.spinner("Running Diagnosis Node…"):
                    report = run_diagnosis_node(
                        student_goals   = student_goals,
                        performance_data= perf_data,
                        use_llm         = use_llm,
                    )

                resources: list = []
                if use_rag and RAG_AVAILABLE and report.learning_gaps:
                    with st.spinner("Retrieving learning resources..."):
                        try:
                            gaps_dicts = [g.to_dict() for g in report.learning_gaps]
                            resources  = retrieve_resources_for_gaps(
                                learning_gaps = gaps_dicts,
                                use_llm       = rag_use_llm_summary,
                                top_k         = rag_top_k,
                            )
                        except Exception as rag_err:
                            st.warning(f"RAG retrieval encountered an error: {rag_err}", icon="⚠️")

                with st.spinner("Generating Study Plan..."):
                    state = {
                        "learning_gaps": [g.to_dict() for g in report.learning_gaps],
                        "resources": resources
                    }
                    plan_result = run_planner_node(state)
                
                state["study_plan"] = plan_result["study_plan"]
                state["final_report_raw"] = plan_result.get("final_report_raw")
                
                with st.spinner("Validating Report Schema..."):
                    final_node_out = final_report_node(state)
                    final_pydantic = final_node_out.get("final_report")
                
                st.session_state[cache_key] = {
                    "hash": config_hash,
                    "report": report,
                    "resources": resources,
                    "plan_md": plan_result["study_plan"],
                    "plan_json": plan_result.get("final_report_raw"),
                    "final_report_obj": final_pydantic
                }
                
                st.session_state[f"_plan_md_{stable_id}"] = plan_result["study_plan"]
                st.session_state[f"_plan_json_{stable_id}"] = plan_result.get("final_report_raw")
                st.session_state[f"_final_report_obj_{stable_id}"] = final_pydantic
                st.session_state[f"_resources_{stable_id}"] = resources
            else:
                cached_data = st.session_state[cache_key]
                report = cached_data["report"]
                resources = cached_data["resources"]
                
                if st.button("🔄 Refresh Analysis", key=f"refresh_{stable_id}"):
                    del st.session_state[cache_key]
                    st.rerun()

            _render_diagnosis(report, row, resources)

            if use_rag and RAG_AVAILABLE:
                if resources:
                    st.success(
                        f"{len(resources)} resource(s) retrieved. See the Resources tab.",
                        icon="✅",
                    )
                elif report.learning_gaps:
                    st.info("No matching resources found for these gaps.")

            with st.expander("Quick Study Recommendations", expanded=False):
                legacy_data = {
                    "attendance_percentage": row.get("attendance_percentage", 100),
                    "study_hours":           row.get("study_hours", 10),
                    "math_score":            row.get("math_score", 100),
                    "science_score":         row.get("science_score", 100),
                    "english_score":         row.get("english_score", 100),
                    "internet_access":       row.get("internet_access", 1),
                }
                recs = generate_recommendations(legacy_data, f"Grade {pred_int}")
                for i, r in enumerate(recs, 1):
                    st.write(f"{i}. {r}")