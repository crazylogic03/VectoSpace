from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver
from src.agent.state import AgentState

def diagnose_node(state: AgentState):
    print("[Node] Executing diagnosis...")
    return {"learning_gaps": "Mocked learning gaps based on performance data."}

def retrieve_node(state: AgentState):
    print("[Node] Executing RAG retrieval...")
    try:
        from rag.retriever import run_retrieval_node
        result = run_retrieval_node(state)
        print(f"[Node] RAG retrieval complete — {len(result.get('resources', []))} resource(s) found.")
        return result
    except Exception as exc:
        print(f"[Node] RAG retrieval failed ({exc}). Returning empty resources.")
        return {"resources": []}

def planner_node(state: AgentState):
    print("[Node] Executing planner...")
    try:
        from agent.planner import run_planner_node
        result = run_planner_node(state)
        print("[Node] Planner complete.")
        return result
    except Exception as exc:
        print(f"[Node] Planner failed ({exc}). Returning fallback plan.")
        plan = "Fallback 4-week study plan focusing on Calculus."
        return {"study_plan": plan, "final_report": {"status": "Complete", "plan": plan}}

def final_report_node(state: AgentState):
    print("[Node] Assembling Final Report...")
    from src.agent.schema import FinalReport
    
    rep_raw = state.get("final_report_raw")
    gaps_raw = state.get("learning_gaps", [])
    
    if hasattr(gaps_raw, "to_dict"):
        gaps_dicts = [g.to_dict() for g in getattr(gaps_raw, "learning_gaps", [])]
    else:
        gaps_dicts = gaps_raw if isinstance(gaps_raw, list) else []

    if isinstance(rep_raw, dict):
        for key in ["identified_learning_gaps", "recommended_resources"]:
            if key in rep_raw and isinstance(rep_raw[key], list):
                processed_list = []
                for item in rep_raw[key]:
                    if isinstance(item, dict):
                        processed_list.append(item.get("area", item.get("title", str(item))))
                    else:
                        processed_list.append(str(item))
                rep_raw[key] = processed_list

    try:
        report = FinalReport(
            student_id="Auto",
            student_name="Student",
            overall_status="Generated",
            predicted_grade="N/A",
            goal_alignment="N/A",
            learning_gaps=gaps_dicts,
            strengths=[],
            priority_actions=[],
            study_plan_metadata=rep_raw,
            retrieved_resources=state.get("resources", [])
        )
        return {"final_report": report}
    except Exception as exc:
        print(f"[Node] Schema validation failed: {exc}")
        return {"final_report": None}

def build_graph():
    workflow = StateGraph(AgentState)
    workflow.add_node("diagnose", diagnose_node)
    workflow.add_node("retrieve", retrieve_node)
    workflow.add_node("plan", planner_node)
    workflow.add_node("final_report", final_report_node)
    workflow.set_entry_point("diagnose")
    workflow.add_edge("diagnose", "retrieve")
    workflow.add_edge("retrieve", "plan")
    workflow.add_edge("plan", "final_report")
    workflow.add_edge("final_report", END)
    memory = MemorySaver()
    app = workflow.compile(checkpointer=memory)
    return app

agent_app = build_graph()
