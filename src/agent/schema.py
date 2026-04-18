from pydantic import BaseModel, Field
from typing import List, Dict, Any, Optional

class ReportGap(BaseModel):
    area: str
    severity: str
    evidence: str
    recommendations: List[str]

class ReportStep(BaseModel):
    step: int
    title: str
    activities: List[str]

class MultiStepStudyPlan(BaseModel):
    executive_summary: str
    identified_learning_gaps: List[str]
    recommended_resources: List[str]
    multi_step_study_plan: List[ReportStep]
    weekly_goals: Dict[str, str]

class FinalReport(BaseModel):
    """
    Pydantic Schema representing the consolidated Final Report
    as defined by the architecture diagram.
    """
    student_id: Optional[str] = None
    student_name: Optional[str] = None
    overall_status: str
    predicted_grade: str
    goal_alignment: str
    learning_gaps: List[ReportGap]
    strengths: List[str]
    priority_actions: List[str]
    study_plan_metadata: Optional[MultiStepStudyPlan] = None
    retrieved_resources: List[Dict[str, str]] = Field(default_factory=list)
