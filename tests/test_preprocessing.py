import os
import sys
import pandas as pd
import numpy as np
import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.ml.utils import preprocess_raw_data

def test_preprocess_drops_identifier_columns():
    """Ensure student_id and final_grade are properly dropped if present."""
    df = pd.DataFrame({
        "student_id": ["S1", "S2"],
        "final_grade": [1, 2],
        "study_hours": [10, 15]
    })
    
    processed = preprocess_raw_data(df, None, None)
    assert "student_id" not in processed.columns, "student_id should be dropped"
    assert "final_grade" not in processed.columns, "final_grade should be dropped"
    assert "study_hours" in processed.columns

def test_preprocess_handles_binary_mappings():
    """Ensure binary cols map yes/no correctly to 1/0."""
    df = pd.DataFrame({
        "internet_access": ["Yes", "no", None],
        "extra_activities": ["YES ", " NO", "Yes"]
    })
    
    processed = preprocess_raw_data(df, None, None)
    assert list(processed["internet_access"]) == [1, 0, 0]
    assert list(processed["extra_activities"]) == [1, 0, 1]

def test_preprocess_handles_missing_data_padded_internally():
    """
    Test nominal columns mappings. Ensure empty categories don't block.
    Note that missing expected feature columns alignment happens post-preprocessing in app.py.
    """
    df = pd.DataFrame({
        "school_type": ["public", "private"],
        "gender": ["F", "M"]
    })
    
    processed = preprocess_raw_data(df, None, None)
    assert "school_type_public" in processed.columns
    assert "gender_f" in processed.columns
