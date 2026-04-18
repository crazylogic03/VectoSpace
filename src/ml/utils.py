import pandas as pd
import numpy as np

def preprocess_raw_data(df, scaler, scale_cols):
    """
    Standard preprocessing for student performance data.
    - Drops identifiers
    - Maps binary categories
    - Maps ordinal travel and education categories
    - One-hot encodes nominals
    - Applies scaling if provided
    """
    df = df.copy()
    if "student_id"  in df.columns: df.drop(columns=["student_id"],  inplace=True)
    if "final_grade" in df.columns: df.drop(columns=["final_grade"], inplace=True)

    for col in df.select_dtypes(exclude="number").columns:
        df[col] = df[col].astype(str).str.strip().str.lower()

    binary_map = {"yes": 1, "no": 0}
    for col in ["internet_access", "extra_activities"]:
        if col in df.columns and not pd.api.types.is_numeric_dtype(df[col]):
            df[col] = df[col].map(binary_map).fillna(0).astype(int)

    travel_map = {"<15 min": 0, "15-30 min": 1, "30-60 min": 2, ">60 min": 3}
    if "travel_time" in df.columns and not pd.api.types.is_numeric_dtype(df["travel_time"]):
        df["travel_time"] = df["travel_time"].map(travel_map).fillna(0).astype(int)

    edu_map = {"no formal": 0, "high school": 1, "diploma": 2,
               "graduate": 3, "post graduate": 4, "phd": 5}
    if "parent_education" in df.columns and not pd.api.types.is_numeric_dtype(df["parent_education"]):
        df["parent_education"] = df["parent_education"].map(edu_map).fillna(0).astype(int)

    nominal_cols = [c for c in ["gender", "school_type", "study_method"] if c in df.columns]
    if nominal_cols:
        df = pd.get_dummies(df, columns=nominal_cols, drop_first=False, dtype=int)

    if scaler is not None and scale_cols:
        cols_to_scale = [c for c in scale_cols if c in df.columns]
        if cols_to_scale:
            df[cols_to_scale] = scaler.transform(df[cols_to_scale])
    return df
