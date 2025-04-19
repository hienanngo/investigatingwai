
# features.py

def compute_disparities(cleaned, faculty_race_cols, student_race_cols):
    cleaned["faculty_total"] = cleaned["Grand total"]

    for race in faculty_race_cols:
        cleaned[f"{race}_faculty_pct"] = (cleaned[faculty_race_cols[race]] / cleaned["faculty_total"]) * 100
        cleaned[f"{race}_student_pct"] = cleaned[student_race_cols[race]]
        cleaned[f"{race}_disparity"] = cleaned[f"{race}_faculty_pct"] - cleaned[f"{race}_student_pct"]

    return cleaned

def add_disparity_index(df, faculty_race_cols):
    df["disparity_index"] = df[[f"{race}_disparity" for race in faculty_race_cols]].abs().mean(axis=1)
    return df
