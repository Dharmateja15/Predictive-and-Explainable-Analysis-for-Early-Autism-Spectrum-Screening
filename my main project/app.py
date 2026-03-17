# app.py
import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os

st.set_page_config(page_title="ASD Screening — ML", layout="centered")
st.title("Autism Screening — Predict")
st.write("Provide questionnaire responses and demographics. This is a research demo, not a clinical tool.")

# -------------------------
# Paths for artifacts
# -------------------------
MODEL_PATH = "asd_best_model.pkl"
ENC_PATH = "label_encoders.pkl"
FEAT_PATH = "feature_columns.json"

# Load model and artifacts
if not os.path.exists(MODEL_PATH):
    st.error(f"Model file not found: {MODEL_PATH}. Place your trained model file here.")
    st.stop()

model = joblib.load(MODEL_PATH)

label_encoders = joblib.load(ENC_PATH) if os.path.exists(ENC_PATH) else {}
feature_columns = None
if os.path.exists(FEAT_PATH):
    import json
    with open(FEAT_PATH, "r") as f:
        feature_columns = json.load(f)

st.success("Artifacts loaded. Model ready.")

# -------------------------
# Helper functions
# -------------------------
def safe_encode(col, val):
    """Encode categorical value using saved LabelEncoder; fallback for unseen values."""
    if col in label_encoders:
        le = label_encoders[col]
        try:
            return int(le.transform([str(val)])[0])
        except Exception:
            return 0
    else:
        try:
            return float(val)
        except:
            return 0

def align_columns(df, model_cols):
    """Ensure df has columns in same order as model expects; fill missing with 0."""
    out = pd.DataFrame(columns=model_cols)
    for c in model_cols:
        out[c] = df[c] if c in df.columns else 0
    return out.astype(float)

# -------------------------
# Input form
# -------------------------
st.subheader("AQ-10 Questionnaire (enter 0 or 1 for each)")
cols = st.columns(5)
answers = {}
for i in range(1, 11):
    key = f"A{i}_Score"
    answers[key] = cols[(i-1) % 5].number_input(key, min_value=0, max_value=1, value=0, step=1)

st.markdown("---")
st.subheader("Demographics / other fields")
age = st.number_input("Age (years)", min_value=1, max_value=120, value=25)
gender = st.selectbox("Gender", ["f", "m", "o", "other"], index=0)
ethnicity = st.text_input("Ethnicity", value="White-European")
jaundice = st.selectbox("Jaundice at birth?", ["no", "yes"], index=0)
autism_family = st.selectbox("Immediate family diagnosed with autism?", ["no", "yes"], index=0)
country_of_res = st.text_input("Country of residence", value="United States")
used_app_before = st.selectbox("Used screening app before?", ["no", "yes"], index=0)
relation = st.selectbox("Relation who completed the test", ["Self", "Parent", "Relative", "Health care professional", "?"], index=0)

# computed sum of AQ-10
result = sum(answers.values())
st.markdown(f"**Computed screening result (sum A1..A10):** {result}")
st.markdown("---")

# -------------------------
# Build input dataframe
# -------------------------
default_order = [
    "A1_Score","A2_Score","A3_Score","A4_Score","A5_Score","A6_Score","A7_Score","A8_Score","A9_Score","A10_Score",
    "age","gender","ethnicity","jaundice","autism","country_of_res","used_app_before","result","age_desc","relation"
]

columns_to_use = feature_columns if feature_columns else default_order

row = {}
for col in columns_to_use:
    if col in answers:
        row[col] = answers[col]
    elif col == "age":
        row[col] = age
    elif col == "gender":
        row[col] = gender
    elif col == "ethnicity":
        row[col] = ethnicity
    elif col == "jaundice":
        row[col] = jaundice
    elif col == "autism":
        row[col] = autism_family
    elif col == "country_of_res":
        row[col] = country_of_res
    elif col == "used_app_before":
        row[col] = used_app_before
    elif col == "result":
        row[col] = result
    elif col == "age_desc":
        if age >= 18:
            row[col] = "18 and more"
        elif age >= 12:
            row[col] = "12 to 17"
        else:
            row[col] = "4 to 11"
    elif col == "relation":
        row[col] = relation
    else:
        row[col] = 0

X_single = pd.DataFrame([row], columns=columns_to_use)

st.write("Prepared input preview:")
st.dataframe(X_single.T)

# -------------------------
# Encode categoricals
# -------------------------
X_proc = X_single.copy()
for c in X_proc.columns:
    X_proc[c] = X_proc[c].apply(lambda v: safe_encode(c, v))

# Align columns
model_cols = model.feature_names_in_ if hasattr(model, "feature_names_in_") else X_proc.columns
X_final = align_columns(X_proc, model_cols)

# -------------------------
# Predict
# -------------------------
if st.button("Predict ASD probability"):
    proba = model.predict_proba(X_final)[:,1][0]
    pred = int(proba >= 0.5)
    label = "ASD (1)" if pred==1 else "No ASD (0)"
    
    st.subheader("Prediction")
    st.write(f"Predicted class: **{label}**")
    st.write(f"Predicted probability of ASD: **{proba:.3f}**")
    
    # Save to CSV
    out_row = X_final.copy()
    out_row["predicted_prob"] = proba
    out_row["predicted_class"] = pred
    if os.path.exists("single_predictions.csv"):
        prev = pd.read_csv("single_predictions.csv")
        new = pd.concat([prev, out_row.reset_index(drop=True)], ignore_index=True)
        new.to_csv("single_predictions.csv", index=False)
    else:
        out_row.to_csv("single_predictions.csv", index=False)
    st.success("Prediction saved to single_predictions.csv")
