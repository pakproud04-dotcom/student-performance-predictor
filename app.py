"""
app.py
------
AI Student Performance Predictor - Streamlit Application

Founder: Saad Hakeem

Takes Quiz, Assignment, and Midterm marks from a student and uses a
trained Linear Regression model to predict the expected Final Marks,
Percentage, Grade, Performance Level, and Recommendation.
"""

import streamlit as st
import joblib
import os
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score

MODEL_PATH = "trained_model.pkl"
DATA_PATH = "students.csv"
MAX_COMPONENT_SCORE = 20  # Quiz / Assignment / Midterm are each out of 20


# ---------------------------------------------------------------------------
# Load trained model — if the pickle isn't present (e.g. fresh cloud deploy),
# train it on the spot from students.csv so the app always works.
# ---------------------------------------------------------------------------
@st.cache_resource
def load_model():
    if os.path.exists(MODEL_PATH):
        return joblib.load(MODEL_PATH)

    if not os.path.exists(DATA_PATH):
        return None

    df = pd.read_csv(DATA_PATH).dropna()
    for col in ["Quiz", "Assignment", "Midterm"]:
        df = df[(df[col] >= 0) & (df[col] <= MAX_COMPONENT_SCORE)]
    df = df[(df["Final"] >= 0) & (df["Final"] <= 100)].drop_duplicates()

    X = df[["Quiz", "Assignment", "Midterm"]]
    y = df["Final"]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model = LinearRegression()
    model.fit(X_train, y_train)
    mae = mean_absolute_error(y_test, model.predict(X_test))
    r2 = r2_score(y_test, model.predict(X_test))

    bundle = {"model": model, "mae": mae, "r2": r2}
    joblib.dump(bundle, MODEL_PATH)
    return bundle


model_data = load_model()


# ---------------------------------------------------------------------------
# Grading & performance logic
# ---------------------------------------------------------------------------
def get_grade(percentage: float) -> str:
    if percentage >= 85:
        return "A+"
    elif percentage >= 80:
        return "A"
    elif percentage >= 75:
        return "B+"
    elif percentage >= 70:
        return "B"
    elif percentage >= 65:
        return "C+"
    elif percentage >= 60:
        return "C"
    elif percentage >= 55:
        return "D+"
    elif percentage >= 50:
        return "D"
    else:
        return "F"


def get_performance(percentage: float) -> str:
    if percentage >= 85:
        return "Excellent"
    elif percentage >= 80:
        return "Very Good"
    elif percentage >= 70:
        return "Good"
    elif percentage >= 60:
        return "Average"
    elif percentage >= 50:
        return "Needs Improvement"
    else:
        return "At Risk"


def get_recommendation(grade: str) -> str:
    recommendations = {
        "A+": "Excellent performance. Keep maintaining your current academic effort.",
        "A": "Very good performance. Continue your current preparation strategy.",
        "B+": "Good performance. Additional preparation can help improve the final result.",
        "B": "Good performance. Additional preparation can help improve the final result.",
        "C+": "Average performance. Focus more on weak academic areas.",
        "C": "Average performance. Focus more on weak academic areas.",
        "D+": "Performance needs improvement. Increase preparation and revision.",
        "D": "Performance needs improvement. Increase preparation and revision.",
        "F": "The student is at risk. Significant improvement is recommended.",
    }
    return recommendations.get(grade, "")


# ---------------------------------------------------------------------------
# Page setup
# ---------------------------------------------------------------------------
st.set_page_config(page_title="AI Student Performance Predictor", page_icon="🎓", layout="centered")

st.title("🎓 AI Student Performance Predictor")
st.caption("Founder: Saad Hakeem  |  Supervised Machine Learning — Linear Regression")

if model_data is None:
    st.error(
        "Trained model not found. Please run `python train_model.py` first "
        "to generate `trained_model.pkl`."
    )
    st.stop()

model = model_data["model"]
mae = model_data["mae"]
r2 = model_data["r2"]

st.divider()

# ---------------------------------------------------------------------------
# Input form
# ---------------------------------------------------------------------------
st.subheader("Enter Student Details")

student_name = st.text_input("Student Name", placeholder="e.g. Saad Hakeem")

col1, col2, col3 = st.columns(3)
with col1:
    quiz = st.number_input("Quiz Marks (0-20)", min_value=0, max_value=100, value=0, step=1)
with col2:
    assignment = st.number_input("Assignment Marks (0-20)", min_value=0, max_value=100, value=0, step=1)
with col3:
    midterm = st.number_input("Midterm Marks (0-20)", min_value=0, max_value=100, value=0, step=1)

predict_clicked = st.button("🔮 PREDICT PERFORMANCE", type="primary", use_container_width=True)

st.divider()

# ---------------------------------------------------------------------------
# Validation + Prediction
# ---------------------------------------------------------------------------
if predict_clicked:
    errors = []

    if not student_name.strip():
        errors.append("Please enter the student's name.")

    if not (0 <= quiz <= MAX_COMPONENT_SCORE):
        errors.append(f"Invalid Quiz Marks. Please enter a value between 0 and {MAX_COMPONENT_SCORE}.")

    if not (0 <= assignment <= MAX_COMPONENT_SCORE):
        errors.append(f"Invalid Assignment Marks. Please enter a value between 0 and {MAX_COMPONENT_SCORE}.")

    if not (0 <= midterm <= MAX_COMPONENT_SCORE):
        errors.append(f"Invalid Midterm Marks. Please enter a value between 0 and {MAX_COMPONENT_SCORE}.")

    if errors:
        for e in errors:
            st.error(e)
    else:
        # Predict
        prediction = model.predict([[quiz, assignment, midterm]])[0]
        predicted_final = max(0, min(100, round(prediction)))
        percentage = predicted_final  # Final is already out of 100

        grade = get_grade(percentage)
        performance = get_performance(percentage)
        recommendation = get_recommendation(grade)

        st.success("Prediction Complete")

        st.markdown(f"### Results for **{student_name}**")

        r1, r2_col = st.columns(2)
        with r1:
            st.metric("Predicted Final Marks", f"{predicted_final}")
        with r2_col:
            st.metric("Predicted Percentage", f"{percentage}%")

        r3, r4 = st.columns(2)
        with r3:
            st.metric("Grade", grade)
        with r4:
            st.metric("Performance", performance)

        st.info(f"**Recommendation:** {recommendation}")

        st.caption(
            "⚠️ Note: This is an estimated prediction based on historical patterns, "
            "not a guaranteed final result."
        )

# ---------------------------------------------------------------------------
# Sidebar: model info
# ---------------------------------------------------------------------------
with st.sidebar:
    st.header("📊 Model Performance")
    st.write(f"**Algorithm:** Linear Regression")
    st.write(f"**MAE:** {mae:.2f}")
    st.write(f"**R² Score:** {r2:.3f}")
    st.divider()
    st.write("**Project:** AI Student Performance Predictor")
    st.write("**Founder:** Saad Hakeem")
    st.write("**ML Type:** Supervised Learning (Regression)")
