"""
====================================================
  Job Salary Prediction - Machine Learning Model
====================================================
"""
# AUTO-INSTALL MISSING PACKAGES
import subprocess
import sys
import joblib

REQUIRED = ["pandas", "numpy", "scikit-learn", "matplotlib"]

print("Checking required packages...")

for package in REQUIRED:
    try:
        __import__("sklearn" if package == "scikit-learn" else package)
        print(f"  [OK] {package}")

    except ImportError:
        print(f"  [INSTALLING] {package} ...")

        subprocess.check_call(
            [sys.executable, "-m", "pip", "install", package],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )

        print(f"  [DONE] {package}")

print()

# IMPORTS
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.ensemble import GradientBoostingRegressor
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split

from sklearn.metrics import (
    r2_score,
    mean_absolute_error,
    mean_squared_error,
    mean_absolute_percentage_error
)

# CONFIGURATION
DATASET_PATH = "data.csv"

CATEGORICAL_COLS = [
    "job_title",
    "education_level",
    "industry",
    "company_size",
    "location",
    "remote_work"
]

NUMERICAL_COLS = [
    "experience_years",
    "skills_count",
    "certifications"
]

TARGET_COL = "salary"

FEATURE_ORDER = [
    "job_title",
    "experience_years",
    "education_level",
    "skills_count",
    "industry",
    "company_size",
    "location",
    "remote_work",
    "certifications"
]

# DISPLAY FUNCTIONS
def separator(char="-", width=60):
    print(char * width)

def section(title):
    print()
    separator("=")
    print(f"  {title}")
    separator("=")

# USER INPUT FUNCTIONS
def prompt_choice(label, options):

    print(f"\n  {label}:")

    for i, opt in enumerate(options, 1):
        print(f"    [{i:2d}] {opt}")

    while True:
        raw = input(f"  Enter number (1-{len(options)}): ").strip()

        if raw.isdigit():
            idx = int(raw) - 1

            if 0 <= idx < len(options):
                return options[idx]

        print(f"  Please enter a number between 1 and {len(options)}.")

def prompt_int(label, lo, hi):

    while True:
        raw = input(f"  {label} ({lo}-{hi}): ").strip()

        if raw.isdigit():
            val = int(raw)

            if lo <= val <= hi:
                return val

        print(f"  Please enter a whole number between {lo} and {hi}.")

# LOAD DATASET
def load_data(path):

    section("STEP 1 - Loading Dataset")

    if not os.path.exists(path):

        print(f"\n  ERROR: File not found -> {path}")
        print("  Make sure the CSV is in the SAME folder as this script.")

        input("\n  Press Enter to exit...")
        sys.exit(1)

    df = pd.read_csv(path)

    # HANDLE MISSING VALUES
    for col in CATEGORICAL_COLS:
        df[col] = df[col].fillna("Unknown")

    for col in NUMERICAL_COLS:
        df[col] = df[col].fillna(df[col].median())

    print(f"\n  Rows            : {len(df):,}")
    print(f"  Columns         : {len(df.columns)}")
    print(f"  Missing Values  : {df.isnull().sum().sum()} total")

    print(f"  Salary Range    : ${df[TARGET_COL].min():,.0f} to ${df[TARGET_COL].max():,.0f}")
    print(f"  Salary Mean     : ${df[TARGET_COL].mean():,.0f}")
    print(f"  Salary Median   : ${df[TARGET_COL].median():,.0f}")

    return df

# PREPROCESSING
def preprocess(df):

    section("STEP 2 - Preprocessing")

    df_enc = df.copy()
    encoders = {}

    print("\n  Encoding categorical columns:")

    for col in CATEGORICAL_COLS:

        le = LabelEncoder()

        df_enc[col] = le.fit_transform(df[col].astype(str))

        encoders[col] = le

        print(f"    {col:<22} -> {len(le.classes_)} classes")

    X = df_enc[FEATURE_ORDER]
    y = df_enc[TARGET_COL]

    return X, y, encoders

# TRAIN TEST SPLIT
def split_data(X, y):

    section("STEP 3 - Train / Test Split (80 / 20)")

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=42
    )

    print(f"\n  Training Samples : {len(X_train):,}")
    print(f"  Test Samples     : {len(X_test):,}")

    return X_train, X_test, y_train, y_test


# TRAIN MODEL
def train_model(X_train, y_train):

    section("STEP 4 - Training Model")

    model = GradientBoostingRegressor(
        n_estimators=300,
        learning_rate=0.05,
        max_depth=6,
        min_samples_leaf=5,
        subsample=0.8,
        random_state=42
    )

    print("\n Training model...")
    model.fit(X_train, y_train)

    print(" Training Complete!")
    return model

# MODEL EVALUATION + VISUALIZATION
def evaluate_model(model, X_train, X_test, y_train, y_test):

    section("STEP 5 - Model Evaluation")

    y_pred_test = model.predict(X_test)
    y_pred_train = model.predict(X_train)

    r2_test = r2_score(y_test, y_pred_test)
    r2_train = r2_score(y_train, y_pred_train)

    mae = mean_absolute_error(y_test, y_pred_test)

    rmse = np.sqrt(mean_squared_error(y_test, y_pred_test))

    mape = mean_absolute_percentage_error(y_test, y_pred_test) * 100

    print(f"\n  {'Metric':<35} {'Train':>10} {'Test':>10}")

    separator("-", 60)

    print(f"  {'R2 Score':<35} {r2_train:>10.4f} {r2_test:>10.4f}")

    print(f"  {'Mean Absolute Error ($)':<35} {'':>10} {mae:>10,.0f}")

    print(f"  {'Root Mean Squared Error ($)':<35} {'':>10} {rmse:>10,.0f}")

    print(f"  {'Mean Abs Percentage Error (%)':<35} {'':>10} {mape:>9.2f}%")

    # FEATURE IMPORTANCE
    print("\n  Feature Importances:")

    separator("-", 60)

    fi = pd.Series(
        model.feature_importances_,
        index=FEATURE_ORDER
    ).sort_values(ascending=False)

    for feat, imp in fi.items():

        bar = "#" * int(imp * 40 / fi.max())

        print(f"    {feat:<22} {imp:.4f}  {bar}")

    # FINAL PERFORMANCE SUMMARY
    print()

    separator("=")

    print("  FINAL MODEL PERFORMANCE SUMMARY")

    separator("=")

    print(f"\n  R2 Score                : {r2_test:.4f}")
    print(f"  Mean Absolute Error     : ${mae:,.0f}")
    print(f"  Root Mean Squared Error : ${rmse:,.0f}")
    print(f"  Mean Percentage Error   : {mape:.2f}%")

    print("\n  Performance Interpretation:")

    if r2_test >= 0.90:
        print("   Excellent prediction accuracy")

    elif r2_test >= 0.75:
        print("   Good prediction accuracy")

    elif r2_test >= 0.60:
        print("   Moderate prediction accuracy")

    else:
        print("   Low prediction accuracy")

    separator("=")

    # =====================================================
    # VISUALIZATION SECTION
    # =====================================================

    section("STEP 5B - Evaluation Graphs")

    # GRAPH 1 - R2 SCORES
    plt.figure(figsize=(6, 5))

    plt.bar(
        ["Train R2", "Test R2"],
        [r2_train, r2_test]
    )

    plt.ylabel("R2 Score")
    plt.title("Train vs Test R2 Score")
    plt.ylim(0, 1)

    plt.show()

    # GRAPH 2 - ERROR METRICS
    plt.figure(figsize=(8, 5))

    metrics_names = ["MAE", "RMSE", "MAPE"]
    metric_values = [mae, rmse, mape]

    plt.bar(metrics_names, metric_values)

    plt.title("Error Metrics")
    plt.ylabel("Metric Value")

    plt.show()

    # GRAPH 3 - FEATURE IMPORTANCE
    plt.figure(figsize=(10, 6))

    fi.sort_values().plot(kind="barh")

    plt.title("Feature Importance")
    plt.xlabel("Importance Score")

    plt.tight_layout()

    plt.show()

    # GRAPH 4 - ACTUAL VS PREDICTED
    plt.figure(figsize=(7, 6))

    plt.scatter(y_test, y_pred_test)

    plt.xlabel("Actual Salary")
    plt.ylabel("Predicted Salary")

    plt.title("Actual vs Predicted Salary")

    plt.plot(
        [y_test.min(), y_test.max()],
        [y_test.min(), y_test.max()]
    )

    plt.tight_layout()

    plt.show()

# USER INPUT
def get_user_input(encoders):

    section("STEP 6 - Predict Your Salary")

    print("\n  Answer each question:\n")

    job_title = prompt_choice(
        "Job Title",
        sorted(encoders["job_title"].classes_)
    )

    education = prompt_choice(
        "Education Level",
        list(encoders["education_level"].classes_)
    )

    industry = prompt_choice(
        "Industry",
        sorted(encoders["industry"].classes_)
    )

    company_size = prompt_choice(
        "Company Size",
        list(encoders["company_size"].classes_)
    )

    location = prompt_choice(
        "Location",
        sorted(encoders["location"].classes_)
    )

    remote_work = prompt_choice(
        "Remote Work",
        list(encoders["remote_work"].classes_)
    )

    experience_yrs = prompt_int(
        "Years of Experience",
        0,
        40
    )

    skills_count = prompt_int(
        "Number of Skills",
        1,
        20
    )

    certifications = prompt_int(
        "Number of Certifications",
        0,
        5
    )

    row = {
        "job_title":
            encoders["job_title"].transform([job_title])[0],

        "experience_years":
            experience_yrs,

        "education_level":
            encoders["education_level"].transform([education])[0],

        "skills_count":
            skills_count,

        "industry":
            encoders["industry"].transform([industry])[0],

        "company_size":
            encoders["company_size"].transform([company_size])[0],

        "location":
            encoders["location"].transform([location])[0],

        "remote_work":
            encoders["remote_work"].transform([remote_work])[0],

        "certifications":
            certifications,
    }

    inputs_display = {
        "Job Title": job_title,
        "Experience Years": experience_yrs,
        "Education Level": education,
        "Skills Count": skills_count,
        "Industry": industry,
        "Company Size": company_size,
        "Location": location,
        "Remote Work": remote_work,
        "Certifications": certifications
    }

    return pd.DataFrame([row])[FEATURE_ORDER], inputs_display

# SHOW PREDICTION
def show_prediction(model, X_input, inputs_display):

    predicted_salary = model.predict(X_input)[0]

    # =====================================================
# CURRENCY CONVERSION
# =====================================================

currency = {
    "USA": ("$", 1),
    "India": ("₹", 83),
    "UK": ("£", 0.79),
    "Canada": ("C$", 1.36),
    "Germany": ("€", 0.92),
    "Australia": ("A$", 1.52),
    "Singapore": ("S$", 1.35),
    "UAE": ("AED", 3.67)
}

selected_country = inputs_display["Location"]

symbol = "$"
rate = 1

if selected_country in currency:
    symbol, rate = currency[selected_country]

converted_salary = predicted_salary * rate

lower = max(0, predicted_salary - 4255)
upper = predicted_salary + 4255

section("PREDICTION RESULT")

print("\n  Your Profile:")

separator("-", 45)

for k, v in inputs_display.items():
    print(f"  {k:<22} : {v}")

separator("-", 45)

print(f"\n  Predicted Salary : {symbol} {converted_salary:,.0f}")

print(f"\n  Confidence Range : {symbol} {lower * rate:,.0f} to {symbol} {upper * rate:,.0f}")

separator("-", 45)

print()
# PREDICTION LOOP
def prediction_loop(model, encoders):

    while True:

        X_input, inputs_display = get_user_input(encoders)

        show_prediction(model, X_input, inputs_display)

        print("  Would you like another prediction?")

        again = input(
            "  Type 'yes' to continue or press Enter to exit: "
        ).strip().lower()

        if again not in ("yes", "y"):

            print("\n  Thank you for using the Salary Prediction Model.\n")

            break

# Main
def main():

    df = load_data(DATASET_PATH)

    X, y, encoders = preprocess(df)

    X_train, X_test, y_train, y_test = split_data(X, y)

    model = train_model(X_train, y_train)

    # Save Model
    import joblib

    joblib.dump(model, "salary_model.pkl")
    joblib.dump(encoders, "encoders.pkl")

    print("Model Saved Successfully!")

    evaluate_model(
        model,
        X_train,
        X_test,
        y_train,
        y_test
    )

    prediction_loop(model, encoders)


# RUN PROGRAM
if __name__ == "__main__":
    main()
