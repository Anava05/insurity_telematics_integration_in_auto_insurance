# Insurity Telematics Insurance Pricing System

This repository contains the complete source code and models for a prototype of a dynamic, telematics-based auto insurance pricing system. The project demonstrates an end-to-end pipeline from data simulation to a sophisticated, dual-view interactive dashboard.

The core of the solution is a **cascading two-stage machine learning architecture** that first quantifies driving behavior and then uses that score, along with traditional actuarial factors, to predict the final premium. The system prioritizes accuracy, transparency, and user experience.

---

## 🚀 Key Features

*   **Sophisticated Two-Stage Model:** A cascading pipeline that decouples behavioral risk scoring from final premium calculation for improved accuracy and interpretability.
*   **State-of-the-Art Optimization:** Uses the **Optuna** framework to automatically find the best hyperparameters for the pricing model, ensuring peak performance.
*   **Explainable AI (XAI):** Integrates the **SHAP** library to provide full transparency into *why* the model makes its decisions, eliminating the "black box" problem.
*   **Dual-View Interactive Dashboard:** A single application built with **Streamlit** that provides a tailored user experience for both the **Customer** (simple, actionable insights) and the **Underwriter** (detailed, analytical breakdown).
*   **Modular & Reproducible:** A script-based workflow that allows anyone to reproduce the entire pipeline, from data generation to model training and application launch.

---

## 🔧 Technology Stack

*   **Language:** Python 3.10+
*   **Data Science:** Pandas, NumPy, Scikit-learn
*   **Machine Learning:** XGBoost
*   **Hyperparameter Optimization:** Optuna
*   **Explainable AI:** SHAP
*   **Web Dashboard:** Streamlit
*   **Core Libraries:** Matplotlib, Joblib

---

## ⚙️ Setup and Installation

Follow these steps carefully to set up your environment and run the project.

### Step 1: Clone the Repository

First, clone this repository to your local machine using Git.

```bash
git clone https://github.com/your-username/insurity-telematics-solution.git
cd insurity-telematics-solution
```

### Step 2: Create a Virtual Environment (Highly Recommended)

Using a virtual environment is a best practice that prevents conflicts with other Python projects on your system.

**On Windows:**
```bash
python -m venv venv
venv\Scripts\activate
```

**On macOS / Linux:**
```bash
python3 -m venv venv
source venv/bin/activate
```

After activation, you will see `(venv)` at the beginning of your terminal prompt.

### Step 3: Install Required Libraries

This project comes with a `requirements.txt` file that lists all necessary libraries with versions that are known to work together.

First, **create the `requirements.txt` file** by running this command in your terminal. This captures the exact versions of the libraries you used to build the project.
```bash
pip freeze > requirements.txt
```
*(You only need to do this once. Then, commit the file to your repository.)*

Now, install all dependencies from this file:
```bash
pip install -r requirements.txt
```

---

## ▶️ Running the Full Pipeline

The scripts are designed to be run in a specific order to build the necessary data and models before launching the application.

### Step 1: Generate Telematics Features

This script simulates the raw driving data and processes it into features for the behavioral model.

```bash
python process_data.py
```
*   **Output:** `raw_driving_data.csv` and `model_ready_features.csv`

### Step 2: Train the Behavioral Risk Model

This script trains the first model to score driver behavior and saves it.

```bash
python train_risk_model.py
```
*   **Output:** `risk_model.joblib`, plus SHAP plots for analysis.

### Step 3: Generate the Premium Dataset

This script creates the rich, complex dataset for our final pricing model.

```bash
python generate_premium_data.py
```
*   **Output:** `premium_features.csv`

### Step 4: Train and Optimize the Final Premium Model

This script uses Optuna to find the best model, trains it, and saves it. This may take a few minutes.

```bash
python train_premium_model.py
```
*   **Output:** `premium_model.joblib` and `premium_model_features.joblib`

### Step 5: Launch the Interactive Dashboard

After all previous steps are complete, run the main application. **Use the `python -m` flag for maximum reliability.**

```bash
python -m streamlit run final_dashboard.py
```

Your web browser will automatically open with the dashboard running.

---

## 🐛 Troubleshooting Common Issues

If you encounter an error, check here first.

*   **Error: `streamlit: The term 'streamlit' is not recognized...`**
    *   **Cause:** The folder containing the `streamlit.exe` command is not in your system's PATH.
    *   **Solution:** **Always** use the `python -m` command to run streamlit. This command is guaranteed to work if the library is installed in your active environment.
        ```bash
        python -m streamlit run final_dashboard.py
        ```

*   **Error: `ModuleNotFoundError: No module named 'streamlit'` (or any other library)**
    *   **Cause:** The required library is not installed in the Python environment you are currently using. This often happens if you forget to activate your virtual environment.
    *   **Solution:**
        1.  Make sure your virtual environment is active (you should see `(venv)` in your prompt).
        2.  Run the installation command again: `pip install -r requirements.txt`.

*   **Error: `FileNotFoundError: [Errno 2] No such file or directory: 'model_ready_features.csv'`**
    *   **Cause:** You have not run the scripts in the correct order. The script you are running depends on a file that has not been created yet.
    *   **Solution:** Run the scripts in the exact sequence listed in the "Running the Full Pipeline" section above.

*   **Any other strange errors related to library functions?**
    *   **Cause:** A library might have been installed with a slightly different version that has a breaking API change.
    *   **Solution:** The `requirements.txt` file is designed to prevent this. Ensure you have installed from it correctly. If you started from scratch, creating and using the `requirements.txt` file is the best way to ensure reproducibility.

---

## 📁 Project Structure

```
.
├── process_data.py             # 1. Simulates and processes telematics data.
├── train_risk_model.py         # 2. Trains the behavioral risk score model.
├── generate_premium_data.py    # 3. Simulates the rich dataset for pricing.
├── train_premium_model.py      # 4. Optimizes and trains the final pricing model.
├── final_dashboard.py          # 5. The main Streamlit application file.
├── risk_model.joblib           # Saved behavioral model.
├── premium_model.joblib        # Saved final pricing model.
├── premium_model_features.joblib # List of features for the pricing model.
├── model_ready_features.csv    # Processed telematics data.
├── premium_features.csv        # The final, rich dataset for pricing.
├── requirements.txt            # List of all Python dependencies.
└── README.md                   # This file.
```
