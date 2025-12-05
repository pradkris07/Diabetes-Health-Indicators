# 🚀 MLOps Project – Diabetes Health Indicators Predictor

End-to-End ML Pipeline with DVC, MLflow, Dagshub, Cookiecutter, Flask & uv

Welcome to the **Diabetes Health Indicators MLOps Project**!
This repository demonstrates how to build a complete production-grade ML system — covering project templating, experiment tracking, data versioning, model training, orchestration, and web deployment.

This project uses **Cookiecutter**, **DVC**, **Dagshub MLflow**, **RandomizedSearchCV**, **feature engineering**, and a simple **Flask web app** for predictions.

---

## 🏗️ Project Setup & Structure

### **Step 1: Create Repository**

* Created a new GitHub repository
* Cloned it locally
* Opened in VS Code:

```bash
code .
```

### **Step 2: Initialize uv Environment**

```bash
uv init
```

### **Step 3: Install Dependencies**

Installed important libraries:

```bash
uv add cookiecutter scikit-learn dagshub mlflow
```

Additional libraries added as needed during development.

### **Step 4: Add Local Packages**

```bash
uv pip install -e .
```

### **Step 5: Generate Project Template**

Used the standard Cookiecutter Data Science template:

```bash
cookiecutter -c v1 https://github.com/drivendata/cookiecutter-data-science
```

### **Step 6: Adjust Directory Structure**

Renamed:

```
src/models → src/model
```

### **Step 7: Commit & Push**

```bash
git add .
git commit -m "Initial project setup"
git push
```

---

## 📊 MLflow Tracking Setup on Dagshub

### **Step 8–10: Configure Dagshub Repository**

1. Visit: [https://dagshub.com/dashboard](https://dagshub.com/dashboard)
2. Create → New Repo
3. Connect to GitHub Repo
4. Copy MLflow experiment tracking URL
5. Test in browser: MLflow UI
6. Project MLflow URL:

```
https://dagshub.com/pradkris07/Diabetes-Health-Indicators.mlflow
```

### **Step 11: Add Dagshub & MLflow**

```bash
uv add dagshub mlflow
```

### **Step 12: Explore the Dataset**

Created a notebook to:

* View dataset structure
* Try various ML algorithms
* Compare model performances

### **Step 13: Commit Work**

```bash
git add .
git commit -m "MLflow setup and data exploration"
git push
```

---

## 📦 Data Versioning with DVC

### **Step 14: Initialize DVC**

```bash
dvc init
```

### **Step 15–16: Configure Local Storage**

Created a temporary local S3-like directory:

```
local_s3/
```

Configured DVC remote:

```bash
dvc remote add -d mylocal local_s3
```

---

## 🧩 Building the ML Pipeline (Inside src/ Directory)

### Added the following modules:

---

### **🔎 logger/**

Utility for logging, error tracing, and debugging modules.

---

### **📥 data_ingestion.py**

* Downloads dataset from GitHub
* Extracts sample records for frontend
* Splits full dataset into train & test files

---

### **✔️ data_validation.py**

* Validates column names
* Verifies column types
* Checks dataset shape consistency

---

### **🔧 data_transformation.py**

Performs:

* Scaling
* Outlier handling
* Balancing with **SMOTEENN**

---

### **🤖 model_trainer.py**

* Trains multiple ML models
* Uses **RandomizedSearchCV** for hyperparameter tuning
* Saves performance metrics
* Updates `final.yaml` with best parameters

---

### **📌 register_model.py**

* Registers trained model in **Dagshub MLflow**
* Logs metrics to **DVC**

---

### **📄 model.yaml**

Contains parameters and configurations for all ML models.

---

### **⚙️ dvc.yaml**

Full pipeline definition for automated orchestration.

---

## ▶️ Running the Pipeline

### **Step 18: Reproduce DVC Pipeline**

```bash
dvc repro
```

### **Step 19: Check Status**

```bash
dvc status
```

### **Step 20: Commit All**

```bash
git add .
git commit -m "Completed DVC pipeline setup"
git push
```

---

## 🌐 Building the Web App (Flask)

### **Step 21: Create Flask App Directory**

```
flask_app/
```

Added:

* `app.py`
* Templates

### **Step 22: Install Flask**

```bash
uv add flask
```

### **Step 23: Run App Locally**

Users can now enter values to test predictions from the sample.csv file.

---

## 🎯 Final Project Workflow Summary

```
Project Template Setup
      ➜ MLflow Integration
          ➜ DVC Data Versioning
              ➜ Data Ingestion
                  ➜ Data Validation
                      ➜ Data Transformation
                          ➜ Model Training
                              ➜ Model Registration
                                  ➜ Flask App for Prediction
```

---

## 🛠️ Technologies Used

* Python
* uv environment manager
* Scikit-learn
* Dagshub MLflow
* DVC
* Cookiecutter Data Science
* Pandas, NumPy
* SMOTEENN
* RandomizedSearchCV
* Flask
* GitHub

---

## 💬 Connect

If you want improvements or enhancements, feel free to open an issue or reach out anytime!

---

If you'd like, I can **add badges**, **add architecture diagrams**, or **create a markdown table for pipeline components**.
