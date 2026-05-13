# 🎓 Student Performance Predictor

A machine learning web application that predicts a student's **math score** based on demographic and academic factors. Built with Python, Flask, and CatBoost — deployed on Render.

## 🌐 Live Demo
👉 [https://student-performance-mlproject-iadv.onrender.com/predict](https://student-performance-mlproject-iadv.onrender.com/predict)

---

## 📌 Features

- Predicts student math scores based on input features
- Clean web interface built with HTML/CSS
- End-to-end ML pipeline (data ingestion → transformation → training → prediction)
- Hyperparameter tuning with multiple models
- Deployed on Render with Gunicorn

---

## 🧠 Input Features

| Feature | Description |
|---|---|
| Gender | Student's gender |
| Race/Ethnicity | Ethnic group of the student |
| Parental Level of Education | Highest education level of parents |
| Lunch | Standard or free/reduced lunch |
| Test Preparation Course | Completed or none |
| Reading Score | Score in reading (0–100) |
| Writing Score | Score in writing (0–100) |

---

## 🗂️ Project Structure
student_performance_mlproject/
│
├── artifacts/                  # Saved model and preprocessor
│   ├── preprocessor.pkl
│   ├── trained_model.pkl
│   ├── raw.csv
│   ├── train.csv
│   └── test.csv
│
├── notebook/                   # EDA and model training notebooks
│
├── src/
│   ├── components/             # Data ingestion, transformation, training
│   ├── pipeline/               # Train and predict pipelines
│   ├── exception.py            # Custom exception handling
│   ├── logger.py               # Logging setup
│   └── utils.py                # Utility functions
│
├── templates/
│   ├── index.html
│   └── home.html               # Prediction form
│
├── app.py                      # Flask app entry point
├── application.py
├── requirements.txt
├── setup.py
└── render.yaml                 # Render deployment config

---

## ⚙️ Tech Stack

- **Language:** Python
- **Framework:** Flask
- **ML Models:** CatBoost, and others (with hyperparameter tuning)
- **Preprocessing:** Scikit-learn (StandardScaler, pipelines)
- **Deployment:** Render + Gunicorn

---

## 🚀 Run Locally

```bash
# Clone the repo
git clone https://github.com/ashwinnm13/student_performance_mlproject.git
cd student_performance_mlproject

# Create virtual environment
python -m venv venv
venv\Scripts\activate  # Windows
# source venv/bin/activate  # Mac/Linux

# Install dependencies
pip install -r requirements.txt

# Run the app
python app.py
```

Visit `http://localhost:5000/predict` in your browser.

---

## 📊 ML Pipeline

1. **Data Ingestion** — Reads raw data and splits into train/test
2. **Data Transformation** — Handles missing values, encodes categoricals, scales numerics
3. **Model Training** — Trains multiple models with hyperparameter tuning, saves best model
4. **Prediction Pipeline** — Loads saved artifacts and predicts on new input

---

## 👨‍💻 Author

**Ashwin NM**
- GitHub: [@ashwinnm13](https://github.com/ashwinnm13)