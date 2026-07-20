# 📦 Logistics AI Concierge & Optimization

> An end-to-end machine learning and interactive application designed to predict delivery delays, optimize logistics routing, and provide real-time operational insights.

---

## 🚀 About the Project
**Logistics AI Concierge** is an intelligent decision-support system built for modern logistics management. Powered by a Deep Learning/Machine Learning backend and an interactive Streamlit user interface, this application helps logistics teams proactively identify delivery risks, evaluate partner performance, and analyze operational bottlenecks in real-time.

---

## 🛠️ Key Features
* **🤖 Real-Time AI Prediction Chatbot:** An interactive step-by-step assistant that collects delivery parameters (distance, weight, logistics partner) and predicts delay probabilities instantly.
* **📊 Analytics Dashboard:** Comprehensive data visualisations tracking logistics partner performance, delivery costs, and delay trends.
* **🧹 Robust Data Preprocessing Pipeline:** Automated data cleaning, handling of missing values, and strict feature engineering to ensure robust model performance.
* **⚙️ Modular Architecture:** Clean separation of concerns between data preprocessing (`preprocessing.py`), model inference (`inference.py`), and the frontend interface (`main.py`).

---

## 📂 Project Structure
```text
Delivery_Logistics/
│
├── app/
│   ├── main.py            # Streamlit UI & Application entry point
│   ├── dashboard.py       # Analytics dashboard component
│   └── inference.py       # Inference pipeline & risk decision logic
│
├── src/
│   └── preprocessing.py   # Data cleaning, scaling, and feature encoding pipeline
│
├── data/
│   └── processed/         # Cleaned datasets used for training and dashboard
│
├── models/
│   ├── logistics_model.h5 # Trained Deep Learning model
│   └── preprocessor.pkl   # Fitted Scikit-Learn ColumnTransformer pipeline
│
├── requirements.txt       # Project dependencies
└── README.md              # Project documentation

📑 Detailed System Documentation
1. Data Preprocessing & Pipeline (src/preprocessing.py)
Single Source of Truth: Centralizes all feature definitions, separating them strictly into numeric columns (distance_km, package_weight_kg, delivery_time_hours, expected_time_hours, delivery_rating, delivery_cost) and categorical columns (delivery_partner, package_type, vehicle_type, delivery_mode, region, weather_condition, delayed, delivery_status).

Automated Data Cleansing: Forces numeric conversions to handle irregularities, standardizes categorical values, and handles missing values to prevent training bottlenecks.

Feature Transformation: Utilizes Scikit-Learn's StandardScaler for numeric features and OneHotEncoder (with handle_unknown='ignore') wrapped inside a ColumnTransformer, which is saved as preprocessor.pkl.

2. Inference Engine (app/inference.py)
Proactive Prediction Pipeline: Automatically transforms user inputs via the saved preprocessor pipeline, converts data types into float32 for TensorFlow compatibility, and performs model inference (logistics_model.h5).

Decision Thresholding: Evaluates delay probabilities using an operational decision threshold (e.g., 0.4) to classify potential delays and generate actionable advice for supply chain management.

Error Handling & Debugging: Equipped with a robust try-except block and traceback logging to capture and diagnose any matrix shape or feature mismatch issues seamlessly.

3. Interactive User Interface (app/main.py & app/dashboard.py)
Step-by-Step Chat Interface (Real-time Prediction): Guides the user through a conversational input flow to capture core delivery parameters (distance, package weight, and logistics partner selection) before dispatching data to the model.

Analytics Dashboard: Provides visual breakdowns of average delay rates per logistics partner, cost distributions, and filterable data tables for deep exploratory data analysis (EDA).

⚙️ Installation & Local Setup
Follow these steps to run the project locally on your machine:

1. Clone the Repository
Bash
git clone [https://github.com/shalehh17/Delivery_Logistics.git](https://github.com/shalehh17/Delivery_Logistics.git)
cd Delivery_Logistics
2. Create and Activate Virtual Environment
Bash
python -m venv venv
# On Windows:
venv\Scripts\activate
# On macOS/Linux:
# source venv/bin/activate
3. Install Dependencies
Bash
pip install -r requirements.txt
4. Run the Preprocessing Pipeline (If data is modified)
Bash
python src/preprocessing.py
5. Launch the Streamlit Application
Bash
streamlit run app/main.py


📊 Tech Stack & Libraries
-Language: Python
-Frontend / UI: Streamlit
-Machine Learning / Preprocessing: Scikit-Learn, Pandas, NumPy, Joblib
-Deep Learning: TensorFlow / Keras
-Version Control: Git & GitHub

💡 How to Use the Application
1.Open the local Streamlit server (typically http://localhost:8501).
2.Use the sidebar to switch between Real-time Prediction and Analytics Dashboard.
3.Real-time Prediction Flow:
  A.Input the delivery distance in kilometers when prompted.
  B.Input the package weight in kilograms.
  C.Select your preferred logistics partner from the dropdown.
  CD.lick Analisis to view the delay risk percentage and operational advice.
4.Analytics Dashboard Flow:
   A. Explore interactive charts visualizing delivery partner performance, metrics summaries, and detailed underlying logs.


