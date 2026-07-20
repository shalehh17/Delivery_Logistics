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

⚙️ Installation & Setup
Follow these steps to run the project locally on your machine:

1. Clone the Repository
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
4. Run the Preprocessing Pipeline (Optional/If data updates)
Bash
python src/preprocessing.py
5. Launch the Streamlit Application
Bash
streamlit run app/main.py
📊 Tech Stack & Libraries
Language: Python

Frontend/UI: Streamlit

Machine Learning / Preprocessing: Scikit-Learn, Pandas, NumPy, Joblib

Deep Learning: TensorFlow / Keras

Version Control: Git & GitHub

💡 Usage Preview
1.Open the Streamlit local server (usually http://localhost:8501).
2.Navigate to Real-time Prediction via the sidebar.
3.Input the required parameters step-by-step (Delivery distance, package weight, and logistics partner).
4.Click Analisis to get real-time risk assessment and operational advice.
5.Switch to Analytics Dashboard to explore historical performance data.
