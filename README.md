📊 MetroPT-3 Dataset — Predictive Maintenance Project

This project uses the MetroPT-3 Dataset from Kaggle — a real world dataset generated from metro train sensors. The data helps you build predictive maintenance and anomaly detection models.

Dataset link (Kaggle):
https://www.kaggle.com/datasets/joebeachcapital/metropt-3-dataset
 
Kaggle
+1

🧾 Dataset Description

MetroPT-3 is a dataset collected from an urban metro train’s Air Production Unit (APU). It contains sensor readings from the train that can help with predictive maintenance, failure prediction, and anomaly detection. 
Kaggle

The main features include:

Pressure readings

Temperature readings

Motor current

Digital control signals

GPS information (latitude, longitude, speed)

The dataset is typically used to train machine learning models that:

Predict failures before they happen

Detect abnormal system behavior

Analyze continuous sensor data over time 
Kaggle

🚀 Project Structure
.
├── data/                      # Dataset CSV files
│   └── metropt3.csv
├── notebooks/
│   └── train_mlflow.ipynb     # MLflow model training notebook
├── src/                       # (Optional) Python scripts
├── README.md
├── .gitignore
└── requirements.txt

📥 Download the Dataset
🧑‍💻 Using Kaggle API

Install the Kaggle API:

pip install kaggle


Configure API token (kaggle.json) in your home folder.

Download dataset:

kaggle datasets download -d joebeachcapital/metropt-3-dataset -p data/


Extract:

unzip data/metropt-3-dataset.zip -d data/

📓 Train Your Model (MLflow)

Open the notebook:

cd notebooks
jupyter notebook


Run train_mlflow.ipynb to:

Load the MetroPT-3 data from data/

Train a model

Log metrics and models with MLflow

📌 Important Notes

Don’t commit large data files or model files to GitHub. Use .gitignore.

Use MLflow to track experiments and models.

This dataset contains time-series sensor data for predictive maintenance tasks. 
Kaggle

📦 Requirements

Make sure you have:

pip install pandas scikit-learn mlflow jupyter

🌐 References

The MetroPT-3 dataset from Kaggle provides sensor data for anomaly detection and predictive maintenance tasks.
