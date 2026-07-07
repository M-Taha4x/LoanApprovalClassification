# Loan Approval Classification
## Overview
This project uses Machone Learning to predict **Loan Status** based on several input features.The goal is to build  an end-to-end ML pipeline including data preprocessing,
exploratory data analysis(EDA),model training,evaluation,and deployment.
---
## Data set
* Dataset Source: **Kaggle**
* Number of Rows: **45000**
* Number of Columns: **14**
* Taget Variable: **Loan_Status**
---
## Features
* person_age	
* person_gender	
* person_education	
* person_income	
* person_emp_exp	
* person_home_ownership	
* loan_amnt	
* loan_intent	
* loan_int_rate	
* loan_percent_income	
* cb_person_cred_hist_length	
* credit_score	
* previous_loan_defaults_on_file
---
## Project Workflow
1. Data Collection
2. Data Cleaning
3. Exploratory Data Analysis
4. Feature Engineering
5. Data Preprocessing
6. Model Training
7. Model Evaluation
8. Model Saving
9. Prediction
10. Streamlit Web Application

---
## Model Used
* Logistic Regression
* Decision Tree Classifier
* Random Forest Classifier
* XGB Classifier
* KNN

---
## Evaluation Metrics
* Accuracy Score
* Precision Score
* R1 Score
* Recall Score
* Roc-Auc Score
* Confusion Matrix

---
## Best Models
    Model Name: ** Random Forest **
    Reason:Because it have the hisghest accuracy score among all the models
    Accuracy	Precision	Recall	F1 Scorre	ROC-AUC Score
	0.929867	0.904492	0.7652	0.829036	0.871057
---
## Technoligies Used
* Python
* Pandas
* NumPy
* Matplotlib
* Seaborn
* Sckit-Learn
* XGBoost
* Streamlit
* Joblib
---
## Project Structure
LoanStatusApproval
├── data/
|  ├── loan_data.csv

├── notebook/
|   ├── loan.ipynb 

├── src/

│ ├── train.py

│ └── predict.py

├── models/
|  ├──randomforest.pkl

|  ├──scaler.pkl 

├── app.py

├── requirements.txt

├── README.md

└── .gitignore
---
## How to Run


1. Clone the repository.

2. Install dependencies

pip install -r requirements.txt

3. Train the model

python src/train.py

4. Run prediction

python src/predict.py

5. Launch Streamlit

streamlit run app.py

---
## Future Improvements

* Hyperparameter tuning
* Better feature engineering
* Deploy on Streamlit Cloud
* Add model explainability
* Improve UI

---
## Author
**Muhammad Taha**