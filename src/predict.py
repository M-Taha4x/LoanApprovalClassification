import pandas as pd
import numpy as np
import joblib
df=pd.read_csv("D:\ML_Projects\LoanStatusApproval\data\loan_data.csv")
model=joblib.load(r"D:\ML_Projects\LoanStatusApproval\model\randomforest.pkl")
scaler=joblib.load(r"D:\ML_Projects\LoanStatusApproval\model\scaler.pkl")
sample=pd.DataFrame(
    {
        'person_age':[20], 
        'person_gender':[1], 
        'person_income':[23000], 
        'person_emp_exp':[1], 
        'loan_amnt':[39000], 
        'loan_int_rate':[6], 
        'loan_percent_income':[0.23], 
        'cb_person_cred_hist_length':[3], 
        'credit_score':[710], 
        'previous_loan_defaults_on_file':[0], 
        'loan_intent_DEBTCONSOLIDATION':[1], 
        'loan_intent_EDUCATION':[0], 
        'loan_intent_HOMEIMPROVEMENT':[0], 
        'loan_intent_MEDICAL':[0], 
        'loan_intent_PERSONAL':[0],
        'loan_intent_VENTURE':[0], 
        'person_home_ownership_MORTGAGE':[1], 
        'person_home_ownership_OTHER':[0],
        'person_home_ownership_OWN':[0], 
        'person_home_ownership_RENT':[0], 
        'person_education_Associate':[0], 
        'person_education_Bachelor':[0], 
        'person_education_Doctorate':[0], 
        'person_education_High School':[0], 
        'person_education_Master':[1]
    }
)
#numerical columns
num_col = [
    "person_age",
    "person_income",
    "person_emp_exp",
    "loan_amnt",
    "loan_int_rate",
    "loan_percent_income",
    "cb_person_cred_hist_length",
    "credit_score"
]
#print(scaler.feature_names_in_)
sample["person_income"] = np.log1p(sample["person_income"])
sample[num_col]=scaler.transform(sample[num_col])
prediction=model.predict(sample)
if prediction[0]==1:
    print("Loan Approved")
else:
    print("Loan Rejected")
probability = model.predict_proba(sample)

approved=df[df['loan_status']==1]
print(approved.head())

print(f"Probability of Rejection : {probability[0][0]*100:.2f}%")
print(f"Probability of Approval  : {probability[0][1]*100:.2f}%")