import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
df=pd.read_csv("D:\ML_Projects\LoanStatusApproval\data\loan_data.csv")
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
cat_col=df.select_dtypes(include=['object','category']).columns
df['person_income']=np.log1p(df['person_income'])
from sklearn.preprocessing import LabelEncoder
le=LabelEncoder()
df['person_gender']=le.fit_transform(df['person_gender'])
df['previous_loan_defaults_on_file']=le.fit_transform(df['previous_loan_defaults_on_file'])
df=pd.get_dummies(
    df,
    columns=['loan_intent'],
    dtype=int
)
df=pd.get_dummies(
    df,
    columns=['person_home_ownership'],
    dtype=int
)
df=pd.get_dummies(
    df,
    columns=['person_education'],
    dtype=int
)
x=df.drop('loan_status',axis=1)
y=df['loan_status']
from sklearn.model_selection import train_test_split
X_train,X_test,Y_train,Y_test=train_test_split(x,y,test_size=0.25,random_state=42,stratify=y)
#num_col=x.select_dtypes(include=['int64','float64']).columns
from sklearn.preprocessing import StandardScaler
scaler=StandardScaler()
X_train[num_col]=scaler.fit_transform(X_train[num_col])
X_test[num_col]=scaler.transform(X_test[num_col])
from sklearn.ensemble import RandomForestClassifier
rf=RandomForestClassifier(n_estimators=100,random_state=42)
rf.fit(X_train,Y_train)
pred_rf=rf.predict(X_test)
joblib.dump(scaler,"D:\ML_Projects\LoanStatusApproval\model\scaler.pkl")
joblib.dump(rf,r"D:\ML_Projects\LoanStatusApproval\model\randomforest.pkl")
print("Model Saved Succesfully")
#print(x.columns.to_list())
print(scaler.feature_names_in_)