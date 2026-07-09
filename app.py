import streamlit as st
st.set_page_config(
    page_title="Loan Status Approval",
    layout="centered"
)
import joblib
import numpy as np
import pandas as pd
model=joblib.load(r"D:\ML_Projects\LoanStatusApproval\model\randomforest.pkl")
scaler=joblib.load(r"D:\ML_Projects\LoanStatusApproval\model\scaler.pkl")
df=pd.read_csv(r"D:\ML_Projects\LoanStatusApproval\data\loan_data.csv")
st.sidebar.title("Loan Status Approval")
st.sidebar.header("Project Info")
st.sidebar.success("Random Forest Classifier")
st.sidebar.write("Muhammad Taha")
st.sidebar.write("GIKI")


st.markdown(""" 
    <h1 style='text-align:center;color:#4CAF50'>
    Loan Status Approval
    </h1>
    """,unsafe_allow_html=True)
st.markdown("""
    <h3 style='text-align:centre;color:gray;'>
    Predicts Loan's Approval(Accepted or Rejected) using ML
    </h3>
     """,unsafe_allow_html=True)
st.divider()
tab1,tab2,tab3,tab4=st.tabs(
    ["Prediction","Dataset","About The Project","About"]
)
with tab1:
    st.info("Enter Person's Information Below")
    col1,col2,col3=st.columns(3)
    with col1:
        gender=st.selectbox(
            "Gender",["Male","Female"]
        )
        education=st.selectbox(
            "Education",["Bachelors","Associate","High School","Master","Doctorate"]
        )
        owner_ship=st.selectbox(
            "Home Ownership",["Rent","Own","Mortgage","Other"]
        )
        intent=st.selectbox(
            "Loan Intent",["Education","Medical","Venture","Personal","Debtconsolidation","HomeImprovement"]
        )
        prev_loan=st.selectbox(
            "Previous Loan Defaults",["Yes","No"]
        )
    with col2:
        age=st.number_input(" Age",min_value=20,max_value=140,value=20)
        income=st.number_input(" Income",min_value=1000,max_value=1000000,value=25000)
        emp_exp=st.number_input("Employment Experience",min_value=0,max_value=125,value=0)
        loan_amt=st.number_input("Loan Amount",min_value=500,max_value=35000,value=500)
    with col3:
        loan_int=st.number_input(" Interest Rate",min_value=5.42,max_value=20.0,value=5.42)
        loan_per_inc=st.number_input("Loan Percent Income",min_value=0.0,max_value=0.66,value=0.0)
        cb_len=st.number_input("Credit History Length",min_value=2,max_value=30,value=2)
        crd_score=st.number_input("Credit Score",min_value=390,max_value=850,value=390)
    st.divider()
    if st.button("Predict Loan Approval Status",use_container_width=True):
        gender_map={
            "Female":0,
            "Male":1
        }
        pre_loan_map={
            "No":0,
            "Yes":1
        }
        edu_map={
            "Associate":(1,0,0,0,0),
            "Bachelors":(0,1,0,0,0),
            "Doctorate":(0,0,1,0,0),
            "High School":(0,0,0,1,0),
            "Master":(0,0,0,0,1)
        }
        own_map={
            "Mortgage":(1,0,0,0),
            "Other":(0,1,0,0),
            "Own":(0,0,1,0),
            "Rent":(0,0,0,1)
        }
        intent_map={
             "Personal":(1,0,0,0,0,0),
            "Education":(0,1,0,0,0,0),
            "Venture":(0,0,1,0,0,0),
            "Medical":(0,0,0,1,0,0),
            "HomeImprovement":(0,0,0,0,1,0),
            "Debtconsolidation":(0,0,0,0,0,1)
        }
        gender=gender_map[gender]
        prev_loan=pre_loan_map[prev_loan]
        edu_ass,edu_bach,edu_doc,edu_hs,edu_ms=edu_map[education]
        own_mort,own_other,own_own,own_rent=own_map[owner_ship]
        int_per,int_edu,int_ven,int_med,int_imp,int_con=intent_map[intent]
        sample=pd.DataFrame({
           'person_age':[age], 
           'person_gender':[gender],
           'person_income':[income], 
           'person_emp_exp':[emp_exp], 
           'loan_amnt':[loan_amt], 
           'loan_int_rate':[loan_int],
           'loan_percent_income':[loan_per_inc], 
           'cb_person_cred_hist_length':[cb_len], 
           'credit_score':[crd_score],
           'previous_loan_defaults_on_file':[prev_loan],
           'loan_intent_DEBTCONSOLIDATION':[int_con], 
           'loan_intent_EDUCATION':[int_edu], 
           'loan_intent_HOMEIMPROVEMENT':[int_imp], 
           'loan_intent_MEDICAL':[int_med], 
           'loan_intent_PERSONAL':[int_per], 
           'loan_intent_VENTURE':[int_ven], 
           'person_home_ownership_MORTGAGE':[own_mort],
           'person_home_ownership_OTHER':[own_other], 
           'person_home_ownership_OWN':[own_own], 
           'person_home_ownership_RENT':[own_rent], 
           'person_education_Associate':[edu_ass], 
           'person_education_Bachelor':[edu_bach], 
           'person_education_Doctorate':[edu_doc], 
           'person_education_High School':[edu_hs], 
           'person_education_Master':[edu_ms]
        })
        sample["person_income"]=np.log1p(sample["person_income"])
        num_col=[
            "person_age",
            "person_income",
            "person_emp_exp",
            "loan_amnt",
            "loan_int_rate",
            "loan_percent_income",
            "cb_person_cred_hist_length",
            "credit_score"
        ]
        sample[num_col]=scaler.transform(sample[num_col])
        col1,col2,col3=st.columns(3)
        st.write(sample)
        with st.spinner("Predicting Loan Approval Status"):
            prediction=model.predict(sample)
            probability = model.predict_proba(sample)
        pred_score=prediction[0]
        st.write(probability)
        if pred_score==1:
            st.success("Loan Approved")
            st.metric("Approval Probability ",f"{probability[0][1]*100:.2f}%")
            st.progress(float(probability[0][1]))
            st.balloons()
        else:
            st.error("Loan Rejected")
            st.metric(
                "Rejection Probability ",f"{probability[0][0]*100:.2f}%"
            )
            st.progress(float(probability[0][0]))
        
with tab2:
    st.header("Dataset Information")
    st.dataframe(df.head())
    if st.checkbox("Show Complete Dataset"):
        st.dataframe(df)
    st.markdown("[Source](https://www.kaggle.com/datasets/taweilo/loan-approval-classification-data)")
    st.write("Number of Rows:45000")
    st.write("Number of Columns: 14")
    st.write("Taget Variable: Loan_Status")
with tab3:
    st.header("About This Project")
    st.subheader("Technologies Used")
    col4,col5=st.columns(2)
    with col4:
        st.write("Python")
        st.write("Pandas")
        st.write("Numpy")
        st.write("Seaborn")
        st.write("Matplotlib")
    with col5:
        st.write("Sckit-Learn")
        st.write("XGBoost")
        st.write("Streamlit")
        st.write("Joblib")
    st.subheader("Model Used")
    st.write("""
     - Random Forest Classifier
     - XGB Classifier
     - KNN
     - Logistic Regression
     - Decision Tree Classifier       
    """)
    st.subheader("Best Model")
    st.write("Random Forest")
    st.subheader("Developer")
    st.write("Muhammad Taha")
    st.subheader("University")
    st.write("GIKI")
    st.subheader("Model Performance")

    col1,col2=st.columns(2)

    with col1:
        st.metric("Accuracy","92.72%")
        st.metric("Precision","88.99%")

    with col2:
        st.metric("Recall","77.02%")
        st.metric("ROC-AUC","87.14%")
        st.success("Thank you for using this Application")
with tab4:
    st.header("About Me")
    st.write("""Muhammad Taha

         BS Artificial Intelligence
        Ghulam Ishaq Khan Institute (GIKI)

    • Machine Learning
    • Data Science
    • Python
    • Streamlit
             """)
    st.markdown(
        "[GitHub Profile](https://github.com/M-Taha4x)"
    )
        