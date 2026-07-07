import streamlit as st
import joblib
import pandas as pd
df=pd.read_csv(r"D:\ML_Projects\LoanStatusApproval\data\loan_data.csv")
st.sidebar.title("Loan Status Approval")
st.sidebar.write(""" 
    Model:
    Random Forest Classifier
    
    Developer:
    Muhammad Taha
    
    University:
    GIKI
""")
st.set_page_config(
    page_title="Loan Status Approval",
    layout="centered"
)
model=joblib.load(r"D:\ML_Projects\LoanStatusApproval\model\randomforest.pkl")
scaler=joblib.load(r"D:\ML_Projects\LoanStatusApproval\model\scaler.pkl")
st.markdown(""" 
    <h1 style='text-align:center;color:#4CAF50'>
    Loan Status Approval
    </h1>
    """,unsafe_allow_html=True)
st.markdown("""
    <h3 style='text-allign:centre;color:gray;'>
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
            "Education",["Bachelor","Associate","High School","Master","Doctorate"]
        )
        owner_ship=st.selectbox(
            "Home OwnerShip",["Rent","Own","Mortgage","Other"]
        )
        intent=st.selectbox(
            "Loan Intent",["Education","Medical","Venture","Personal","Debtconsolidation","HomeImprovement"]
        )
        prev_loan=st.selectbox(
            "Previous Loan Defaults",["Yes","No"]
        )
    with col2:
        age=st.number_input("Person Age",min_value=20,max_value=140)
        income=st.number_input("Person Income",min_value=8.98,max_value=15.78)
        emp_exp=st.number_input("Person Emp Exp",min_value=0,max_value=125)
        loan_amt=st.number_input("Loan Amount",min_value=500,max_value=35000)
    with col3:
        loan_int=st.number_input("Loan Interest Rate",min_value=5.42,max_value=20)
        loan_per_inc=st.number_input("Loan Percent Income",min_value=0,max_value=0.66)
        cb_len=st.number_input("Credit History Length",min_value=2,max_value=30)
        crd_score=st.number_input("Credit Score",min_value=390,max_value=850)
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
            "Home Improvement":(0,0,0,0,1,0),
            "Debt Consolidation":(0,0,0,0,0,1)
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
        with st.spinner("Predicting Loan Approval Status"):
            prediction=model.predict(sample)
        pred_score=prediction[0]
        probability = model.predict_proba(sample)
        if pred_score==1:
            st.metric("Loan Approved ",f"Probabilit of Approval: {probability[0][1]}")
            st.balloons()
        else:
            st.metric("Loan Rejected",f"Probability of Rejection: {probability[0][0]}")
            st.balloons()
with tab2:
    st.header("Dataset Information")
    show_df=st.checkbox("Dataset")
    if show_df:
        st.dataframe(df)
    st.write(" Dataset Source: Kaggle")
    st.write("Number of Rows:45000")
    st.write("Number of Columns: 14")
    st.write("Taget Variable: Loan_Status")
with tab3:
    col4,col5=st.columns(2)
    st.header("About This Project")
    st.subheader("Technologies Used")
    with col4:
        st.write("Pyhthon")
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
    st.success("Thank you for using this Application")
with tab4:
    st.header("About Myself")
    st.write("""Hello myself Taha From Pakistan,
             - currently pursuing BS AI(4th sem) from GIKI,
             -its my third ml model which I worked on.
             - 
             """)
        