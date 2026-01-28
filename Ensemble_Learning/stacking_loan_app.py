import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import StackingClassifier
from sklearn.metrics import accuracy_score

# ===============================
# LOAD & PREPARE DATA
# ===============================
@st.cache_data
def load_data():
    df = pd.read_csv("train_u6lujuX_CVtuZ9i.csv")

    # Fill missing
    cat_num = df[['Credit_History','Dependents','Loan_Amount_Term']]
    for i in cat_num:
        df[i] = df[i].fillna(df[i].mode()[0])

    df['LoanAmount'] = df['LoanAmount'].fillna(df['LoanAmount'].mean())
    df.dropna(inplace=True)

    # Encode categorical cols
    cat_cols = df.select_dtypes(include='object')
    for col in cat_cols:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col])

    return df

df = load_data()

used_features = [
    'ApplicantIncome','CoapplicantIncome','LoanAmount',
    'Loan_Amount_Term','Credit_History','Self_Employed','Property_Area'
]

X = df[used_features]
y = df['Loan_Status']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)

# Scaling
ss = StandardScaler()
X_train = ss.fit_transform(X_train)
X_test = ss.transform(X_test)

# ===============================
# BUILD BASE MODELS + STACKING
# ===============================
base_models = [
    ('lr', LogisticRegression()),
    ('dt', DecisionTreeClassifier(max_depth=3)),
    ('knn', KNeighborsClassifier(n_neighbors=5))
]

# Fit base models for individual reporting
for _, model in base_models:
    model.fit(X_train, y_train)

stack_model = StackingClassifier(
    estimators = base_models,
    final_estimator = LogisticRegression(),
    cv = 5
)

stack_model.fit(X_train, y_train)

# ===============================
# STREAMLIT UI
# ===============================
st.title("🎯 Smart Loan Approval System – Stacking Model")
st.write("This app predicts loan approval using a Stacking Ensemble Machine Learning model.")

st.sidebar.header("Applicant Financial Information")
ApplicantIncome = st.sidebar.number_input("Applicant Income", value=5000)
CoapplicantIncome = st.sidebar.number_input("Coapplicant Income", value=0)
LoanAmount = st.sidebar.number_input("Loan Amount", value=150)
LoanTerm = st.sidebar.number_input("Loan Amount Term (months)", value=360)

st.sidebar.header("Applicant Profile")
CreditHistory = st.sidebar.radio("Credit History", ("Yes","No"))
EmploymentStatus = st.sidebar.selectbox("Employment Status", ("Salaried","Self-Employed"))
PropertyArea = st.sidebar.selectbox("Property Area", ("Urban","Semi-Urban","Rural"))

# Map UI inputs
CreditHistory = 1 if CreditHistory == "Yes" else 0
EmploymentStatus = 1 if EmploymentStatus == "Self-Employed" else 0
PropertyArea = {"Urban":2, "Semi-Urban":1, "Rural":0}[PropertyArea]

# Prepare input
input_data = np.array([[ApplicantIncome, CoapplicantIncome, LoanAmount,
                        LoanTerm, CreditHistory, EmploymentStatus, PropertyArea]])

input_scaled = ss.transform(input_data)

st.subheader("📦 Model Architecture (Stacking Ensemble)")
st.info("""
**Base Models**
- Logistic Regression
- Decision Tree
- KNN

**Meta Model**
- Logistic Regression
""")

if st.button("🔘 Check Loan Eligibility (Stacking Model)"):

    # Base model predictions
    base_preds = {}
    for name, model in base_models:
        pred = model.predict(input_scaled)[0]
        base_preds[name] = "Approved" if pred == 1 else "Rejected"

    # Meta model prediction
    final_pred = stack_model.predict(input_scaled)[0]
    final_label = "Approved" if final_pred == 1 else "Rejected"

    st.subheader("🧠 Final Stacking Decision")
    if final_label == "Approved":
        st.success(f"✅ Loan Approved")
    else:
        st.error(f"❌ Loan Rejected")
