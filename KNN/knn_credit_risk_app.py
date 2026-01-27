import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, confusion_matrix

df = pd.read_csv("credit_risk_dataset.csv")

# Handle missing values
df = df.dropna()

st.header("Customer Risk Prediction System (KNN)")
st.subheader("This system predicts customer risk by comparing them with similar customers.")

X = df[['person_age', 'person_income', 'loan_amnt', 'cb_person_cred_hist_length']]
y = df['loan_status']

# Cache the training data for efficiency
ss = StandardScaler()
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)
X_train = ss.fit_transform(X_train)
X_test = ss.transform(X_test)
st.sidebar.title("User Input Controls")
age = st.sidebar.slider("Age", min_value=int(df['person_age'].min()), max_value=int(df['person_age'].max()))
inc = st.sidebar.number_input("Annual Income", min_value=0.0, max_value=float(df['person_income'].max()))
loan = st.sidebar.number_input("Loan Amount", min_value=0.0, max_value=float(df['loan_amnt'].max()))
credit = st.sidebar.selectbox("Credit History", ['Yes', 'No'])
k = st.sidebar.slider("K-Value", min_value=1, max_value=15)

knn = KNeighborsClassifier(n_neighbors=k, metric='euclidean')
knn.fit(X_train, y_train)
y_pred_acc = knn.predict(X_test)

credit_encoded = 1 if credit == 'Yes' else 0

st.metric("Accuracy:", round(accuracy_score(y_test, y_pred_acc), 4))
st.write("Confusion Matrix:")
st.write(confusion_matrix(y_test, y_pred_acc))

X_pred = ss.transform(np.array([[age, inc, loan, credit_encoded]]))
button = st.sidebar.button("Predict")
if button:
    try:
        y_pred = knn.predict(X_pred)
        result = "High Risk" if y_pred[0] == 1 else "Low Risk"
        st.metric("Predicted Risk:", result)
    except Exception as e:
        st.error(f"Prediction error: {str(e)}")
