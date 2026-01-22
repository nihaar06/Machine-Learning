import streamlit as st
import pandas as pd
import numpy as np
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score,confusion_matrix,classification_report

st.header("Smart Loan Approval System")
st.subheader("This system uses Support Vector Machines to predict loan approval.")
df=pd.read_csv("train_u6lujuX_CVtuZ9i.csv")

obj_cols=df.select_dtypes(include='object').columns
for i in obj_cols:
    df[i].fillna(df[i].mode()[0],inplace=True)

cols=['LoanAmount','Credit_History']
for i in cols:
    df[i].fillna(df[i].mean(),inplace=True)
cols=['ApplicantIncome','LoanAmount','Credit_History','Self_Employed','Property_Area']




import seaborn as sns
from sklearn.preprocessing import LabelEncoder
le=LabelEncoder()
df['Loan_Status']=le.fit_transform(df['Loan_Status'])
df['Self_Employed']=le.fit_transform(df['Self_Employed'])
df['Property_Area']=le.fit_transform(df['Property_Area'])
# for i in cols:
#     q1=df[i].quantile(0.25)
#     q3=df[i].quantile(0.75)
#     iqr=q3-q1
#     lb=q1-1.5*iqr
#     ub=q3+1.5*iqr
#     df=df[(df[i]>=lb) & (df[i]<=ub)]
X=df[cols]
y=df['Loan_Status']

X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=42)

ss=StandardScaler()
X_train=ss.fit_transform(X_train)
X_test=ss.transform(X_test)

inc=st.number_input("Applicant Income")
loan=st.number_input("Loan Amount")
credit=st.selectbox("Credit History(Yes/No)",['Yes','No'])
emp_status=st.selectbox("Employment Status",['Yes','No'])
prop=st.selectbox("Property Area",['Urban','Semiurban','Rural'])

if credit=='Yes':
    credit=1
else:
    credit=0

if emp_status=='Yes':
    emp_status=1
else:
    emp_status=0

if prop=='Urban':
    prop=2
elif prop=='Semiurban':
    prop=1
else:
    prop=0

model=st.radio("Select SVM model:",['linear','poly','RBF'])
inp=np.array([inc,loan,credit,emp_status,prop]).reshape(1,-1)
if(model=='linear'):
    svm=SVC(kernel='linear',C=1)
elif(model=='poly'):
    svm=SVC(kernel='poly',degree=5,C=1)
else:
    svm=SVC(kernel='rbf',gamma='scale',C=1)
svm.fit(X_train,y_train)
b=st.button("Predict")
acc = svm.score(X_test, y_test)
st.metric("Model Accuracy", f"{acc*100:.2f}%")
if(b):
    res=svm.predict(ss.transform(inp))[0]
    result="Approved" if res==1 else "Rejected"
    st.success(f"Loan Status:{result}")
