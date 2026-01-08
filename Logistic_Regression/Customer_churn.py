import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import classification_report,confusion_matrix,ConfusionMatrixDisplay,accuracy_score
import streamlit as st
import seaborn as sns

st.set_page_config("Customer Churn Prediction")

def load_css(file):
    try:
        with open(file) as f:
            st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)
    except FileNotFoundError:
        st.warning(f"⚠ CSS file '{file}' not found — loading without custom styles.")


load_css('style.css')

st.markdown("""
<div class="card">
            <h1>Logisitic Regression</h1>
            <p>Customer Churn Prediction Using Logistic Regression</p>
            </div>""",unsafe_allow_html=True)

df=pd.read_csv('WA_Fn-UseC_-Telco-Customer-Churn.csv')

st.markdown("<div class='card'><h2>Dataset Preview</h2></div>",unsafe_allow_html=True)
st.dataframe(df.head())

ohe=OneHotEncoder(sparse_output=False)
cat_cols=df.select_dtypes(include=['object']).columns.tolist()

df['Churn']=ohe.fit_transform(df[['Churn']])

df['TotalCharges']=pd.to_numeric(df['TotalCharges'],errors='coerce')
df['TotalCharges'].fillna(df['TotalCharges'].median(),inplace=True)
x=df.select_dtypes(include=['int64','float64'])
y=df['Churn']
x=x.drop(['SeniorCitizen','Churn'],axis=1)

X_train,X_test,y_train,y_test=train_test_split(x,y,test_size=0.25,random_state=42)
scaler=StandardScaler()
X_train=scaler.fit_transform(X_train)
X_test=scaler.transform(X_test)

model=LogisticRegression()
model.fit(X_train,y_train)
y_pred=model.predict(X_test)

st.markdown("<div class='card'><h2>Model Performance</h2></div>",unsafe_allow_html=True)
st.metric("Accuracy Score",f"{accuracy_score(y_test,y_pred)*100:.2f}%")

st.markdown(f"""<div class='card'>
            <h2>Classification Report</h2>
            """,unsafe_allow_html=True)
report_dict=classification_report(y_test,y_pred,output_dict=True)
df_report = pd.DataFrame(report_dict).transpose()
st.dataframe(df_report)

st.markdown(f"<div class='card'><h2>Confusion Matrix</h2></div>",unsafe_allow_html=True)
cm = confusion_matrix(y_test, y_pred)

fig, ax = plt.subplots()
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax)

st.pyplot(fig)

st.markdown("<div class='card'><h2>Churn Prediction</h2></div>",unsafe_allow_html=True)
t=st.slider("Adjust Tenure",min_value=df['tenure'].min(),max_value=df['tenure'].max(),value=int(df['tenure'].mean()),step=1)
m=st.slider("Adjust Monthly Charges",min_value=int(df['MonthlyCharges'].min()),max_value=int(df['MonthlyCharges'].max()),value=int(df['MonthlyCharges'].mean()),step=1)
total=st.slider("Adjust Total Charges",min_value=int(df['TotalCharges'].min()),max_value=int(df['TotalCharges'].max()),value=int(df['TotalCharges'].mean()),step=1)
churn=model.predict(scaler.transform([[t,m,total]]))
st.markdown(f"<div class='prediction-box'>Churn-Prediction:{churn}</div>",unsafe_allow_html=True)


