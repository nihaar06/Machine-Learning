import pandas as pd
import numpy as np
import streamlit as st
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score,mean_absolute_error,mean_squared_error,root_mean_squared_error
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

st.set_page_config("Linear Regression Practice")

def load_css(file):
    with open(file) as f:
        st.markdown(f"<style>{f.read()}<style>",unsafe_allow_html=True)

load_css("style.css")

st.markdown("""<div class='card'>
            <h1>Simple Linear Regression Practice</h1>
            </div>""",unsafe_allow_html=True)

@st.cache_data
def load_data():
    return pd.read_csv("Experience-Salary.csv")

data=load_data()

st.markdown("<div class='card'>",unsafe_allow_html=True)
st.subheader("Dataset Preview")
st.dataframe(data.head())
st.markdown("</div>",unsafe_allow_html=True)

x,y=data[['salary(in thousands)']],data['exp(in months)']

scaler=StandardScaler()
x_train,x_test,y_train,y_test=train_test_split(x,y,random_state=42,test_size=0.2)
x_train=scaler.fit_transform(x_train)
x_test=scaler.transform(x_test)

model=LinearRegression()
model.fit(x_train,y_train)

y_pred=model.predict(x_test)

st.markdown("<div class='card'>",unsafe_allow_html=True)
st.subheader("Salary vs Experience")
fig,ax=plt.subplots()
ax.scatter(data['salary(in thousands)'],data['exp(in months)'],alpha=0.4)
ax.plot(data['salary(in thousands)'],model.predict(scaler.transform(x)),color='red')
ax.set_xlabel("Salary(in thousands)")
ax.set_ylabel("Experience(in months)")
st.plotly_chart(fig)

r2=r2_score(y_test,y_pred)
mse=mean_squared_error(y_test,y_pred)
mae=mean_absolute_error(y_test,y_pred)
rmse=root_mean_squared_error(y_test,y_pred)

st.markdown("<div class='card'>",unsafe_allow_html=True)
st.subheader("Model Performance")
c1,c2=st.columns(2)
c1.metric("R2 Score:",f'{r2:.2f}')
c2.metric("Mean Absolute Error:",f'{mae:.2f}')
c3,c4=st.columns(2)
c3.metric("Mean Squared Error",f'{mse:.2f}')
c4.metric("Root Mean Squared Error",f'{rmse:.2f}')

st.markdown(f"""<div class='card'>
            <h3>Slope & Intercept</h3>
            <p><b>Intercept(c):</b>{model.intercept_:.3f}<br>
            <b>Slope(m):{model.intercept_:.3f}""",unsafe_allow_html=True)

st.markdown("<div class='card'>",unsafe_allow_html=True)
st.subheader("Predicted Experience")
salary=st.slider("Salary(In thousands)",data['salary(in thousands)'].min(),data['salary(in thousands)'].max(),20.0)
exp=model.predict(scaler.transform([[salary]]))[0]
st.markdown(f"<div class='prediction-box'>Predicted Experince(in Months):{exp:.2f}<div>",unsafe_allow_html=True)