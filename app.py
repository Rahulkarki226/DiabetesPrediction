import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import seaborn as sns

# Dataset Loading
df = pd.read_csv('diabetes.csv')

# Heading
st.title("Diabetes Checkup")
st.sidebar.header('Patient Data')
st.subheader('Training Datasets')
st.write(df.describe())

# X and Y Data
X = df.drop(['Outcome'], axis=1)
y = df['Outcome']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

# Function
def user_report():
    pregnancies = st.sidebar.slider('Pregnancies', 0, 17, 3)
    glucose = st.sidebar.slider('Glucose', 0, 200, 120)
    bp = st.sidebar.slider('BloodPressure', 0, 122, 70)
    skinthickness = st.sidebar.slider('SkinThickness', 0, 100, 20)
    insulin = st.sidebar.slider('Insulin', 0, 846, 79)
    bmi = st.sidebar.slider('BMI', 0, 67, 20)
    dpf = st.sidebar.slider('DiabetesPedigreeFunction', 0.0, 2.4, 0.47)
    age = st.sidebar.slider('Age', 21, 88, 33)

    user_report_data = {
        'Pregnancies': pregnancies,
        'Glucose': glucose,
        'BloodPressure': bp,
        'SkinThickness': skinthickness,
        'Insulin': insulin,
        'BMI': bmi,
        'DiabetesPedigreeFunction': dpf,
        'Age': age
    }
    report_data = pd.DataFrame(user_report_data, index=[0])
    return report_data

# Patient Data
user_data = user_report()
st.subheader('Patient Data')
st.write(user_data)

# Model
rf = RandomForestClassifier()
rf.fit(X_train, y_train)
user_result = rf.predict(user_data)

# Color Function
if user_result[0] == 0:
    color = 'Green'
    output = 'Healthy'
else:
    color = 'red'
    output = 'Diabetic'

# Result
st.subheader('Your Report')
st.markdown(f"<h6 style='color: {color};'>{output}</h6>", unsafe_allow_html=True)
st.subheader('Accuracy: ')
st.wreite(str(accuracy_score(y_test, rf.predict(X_test)) * 100) + '%')

# Age vs Insulin
st.header('Insulin Value Graph (Others vs Yours)')
fig_i = plt.figure()
ax9 = sns.scatterplot(x = 'Age', y = 'Insulin', data = df, hue = 'Outcome', palette='rocket')
ax10 = sns.scatterplot(x = user_data['Age'], y = user_data['Insulin'], s = 150, color = color)
plt.xticks(np.arange(10,100,5))
plt.yticks(np.arange(0,900,50))
plt.title('0 - Healthy & 1 - Diabetic')
st.pyplot(fig_i)