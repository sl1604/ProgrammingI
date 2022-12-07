#!/usr/bin/env python
# coding: utf-8


import pandas as pd
import numpy as np

s = pd.read_csv('./social_media_usage.csv')

def clean_sm(x):
    x = np.where(x == 1, 1, 0)
    return (x)


ss = pd.DataFrame({
    'sm_li':clean_sm(s["web1h"]),
    'income':np.where(s["income"]>9, np.nan, s["income"]),
    'education':np.where(s["educ2"]>8, np.nan, s["educ2"]),
    'parent':np.where(s["par"]==1, 1,0),
    'marital': np.where(s["marital"]==1, 1,0),
    'female': np.where(s["gender"]==2, 1,0),
    'age': np.where(s["age"]>98, np.nan, s["age"])
}).dropna()

y = ss["sm_li"]

x = ss.drop('sm_li', axis = 1)


from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix



x_train, x_test, y_train, y_test = train_test_split(x,
                                                   y,
                                                   stratify=y,
                                                   test_size=0.2,
                                                   random_state=123)

lr=LogisticRegression()
lr.fit(x_train, y_train)
y_pred = lr.predict(x_test)



import streamlit as st
st.markdown("###LinkedIn User Prediction Tool")

st.image("https://344277848-files.gitbook.io/~/files/v0/b/gitbook-legacy-files/o/assets%2F-MFpeVXNyJzRxneVTRCa%2F-MMIZW6_ACYgxYTto9aJ%2F-MMI_Ej9MmRSHhBSNpcS%2Fgeorgetown-msb-mcdonough-best_business_school-1.jpg?alt=media&token=137e2690-f3be-4f9c-85d0-7ada9c49b9c4")


gender_answer = st.selectbox(label="What gender do you identify as?",
options=("male",
         "female",
         "other",
         "Do not know",
         "Skip"))

marital_answer = st.selectbox(label="What is your marital status?",
options=("Married",
         "Living with a partner",
         "Divorced",
         "Separated",
         "Widowed",
         "Never been married",
         "Do not know",
         "Skip"))

                       
parent_answer = st.selectbox(label="Are you a parent of a child under 18 living in your home?",
options=("Yes", "No", "Do not know", "Skip"))

age_answer = st.number_input('Insert your age in number', format='%.0f')

income_answer = st.selectbox(
    'What is your annual income',
    ('Less than 10,000', '10,000 - 20,0000', '20,000-30,000', '30,000-40,000', '40,000-50,000', '50,000-75,000', '75,000-100,000', '100,000-150,000', '150,000+', 'Do not know', 'Skip'))

education_answer = st.selectbox(label="What is your highest obtained education level?",
options=(
            "Less than high school (Grades 1-8 or no formal schooling)",
            "High school incomplete (Grades 9-11 or Grade 12 with NO diploma)",
            "High school graduate (Grade 12 with diploma or GED certificate)",
            "Some college, no degree (includes some community college)",
            "Two-year associate degree from a college or university",
            "Four-year college or university degree/Bachelors degree (e.g., BS, BA, AB)",
            "Some postgraduate or professional schooling, no postgraduate degree (e.g. some graduate school)",
            "Postgraduate or professional degree, including masters, doctorate, medical or law degree (e.g., MA, MS, PhD, MD, JD)",
            "Do not know",
            "Skip"))


if gender_answer == "female":
    gender_answer = 1
else:
    gender_answer = 2
    

if parent_answer == "Yes":
    parent_answer = 1
else:
    parent_answer = 0
    
    
if marital_answer == "Married":
    marital_answer = 1
else:
    marital_answer = 0
    
    
if income_answer == "Less than 10,000":
    income_answer = 1
elif income_answer == "10,000 - 20,0000":
    income_answer = 2
elif income_answer == "30,000-40,000":
    income_answer = 3
elif income_answer == "40,000-50,000":
    income_answer = 4
elif income_answer == "50,000-75,000":
    income_answer = 5
elif income_answer == "75,000-100,000":
    income_answer = 6
elif income_answer == "100,000-150,000":
    income_answer = 7
elif income_answer == "150,000+":
    income_answer = 8
else:
    income_answer = 9
    
    
if education_answer == "Less than high school (Grades 1-8 or no formal schooling)": 
    education_answer = 1
elif education_answer == "High school incomplete (Grades 9-11 or Grade 12 with NO diploma)":
    education_answer = 2
elif education_answer == "High school graduate (Grade 12 with diploma or GED certificate)":
    education_answer = 3
elif education_answer == "Some college, no degree (includes some community college)":
    education_answer = 4
elif education_answer == "Two-year associate degree from a college or university":
    education_answer = 5
elif education_answer == "Four-year college or university degree/Bachelors degree (e.g., BS, BA, AB)":
    education_answer = 6
elif education_answer == "Some postgraduate or professional schooling, no postgraduate degree (e.g. some graduate school)":
    education_answer = 7
elif education_answer == "Postgraduate or professional degree, including masters, doctorate, medical or law degree (e.g., MA, MS, PhD, MD, JD)":
    education_answer = 8
else:
    education_answer = 9



answer_data = pd.DataFrame({
    "income": [income_answer],
    "education":[education_answer],
    "parent": [parent_answer],
    "marital": [marital_answer],
    "female": [gender_answer],
    "age": [age_answer]
})

if st.button('Predict'):

    predicted_class = lr.predict(answer_data)

    probs = lr.predict_proba(answer_data)

    st.text(f"Predicted class: {predicted_class[0]}") #0 = not linkedin
    st.text(f"Probability that this person uses linkedin: {probs[0][1]}")

st.image("https://upload.wikimedia.org/wikipedia/commons/0/01/LinkedIn_Logo.svg")