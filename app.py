#### LOCAL APP
#### Sentiment analysis application
#### Import packages
import streamlit as st
from textblob import TextBlob
import plotly.graph_objects as go
#### Add header to describe app
st.markdown("# Sentiment Analysis Application!!")
#### Create text input box and save incoming text to variable called text
text = st.text_input("Enter text:", value = "Enter text here")
#### TextBlob to analyze input text
score = TextBlob(text).sentiment.polarity
#### Create label (called sent) from TextBlob polarity score to use in summary below
if score > .15:
    label = "Positive"
elif score < -.15:
    label = "Negative"
else:
    label = "Neutral"
    
##### Show results
#### Print sentiment score, label, and language
st.markdown(f"Sentiment label: **{label}**")
#### Create sentiment gauge
fig = go.Figure(go.Indicator(
    mode = "gauge+number",
    value = score,
    title = {'text': f"Sentiment: {label}"},
    gauge = {"axis": {"range": [-1, 1]},
            "steps": [
                {"range": [-1, -.15], "color":"red"},
                {"range": [-.15, .15], "color":"gray"},
                {"range": [.15, 1], "color":"lightgreen"}
            ],
            "bar":{"color":"yellow"}}
))
st.plotly_chart(fig)