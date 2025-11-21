import streamlit as st
import pandas as pd


# styles
with open("styles/main.css") as f:
    st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)


st.write("DATA ANALYSIS PROJECT")   

#load dataset (world-happiness-report.csv)
df = pd.read_csv("WHR2024.csv")
st.dataframe(df, use_container_width=True)
# st.write(df.head())   