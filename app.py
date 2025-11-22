import streamlit as st
import pandas as pd

# styles
with open("styles/main.css") as f:
    st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

#load dataset (world-happiness-report.csv)
df = pd.read_csv("WHR2024.csv")

st.write("UNDERSTANDING GLOBAL HAPPINESS THROUGH DATA ANALYSIS")   
tabs = st.tabs(["Overview", "Dataset", "Exploration", "Analysis", "Conclusions"])

with tabs[0]:
    st.write("Briefly introduce the dataset, research question, and selected analysis technique.")

with tabs[1]:
    st.write("Display the dataset's structure (e.g., tables, column descriptions) using Streamlit's data visualization tools.")
    st.dataframe(df, use_container_width=True)
    # st.write(df.head())   

with tabs[2]:
    st.write("Demonstrate how you prepared the dataset for analysis (e.g., handling missing values, cleaning steps). Include visualizations e.g., histograms, heatmaps to explain data distribution or transformations.")

with tabs[3]:
    st.write("Provide interactive visualizations of your results (e.g., scatter plots, cluster maps, regression lines). Highlight key insights, patterns, trends, or anomalies. Add filters or sliders to allow users to explore the data further.")

with tabs[4]:
    st.write("- Summarize the main takeaways and provide actionable recommendations based on your findings. Include text boxes or dropdowns to let users explore different insights interactively.")
