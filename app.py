import streamlit as st


# styles
with open("styles/main.css") as f:
    st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)


st.write("DATA ANALYSIS PROJECT")