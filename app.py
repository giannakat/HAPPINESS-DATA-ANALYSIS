import streamlit as st
import pandas as pd

# styles
with open("styles/main.css") as f:
    st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

#load dataset (world-happiness-report.csv)
df = pd.read_csv("WHR2024.csv")

tabs = st.tabs(["Overview", "Dataset", "Exploration", "Analysis", "Conclusions"])

with tabs[0]:
    st.title("UNDERSTANDING GLOBAL HAPPINESS THROUGH DATA ANALYSIS")   
    
    col1, col2 = st.columns([1, 2])
    with col1:
        st.subheader("WORLD HAPPINESS")
    
    with col2:
        st.image("assets/happiness.jpg", use_container_width=True)

    st.divider()

    st.subheader("Research Question")
    st.write("""
    The goal of this project is to identify and understand which factors have the greatest 
    influence on a country’s happiness score in 2024. Specifically, the study aims to answer 
    the following questions:
    """)

    st.markdown("""
    - **Which variables are most strongly correlated with happiness scores?**  
    - **How do these contributing factors differ among countries or regions?**  
    - **Can we group countries into clusters based on their happiness-related indicators?**  
    - **Can we predict happiness scores based on measurable socioeconomic factors?**
    """)

    st.divider()

    st.subheader("Chosen Data Analysis Technique")
    st.markdown("""
    - **Descriptive Analysis** To summarize and visualize the distribution of happiness scores and key indicators using charts and tables.
    - **Correlation Analysis** To determine which variables are most strongly related to the happiness score.
    - **Multiple Linear Regression** To model the relationship between happiness and its predictors (GDP, social support, life expectancy, etc.) and estimate the relative influence of each factor.
    - **Cluster Analysis** To group countries with similar happiness profiles and identify regional or socioeconomic patterns among these clusters. Techniques such as K-Means Clustering will be explored for this purpose.
    """)

    st.divider()

    st.subheader("Expected Outcome")

    st.write("""
    We expect to identify the key factors that most significantly influence happiness and visualize their effects through correlation and regression analysis. Through cluster analysis, we also aim to discover groups of countries with similar happiness characteristics, revealing global patterns based on socioeconomic and cultural similarities.
    These findings can help provide meaningful insights into how different factors contribute to happiness, supporting research and policy discussions about improving quality of life worldwide.
    """)
    
    with st.expander('References'):
        st.write("link")

with tabs[1]:
    st.write("Display the dataset's structure (e.g., tables, column descriptions) using Streamlit's data visualization tools.")
    st.dataframe(df, use_container_width=True)
    # st.write(df.head())   

with tabs[2]:
    st.write("Demonstrate how you prepared the dataset for analysis (e.g., handling missing values, cleaning steps). Include visualizations e.g., histograms, heatmaps to explain data distribution or transformations.")

with tabs[3]:
    st.write("Provide interactive visualizations of your results (e.g., scatter plots, cluster maps, regression lines). Highlight key insights, patterns, trends, or anomalies. Add filters or sliders to allow users to explore the data further.")

#conclusion and recommendations
with tabs[4]:
    st.title("Conclusions & Recommendations")

    st.subheader("Summary of Findings")

    st.write(
        """
        Based on the analysis conducted:

        - Happiness scores showed a **moderately varied distribution** across countries,
          with most nations clustering around the middle range.
        - Variables such as **GDP per capita, social support, and healthy life expectancy**
          displayed the strongest relationships with happiness scores.
        - Correlation analysis revealed that **GDP per capita had one of the highest positive correlations**
          with happiness.
        - Cluster analysis suggested that countries can be grouped based on **socioeconomic
          and well-being indicators**, showing clear regional or developmental patterns.
        """
    )

    st.divider()

    st.subheader("Explore Insights")

    insight_option = st.selectbox(
        "Select an insight to view:",
        [
            "Overall Conclusion",
            "Strongest Correlation",
            "Cluster Pattern",
            "Limitations of the Analysis",
        ]
    )

    if insight_option == "Overall Conclusion":
        st.text_area(
            "Overall Conclusion",
            "The analysis suggests that socioeconomic factors such as GDP, social support, and life expectancy play a major role in determining happiness levels across countries.",
            height=120
        )

    elif insight_option == "Strongest Correlation":
        st.text_area(
            "Strongest Correlation",
            "GDP per capita showed the strongest correlation with happiness, indicating that higher economic output and resources contribute significantly to well-being.",
            height=120
        )

    elif insight_option == "Cluster Pattern":
        st.text_area(
            "Cluster Pattern",
            "Cluster analysis revealed groups of countries with similar happiness profiles, often aligning with geographic regions or economic development levels.",
            height=120
        )

    elif insight_option == "Limitations of the Analysis":
        st.text_area(
            "Limitations",
            "Missing values required the removal of some rows using dropna(), which may have reduced dataset representation. Model accuracy may also vary due to limited features.",
            height=120
        )

    st.divider()

    st.subheader("Actionable Recommendations")

    rec_category = st.selectbox(
        "Select recommendation category:",
        [
            "For Policymakers",
            "For Researchers",
            "For Future Data Collection",
        ]
    )

    if rec_category == "For Policymakers":
        default_text = (
            "Invest in factors with the strongest impact on happiness, such as healthcare, "
            "economic stability, and social support systems."
        )
    elif rec_category == "For Researchers":
        default_text = (
            "Include additional variables such as mental health indicators or cultural factors "
            "to improve model accuracy and explanatory power."
        )
    else:
        default_text = (
            "Ensure more complete data reporting to reduce missing values and avoid dropping rows "
            "that affect representation."
        )

    st.text_area("Recommendation Details", default_text, height=120)

    st.divider()

    st.subheader("Final Thoughts")

    st.write(
        """
        These findings highlight the importance of socioeconomic conditions in shaping global happiness.
        Understanding these relationships can support informed decision-making and future research efforts
        aimed at improving quality of life.
        """
    )



