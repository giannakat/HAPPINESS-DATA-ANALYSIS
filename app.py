import streamlit as st
import pandas as pd

# styles
with open("styles/main.css") as f:
    st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

#load dataset (world-happiness-report.csv)
df = pd.read_csv("WHR2024.csv")

tabs = st.tabs(["Overview", "Dataset", "Exploration", "Analysis", "Conclusions"])

with tabs[0]:
    st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Poppins:wght@800&display=swap');

    .global-happiness {
        font-family: 'Poppins', sans-serif;
        font-weight: 800;
        font-size: 2.8rem;
        color: #0f6a69;
    }
    </style>
    <br>
    <h1 style="font-size:2.2rem; font-weight:700; text-align:center;">
    UNDERSTANDING<br> 
    <span class="global-happiness" style="color:#FDB12A; font-size:4.8rem;">GLOBAL HAPPINESS</span><br> 
    THROUGH DATA ANALYSIS
    </h1>
    <br><br><br>
    """, unsafe_allow_html=True)

    col1, col2 = st.columns([2, 2])
    with col1:
        st.markdown(
        """
        <div style='text-align: justify;'>
        <h4>Introduction</h4>
        This project analyzes the <b>World Happiness Report 2024</b> to understand what drives 
        happiness across nations. By applying correlation, regression, and clustering techniques, 
        we uncover meaningful patterns and relationships among global well-being indicators.
        </div>
        """,
        unsafe_allow_html=True
        )
            
    with col2:
        st.image("assets/happiness.jpg", use_container_width=True)

    st.write("")
    st.write("")
    st.write("")

    st.divider()

    st.subheader("Research Question")
    st.write("""
    The goal of this project is to identify and understand which factors have the greatest 
    influence on a country’s happiness score in 2024. Specifically, the study aims to answer 
    the following questions:
    """)

    # List of research questions
    questions = [
        "Which variables are most strongly correlated with happiness scores?",
        "How do these contributing factors differ among countries or regions?",
        "Can we group countries into clusters based on their happiness-related indicators?",
        "Can we predict happiness scores based on measurable socioeconomic factors?"
    ]

    # Display each question in a box
    for q in questions:
        st.markdown(f"""
        <div style='padding: 15px; margin-bottom: 10px; background-color:#FFF8E1; 
                    border-left: 4px solid #FBC02D; border-radius: 8px;'>
            <b>{q}</b>
        </div>
        """, unsafe_allow_html=True)


    st.write("")
    st.write("")

    st.divider()

    st.subheader("Chosen Data Analysis Technique")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("""
        <div style='padding:15px; border-radius:10px; background:#FFF8E1; border-left:4px solid #FDB12A;'>
            <b>Descriptive Analysis</b><br>
            Summarizes the distribution of happiness scores and key indicators.
        </div>
        """, unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)

        st.markdown("""
        <div style='padding:15px; border-radius:10px; background:#FFF8E1; border-left:4px solid #FDB12A;'>
            <b>Correlation Analysis</b><br>
            Examines relationships between happiness scores and related variables.
        </div>
        """, unsafe_allow_html=True)

    with col2:
        st.markdown("""
        <div style='padding:15px; border-radius:10px; background:#FFF8E1; border-left:4px solid #FDB12A;'>
            <b>Multiple Linear Regression</b><br>
            Models how predictors such as GDP and social support influence happiness.
        </div>
        """, unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)

        st.markdown("""
        <div style='padding:15px; border-radius:10px; background:#FFF8E1; border-left:4px solid #FDB12A;'>
            <b>Cluster Analysis</b><br>
            Groups countries with similar happiness profiles (e.g., via K-Means).
        </div>
        """, unsafe_allow_html=True)


    st.write("")
    st.write("")

    st.divider()

    st.subheader("Expected Outcome")

    st.markdown("""
    <div style='padding: 15px; background-color:#FFF8E1; border-left: 4px solid #FDB12A; border-radius: 8px; text-align: justify;'>
    This study is expected to identify the key factors that most significantly influence national
    happiness scores. Through correlation and regression analysis, we aim to quantify the impact
    of indicators such as GDP, social support, life expectancy, freedom, and generosity.  
    Cluster analysis is expected to reveal groups of countries with similar happiness profiles,
    highlighting regional patterns and socioeconomic similarities.  
    <br>
    Overall, these findings will provide actionable insights into global well-being and help
    policymakers understand which areas require attention to improve quality of life.</div>
    """, unsafe_allow_html=True)

    st.write("")
    st.write("")
    st.write("")
    
    with st.expander('References'):
        st.write("Dataset: https://www.kaggle.com/datasets/ajaypalsinghlo/world-happiness-report-2024")

# DATASET
with tabs[1]:
    
    st.title("Dataset Overview")

    #ABOUT THE DATASET
    st.subheader("About the Dataset")

    st.write("""
    The **World Happiness Report** is a global survey that evaluates the state of happiness
    across countries. It is widely used by governments, organizations, and policymakers to
    understand well-being and guide policy decisions.

    The report combines research from multiple fields—including economics, psychology,
    public policy, survey analysis, and statistics—to evaluate and compare life
    satisfaction among nations. Data is primarily sourced from the **Gallup World Poll**.

    **Interpretation of the Columns**  
    The happiness score (*Ladder Score*) represents each country’s overall life evaluation.
    The remaining variables estimate the contribution of six key factors:
    - Economic production (Log GDP per capita)  
    - Social support  
    - Healthy life expectancy  
    - Freedom to make life choices  
    - Generosity  
    - Perceptions of corruption  

    These factors do **not directly add up to the total score**, but they help explain why
    some countries rank higher or lower by comparing them to *Dystopia*—a hypothetical
    country with the lowest global values for each factor.
    """)

    st.divider()

    st.write(
        """
        Below is a structured summary of the World Happiness Report 2024 dataset,
        including column names, data types, and descriptions of each variable.
        """
    )

    # DATASET PREVIEW
    with st.expander("📌 Dataset Preview"):
        st.dataframe(df, use_container_width=True)

    st.divider()

    # STRUCTURE TABLE
    st.subheader("📊 Dataset Structure")

    structure_df = pd.DataFrame({
        "Data Type": df.dtypes.astype(str),
        "Description": [
            "Name of the country",
            "Overall happiness (ladder) score",
            "Upper bound of estimated happiness",
            "Lower bound of estimated happiness",
            "Contribution of GDP per capita",
            "Contribution of social support",
            "Contribution of healthy life expectancy",
            "Contribution of freedom to make life choices",
            "Contribution of generosity",
            "Contribution of perceived corruption",
            "Baseline + unexplained component of happiness"
        ]
    })

    st.dataframe(structure_df, use_container_width=True)

    st.divider()

    # SUMMARY STATISTICS
    st.subheader("📈 Summary Statistics")
    st.write(df.describe())

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



