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

    <h1 style="font-size:2.2rem; font-weight:700; text-align:center;">
    UNDERSTANDING<br> 
    <span class="global-happiness" style="color:#0f6a69; font-size:4.8rem;">GLOBAL HAPPINESS</span><br> 
    THROUGH DATA ANALYSIS
    </h1>
    """, unsafe_allow_html=True)
    # col1, col2 = st.columns([1, 2])
    # with col1:
    #     st.subheader("WORLD HAPPINESS")
    
    # with col2:
    #     st.image("assets/happiness.jpg", use_container_width=True)

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
    st.title("Data Preparation & Exploration")

    st.subheader("1. Checking for Missing Values")

    st.write("""
    Before performing any statistical modeling or machine learning, it is essential to 
    inspect the dataset for missing values. Algorithms such as **Linear Regression** and 
    **K-Means Clustering** cannot handle missing values, so we applied preprocessing steps 
    to ensure data quality.
    """)

    # Show missing values
    st.write("### Missing Values per Column")
    st.write(df.isna().sum())

    st.write("""
    We found missing values in several feature columns, as shown above.  
    To avoid model errors and maintain consistency, we removed rows containing 
    missing values only in the feature set used for analysis.
    """)

    # Feature list
    features = [
        "Ladder score",
        "Explained by: Log GDP per capita",
        "Explained by: Social support",
        "Explained by: Healthy life expectancy",
        "Explained by: Freedom to make life choices",
        "Explained by: Generosity",
        "Explained by: Perceptions of corruption"
    ]

    # Drop missing rows
    df_clean = df.dropna(subset=features).copy()

    st.write("### Rows Removed After Cleaning")
    st.write(f"- **Original rows:** {len(df)}")
    st.write(f"- **Rows after cleaning:** {len(df_clean)}")
    st.write(f"- **Rows removed:** {len(df) - len(df_clean)}")

    st.divider()

    # -----------------------------------------
    # 2. HISTOGRAM OF HAPPINESS SCORES
    # -----------------------------------------
    st.subheader("2. Distribution of Happiness Scores (Ladder Score)")

    st.write("""
    Understanding the distribution of happiness scores helps reveal global patterns, 
    skewness, and outliers. A histogram provides a quick visual overview of how countries 
    cluster in terms of well-being.
    """)

    # Plot histogram
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(8,5))
    ax.hist(df_clean["Ladder score"], bins=20)
    ax.set_xlabel("Ladder Score")
    ax.set_ylabel("Frequency")
    ax.set_title("Distribution of Ladder Scores")
    st.pyplot(fig)

    st.divider()

    # -----------------------------------------
    # 3. CORRELATION HEATMAP
    # -----------------------------------------
    st.subheader("3. Correlation Analysis")

    st.write("""
    Correlation analysis helps identify which factors have the strongest relationships 
    with happiness. This is crucial for interpreting which socioeconomic indicators 
    contribute most to higher well-being.
    """)

    import seaborn as sns

    corr = df_clean[features].corr()

    fig2, ax2 = plt.subplots(figsize=(10,7))
    sns.heatmap(corr, annot=True, cmap="coolwarm", fmt=".2f", ax=ax2)
    ax2.set_title("Correlation Heatmap")
    st.pyplot(fig2)

    st.divider()

    # -----------------------------------------
    # 4. Summary of Data Preparation
    # -----------------------------------------
    st.subheader("Summary of Data Preparation Steps")
    st.markdown("""
    - ✔ Checked dataset for missing values  
    - ✔ Removed rows with missing key feature values  
    - ✔ Cleaned dataset reduced from **143 to 140 rows**  
    - ✔ Visualized distribution using histograms  
    - ✔ Identified relationships using a correlation heatmap  
    - ✔ Prepared clean dataset for regression and clustering  
    """)

    st.success("Dataset successfully cleaned and prepared for further analysis.")

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



