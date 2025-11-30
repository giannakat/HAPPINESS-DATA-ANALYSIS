import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.cluster import KMeans
import numpy as np
from streamlit_plotly_events import plotly_events
import plotly.express as px

# styles
with open("styles/main.css") as f:
    st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

#load dataset (world-happiness-report.csv)
df = pd.read_csv("WHR2024.csv")

tabs = st.tabs(["Overview", "Dataset", "Exploration", "Analysis", "Conclusions"])

with tabs[0]:
    st.markdown("""
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
        <div class="custom-box" style='margin-bottom: 10px;'>
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
        <div class="custom-box"'>
            <b>Descriptive Analysis</b><br>
            Summarizes the distribution of happiness scores and key indicators.
        </div>
        """, unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)

        st.markdown("""
        <div class="custom-box">
            <b>Correlation Analysis</b><br>
            Examines relationships between happiness scores and related variables.
        </div>
        """, unsafe_allow_html=True)

    with col2:
        st.markdown("""
        <div class="custom-box">
            <b>Multiple Linear Regression</b><br>
            Models how predictors such as GDP and social support influence happiness.
        </div>
        """, unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)

        st.markdown("""
        <div class="custom-box">
            <b>Cluster Analysis</b><br>
            Groups countries with similar happiness profiles (e.g., via K-Means).
        </div>
        """, unsafe_allow_html=True)


    st.write("")
    st.write("")

    st.divider()

    st.subheader("Expected Outcome")

    st.markdown("""
    <div class="custom-box" style='text-align: justify;'>
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


# ANALYSIS & INSIGHTS SECTION
with tabs[3]:
    st.title("Analysis & Insights")

    st.markdown("---")

    subtabs = st.tabs(["Analysis", "World"])

    with subtabs[0]:
        st.markdown("---")

        # -------------------------------
        # 1. Happiness Filter
        # -------------------------------
        st.subheader("Happiness Filter")
        min_score, max_score = st.slider(
            "Happiness Score Range",
            float(df_clean["Ladder score"].min()),
            float(df_clean["Ladder score"].max()),
            (float(df_clean["Ladder score"].min()), float(df_clean["Ladder score"].max())),
            key="analysis"
        )

        # Apply filters
        df_filtered = df_clean.copy()
        df_filtered = df_filtered[(df_filtered["Ladder score"] >= min_score) & (df_filtered["Ladder score"] <= max_score)]
    
        st.markdown("> TIP: Use the slider to select a range of happiness scores and filter out countries!")

        # -------------------------------
        # 2. Interactive Scatter Plots
        # -------------------------------
        st.markdown("---")
        
        st.subheader("Interactive Scatter Plot")
        factor_dict = {
            "GDP per capita": "Explained by: Log GDP per capita",
            "Social Support": "Explained by: Social support",
            "Life Expectancy": "Explained by: Healthy life expectancy",
            "Freedom": "Explained by: Freedom to make life choices",
            "Generosity": "Explained by: Generosity",
            "Corruption": "Explained by: Perceptions of corruption"
        }

        selected_factor = st.selectbox("Select Factor for Scatter Plot:", list(factor_dict.keys()))
        col_name = factor_dict[selected_factor]

        fig_scatter = px.scatter(
            df_filtered,
            x=col_name,
            y="Ladder score",
            hover_name="Country name",
            hover_data={
                "Ladder score": True,
                col_name: True,
                "Explained by: Social support": True,
                "Explained by: Healthy life expectancy": True,
                "Explained by: Freedom to make life choices": True,
                "Explained by: Generosity": True,
                "Explained by: Perceptions of corruption": True
            },
            title=f"Happiness vs {selected_factor}",
        )

        # Regression lines
        X = df_filtered[[col_name]].values
        y = df_filtered["Ladder score"].values
        model = LinearRegression()
        model.fit(X, y)
        df_filtered['pred'] = model.predict(X)
        fig_scatter.add_traces(
            px.line(df_filtered, x=col_name, y='pred').data
        )

        st.plotly_chart(fig_scatter, use_container_width=True)
        st.markdown("> TIP: Hover on each dot to see country details.")

        # -------------------------------
        # 3. Correlation Heatmap
        # -------------------------------
        st.markdown("---")
        st.subheader("Correlation Heatmap")
        features = [
            "Ladder score",
            "Explained by: Log GDP per capita",
            "Explained by: Social support",
            "Explained by: Healthy life expectancy",
            "Explained by: Freedom to make life choices",
            "Explained by: Generosity",
            "Explained by: Perceptions of corruption"
        ]
        corr = df_filtered[features].corr()

        fig2, ax2 = plt.subplots(figsize=(10,7))
        sns.heatmap(corr, annot=True, cmap="coolwarm", fmt=".2f", ax=ax2)
        st.pyplot(fig2)
        st.markdown("> This heatmap shows correlations in the filtered dataset. GDP, Social Support, and Life Expectancy are strongest. Freedom, Generosity, and Corruption show weaker or inconsistent patterns.")

        # -------------------------------
        # 4. K-Means Cluster Map
        # -------------------------------
        st.markdown("---")
        st.subheader("Interactive Cluster Map (K-Means)")
        k = st.slider("Select number of clusters (k):", 2, 6, 3)
        kmeans_features = features[1:]  # exclude Ladder score
        kmeans_model = KMeans(n_clusters=k, random_state=42)
        df_filtered['Cluster'] = kmeans_model.fit_predict(df_filtered[kmeans_features])

        st.markdown("> TIP: Use this slider to choose how many clusters to divide countries into. Hover on each dot to see country details.")

        if 'Country name' in df_filtered.columns:
            fig3 = px.scatter_geo(
                df_filtered,
                locations="Country name",
                locationmode="country names",
                color="Cluster",
                hover_name="Country name",
                hover_data=["Ladder score"] + kmeans_features,
                title=f"Happiness Clusters (k = {k})",
                projection="natural earth"
            )
            st.plotly_chart(fig3, use_container_width=True)

        st.markdown("> Clusters group countries by overall well-being: middle-range happiness dominates globally, high-income countries cluster together, regional differences visible (Europe high, Africa low, Asia mixed).")

        # -------------------------------
        # 5. Key Insights & Patterns
        # -------------------------------
        st.markdown("---")
        st.subheader("Key Insights & Patterns")

        st.markdown("- **GDP per capita, Social Support, Life Expectancy** are the strongest factors for happiness.")
        with st.expander("Why this matters"):
            st.markdown("""
            - **Switzerland, Norway, Denmark**: high GDP, strong social support, long healthy lives → high happiness.
            """)

        st.markdown("- **Freedom & Generosity** have weaker links to happiness.")
        with st.expander("Why this matters"):
            st.markdown("""
            - **Costa Rica**: high generosity but moderate happiness.
            - **Japan**: high freedom but average happiness.
            """)

        st.markdown("- **Factor effects change with happiness level**")
        with st.expander("Why this matters"):
            st.markdown("""
            - Countries with low happiness: factors like GDP, freedom, and support → barely make a difference anymore, very small impact.
            - Countries with high happiness: these factors → much stronger, really boost happiness.
            - General trend: the happier a country is, the more these factors matter; the sadder a country is, the less they matter.
            """)


        st.markdown("- **Corruption scores** sometimes don’t match happiness.")
        with st.expander("Why this matters"):
            st.markdown("""
            - **Singapore**: some perceived corruption but high happiness.
            """)

        st.markdown("- **Clusters** group countries by income and happiness.")
        with st.expander("Why this matters"):
            st.markdown("""
            - High-income: **Switzerland, Norway, Singapore** → high happiness.
            - Middle-income: **Mexico, Thailand, Brazil** → medium happiness.
            - Low-income: **Chad, Sierra Leone, Afghanistan** → lower happiness.
            """)

        st.markdown("- **Regions differ**: Europe = high, Africa = low, Asia = mixed.")
        with st.expander("Why this matters"):
            st.markdown("""
            - Europe: **Finland, Denmark** → high happiness.
            - Africa: **Nigeria, Kenya** → lower happiness.
            - Asia: **Japan, South Korea** → high GDP but mixed happiness; **India, Pakistan** → lower scores.
            """)

        st.markdown("- **Outliers** exist in scatter plots.")
        with st.expander("Why this matters"):
            st.markdown("""
            - **USA**: very high GDP, moderate happiness.
            - **Bhutan**: lower GDP, relatively high happiness.
            """)

        st.markdown("- **Two factors together matter more**")
        with st.expander("Why this matters"):
            st.markdown("""
            - **Norway**: high GDP + strong social support = highest happiness.
            - **Iceland**: healthy life expectancy also high → higher happiness.
            """)

        st.markdown("- **Most countries are in the middle range**")
        with st.expander("Why this matters"):
            st.markdown("""
            - **Spain, Italy, Brazil** → middle-range happiness scores.
            - Extremes: **Finland** (high); **Chad**, **Afghanistan** (low).
            """)

        st.markdown("- **Small countries with high generosity can be happy**")
        with st.expander("Why this matters"):
            st.markdown("""
            - **Costa Rica, New Zealand** → high generosity + social cohesion boosts happiness.
            """)

        st.markdown("- **Freedom matters more in some regions**")
        with st.expander("Why this matters"):
            st.markdown("""
            - Europe: **Sweden, Denmark** → stronger link between freedom and happiness.
            - Asia/Africa: less effect; culture/government style may reduce impact.
            """)

    with subtabs[1]:
        # -------------------------------
        # Interactive World Map with Click Info
        # -------------------------------

        st.markdown("---")
        st.subheader("Factor Filters")

        # Sliders for filtering factors
        min_happiness, max_happiness = st.slider(
            "Happiness Score Range",
            float(df_clean["Ladder score"].min()),
            float(df_clean["Ladder score"].max()),
            (float(df_clean["Ladder score"].min()), float(df_clean["Ladder score"].max())),
            key="world"
        )
        
        min_gdp, max_gdp = st.slider(
            "GDP per Capita Range",
            float(df_clean["Explained by: Log GDP per capita"].min()),
            float(df_clean["Explained by: Log GDP per capita"].max()),
            (float(df_clean["Explained by: Log GDP per capita"].min()), float(df_clean["Explained by: Log GDP per capita"].max()))
        )

        min_support, max_support = st.slider(
            "Social Support Range",
            float(df_clean["Explained by: Social support"].min()),
            float(df_clean["Explained by: Social support"].max()),
            (float(df_clean["Explained by: Social support"].min()), float(df_clean["Explained by: Social support"].max()))
        )

        min_life, max_life = st.slider(
            "Life Expectancy Range",
            float(df_clean["Explained by: Healthy life expectancy"].min()),
            float(df_clean["Explained by: Healthy life expectancy"].max()),
            (float(df_clean["Explained by: Healthy life expectancy"].min()), float(df_clean["Explained by: Healthy life expectancy"].max()))
        )

        min_freedom, max_freedom = st.slider(
            "Freedom Range",
            float(df_clean["Explained by: Freedom to make life choices"].min()),
            float(df_clean["Explained by: Freedom to make life choices"].max()),
            (float(df_clean["Explained by: Freedom to make life choices"].min()), float(df_clean["Explained by: Freedom to make life choices"].max()))
        )

        min_gen, max_gen = st.slider(
            "Generosity Range",
            float(df_clean["Explained by: Generosity"].min()),
            float(df_clean["Explained by: Generosity"].max()),
            (float(df_clean["Explained by: Generosity"].min()), float(df_clean["Explained by: Generosity"].max()))
        )

        min_corr, max_corr = st.slider(
            "Corruption Range",
            float(df_clean["Explained by: Perceptions of corruption"].min()),
            float(df_clean["Explained by: Perceptions of corruption"].max()),
            (float(df_clean["Explained by: Perceptions of corruption"].min()), float(df_clean["Explained by: Perceptions of corruption"].max()))
        )

        # Apply all filters
        df_filtered = df_clean[
            (df_clean["Ladder score"] >= min_happiness) & (df_clean["Ladder score"] <= max_happiness) &
            (df_clean["Explained by: Log GDP per capita"] >= min_gdp) & (df_clean["Explained by: Log GDP per capita"] <= max_gdp) &
            (df_clean["Explained by: Social support"] >= min_support) & (df_clean["Explained by: Social support"] <= max_support) &
            (df_clean["Explained by: Healthy life expectancy"] >= min_life) & (df_clean["Explained by: Healthy life expectancy"] <= max_life) &
            (df_clean["Explained by: Freedom to make life choices"] >= min_freedom) & (df_clean["Explained by: Freedom to make life choices"] <= max_freedom) &
            (df_clean["Explained by: Generosity"] >= min_gen) & (df_clean["Explained by: Generosity"] <= max_gen) &
            (df_clean["Explained by: Perceptions of corruption"] >= min_corr) & (df_clean["Explained by: Perceptions of corruption"] <= max_corr)
        ]

        st.markdown("<span style='color:red;'><b><i><u>DISCLAIMER: The sliders are not absolute measures of actual income, social support, life expectancy, freedom, generosity, and corruption. They only show how much each factor affects a country's happiness relative to others. Actual values may differ in real life.<u></i></b></span>", unsafe_allow_html=True)


        st.markdown("---")
        st.subheader("World Map")

        # Simple description data for click
        df_map = df_filtered.copy()
        df_map['Short Description'] = df_map.apply(lambda row: "\n".join([
            f"- **Happiness Score**: {row['Ladder score']:.3f} → {'high' if row['Ladder score'] > df_map['Ladder score'].mean() else 'low'} (relative to global average).",
            f"- **GDP per Capita Impact**: {row['Explained by: Log GDP per capita']:.3f} → {'higher impact on happiness' if row['Explained by: Log GDP per capita'] > df_map['Explained by: Log GDP per capita'].mean() else 'lower impact on happiness'}",
            f"- **Social Support Impact**: {row['Explained by: Social support']:.3f} → {'higher impact on happiness' if row['Explained by: Social support'] > df_map['Explained by: Social support'].mean() else 'lower impact on happiness'}",
            f"- **Life Expectancy Impact**: {row['Explained by: Healthy life expectancy']:.3f} → {'higher impact on happiness' if row['Explained by: Healthy life expectancy'] > df_map['Explained by: Healthy life expectancy'].mean() else 'lower impact on happiness'}",
            f"- **Freedom Impact**: {row['Explained by: Freedom to make life choices']:.3f} → {'higher impact on happiness' if row['Explained by: Freedom to make life choices'] > df_map['Explained by: Freedom to make life choices'].mean() else 'lower impact on happiness'}",
            f"- **Generosity Impact**: {row['Explained by: Generosity']:.3f} → {'higher impact on happiness' if row['Explained by: Generosity'] > df_map['Explained by: Generosity'].mean() else 'lower impact on happiness'}",
            f"- **Corruption Impact**: {row['Explained by: Perceptions of corruption']:.3f} → {'higher impact on happiness' if row['Explained by: Perceptions of corruption'] > df_map['Explained by: Perceptions of corruption'].mean() else 'lower impact on happiness'}"
        ]), axis=1)


        # World map plotly
        min_score = df_map["Ladder score"].min()
        max_score = df_map["Ladder score"].max()

        # create 5 ticks including min and max
        tick_vals = np.linspace(min_score, max_score, 5)
        tick_text = [f"{v:.2f}" for v in tick_vals]
        tick_text[0] = f"Lowest ({tick_text[0]})"
        tick_text[-1] = f"Highest ({tick_text[-1]})"

        fig_map = px.choropleth(
            df_map,
            locations="Country name",
            locationmode="country names",
            color="Ladder score",
            labels={"Ladder score": "Happiness Score"},
            hover_name="Country name",
            hover_data={
                "Ladder score": True,
                "Explained by: Log GDP per capita": True,
                "Explained by: Social support": True,
                "Explained by: Healthy life expectancy": True,
                "Explained by: Freedom to make life choices": True,
                "Explained by: Generosity": True,
                "Explained by: Perceptions of corruption": True,
            },
            color_continuous_scale="Viridis",
            title="Click on a country to see details",
        )
        fig_map.update_layout(
            margin={"r":0,"t":50,"l":0,"b":0},  # remove extra margins
            height=450,
            coloraxis_colorbar=dict(
                title="Happiness Score",
                tickvals=tick_vals,
                ticktext=tick_text
            )
        )
        fig_map.update_traces(
            hovertemplate=
            "<b>%{location}</b><br><br>" +
            "Happiness Score: %{customdata[0]:.3f}<br><br>" +
            "GDP per Capita: %{customdata[1]:.3f}<br>" +
            "Social Support: %{customdata[2]:.3f}<br>" +
            "Life Expectancy: %{customdata[3]:.3f}<br>" +
            "Freedom: %{customdata[4]:.3f}<br>" +
            "Generosity: %{customdata[5]:.3f}<br>" +
            "Corruption: %{customdata[6]:.3f}<extra></extra>"
        )

        st.plotly_chart(fig_map, use_container_width=True) # display map

        # country selector
        country_info = st.empty()

        country_list = df_map["Country name"].sort_values().tolist()
        clicked_country = st.selectbox("Select / type a country to see details:", ["None"] + country_list)

        if clicked_country != "None":
            info_row = df_map[df_map["Country name"] == clicked_country].iloc[0]
            st.markdown(f"**{clicked_country} Details:**")
            st.markdown(info_row["Short Description"])



#conclusion and recommendations
with tabs[4]:
    st.markdown("""
    <br>
    <h1 style="font-size:2.2rem; font-weight:700; text-align:center;">
    <span class="global-happiness" style="color:#FDB12A; font-size:2.8rem;">Conclusions & Recommendations</span><br> 
    </h1>
    """, unsafe_allow_html=True)

    st.markdown("### A Look Back at What We Discovered")

    st.markdown("""
    Based on the analysis conducted:

    - Happiness tends to settle around the global “middle ground,” showing diverse yet 
      balanced distribution across countries.
    - Variables such as **GDP per capita, social support, and healthy life expectancy**
        displayed the strongest relationships with happiness scores.
    - Among all factors, **GDP per capita** shines the brightest in its connection to 
      happiness, highlighting how financial stability supports overall well-being.
    - Cluster analysis suggested that countries can be grouped based on **socioeconomic
        and well-being indicators**, showing clear regional or developmental patterns.
    """)

    st.divider()

    st.markdown("### Dive Deeper Into an Insight")

    insight_option = st.selectbox(
        "Choose a perspective to explore:",
        [
            "Overall Conclusion",
            "Strongest Correlation",
            "Cluster Pattern",
            "Limitations of the Analysis",
        ]
    )

    insight_texts = {
        "Overall Conclusion":
            "The findings emphasize a simple truth: societies flourish when their people are "
            "supported economically, socially, and health-wise. Happiness isn’t random,it reflects "
            "the environment people live in.",
        
        "Strongest Correlation":
            "GDP per capita stands out as the most influential factor. Countries with stronger "
            "economic foundations tend to foster happier populations, suggesting that financial "
            "stability provides both comfort and opportunity.",

        "Cluster Pattern":
            "Countries don’t exist in isolation, and neither do their happiness levels. Cluster "
            "analysis shows that nations with similar socioeconomic backgrounds naturally group "
            "together, reflecting shared development paths and cultural dynamics.",

        "Limitations of the Analysis":
            "While the dataset provides valuable insights, the removal of missing entries may have "
            "slightly narrowed the representation. Additionally, happiness is shaped by many subtle "
            "factors that were not fully captured here, such as culture, values, and mental health."
    }

    st.text_area("Insight Details", insight_texts[insight_option], height=150)

    st.divider()

    st.markdown("### Recommendations Moving Forward")

    rec_category = st.selectbox(
        "Select a recommendation focus:",
        [
            "For Policymakers",
            "For Researchers",
            "For Future Data Collection",
        ]
    )

    recommendations = {
        "For Policymakers":
            "Strengthen the foundations that consistently elevate happiness such as reliable healthcare, "
            "accessible social support, and economic pathways that give citizens a sense of security "
            "and possibility.",

        "For Researchers":
            "Happiness is complex, to understand it more deeply, future research should explore "
            "additional angles such as cultural norms, psychological well-being, environmental "
            "factors, and education-related indicators.",

        "For Future Data Collection":
            "Broader and more complete data can unlock more accurate stories. Improving reporting "
            "quality, reducing missing values, and gathering more behavioral and cultural metrics "
            "will enrich future analyses."
    }

    st.text_area("Recommendation Details", recommendations[rec_category], height=150)

    st.divider()

    st.subheader("Final Thoughts")

    st.write(
    """
    Happiness is more than a number — it is a reflection of how effectively countries nurture 
    the lives of their people. By understanding the patterns, connections, and limitations 
    uncovered in this analysis, we move closer to building societies where well-being is not 
    just an aspiration, but an achievable reality for all.
    """
    )



