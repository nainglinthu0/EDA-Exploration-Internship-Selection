import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.figure_factory as ff

# 1. Global Page Configuration
st.set_page_config(page_title="Internship Selection Analysis", page_icon="🎓", layout="wide")

# Custom CSS for Dark Mode/Black background aesthetic
st.markdown("""
    <style>
    /* Main body background color - Pure Black */
    .stApp { background-color: #000000; color: #ffffff; }
    
    /* Sidebar styling - Dark Charcoal grey with clean borders */
    section[data-testid="stSidebar"] { background-color: #111111; border-right: 1px solid #222222; }
    section[data-testid="stSidebar"] .stMarkdown, section[data-testid="stSidebar"] label { color: #ffffff !important; }
    
    /* Metric Card styling - Charcoal grey background with white text */
    .stMetric { background-color: #111111; padding: 15px; border-radius: 8px; border: 1px solid #333333; box-shadow: 0 1px 3px rgba(255,255,255,0.05); }
    div[data-testid="stMetricValue"] > div { color: #ffffff !important; }
    div[data-testid="stMetricLabel"] > div { color: #aaaaaa !important; }
    
    /* Typography adjustments */
    h1, h2, h3, h4, h5, h6, p, span, label { color: #ffffff !important; }
    
    /* Info box overrides for dark theme */
    div.stAlert { background-color: #111111 !important; color: #ffffff !important; border-color: #444444 !important; }
    div.stAlert p { color: #ffffff !important; }
    </style>
""", unsafe_allow_html=True)

# Helper function to apply dark theme layout template to Plotly charts
def apply_dark_theme(fig):
    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color="#ffffff")
    )
    return fig

# 2. Data Processing Engine
@st.cache_data
def initialize_and_cache_data():
    try:
        df = pd.read_csv('internship_selection_prediction.csv')
    except FileNotFoundError:
        st.error("Operational Error: 'internship_selection_prediction.csv' could not be found.")
        st.stop()
    
    # Feature Engineering Logic from formulas
    df['total_soft_skills'] = df['soft_skills_score'] + df['communication_score']
    df['Technical_Avg'] = df[['skills_score', 'coding_test_score', 'github_score']].mean(axis=1)
    df['Behavioral_Avg'] = df[['communication_score', 'soft_skills_score', 'interview_score']].mean(axis=1)
    
    df['Tech_Category'] = np.where(df['Technical_Avg'] >= df['Technical_Avg'].median(), 'High Tech', 'Low Tech')
    df['Beh_Category'] = np.where(df['Behavioral_Avg'] >= df['Behavioral_Avg'].median(), 'High Beh', 'Low Beh')
    return df

df = initialize_and_cache_data()

# 3. Sidebar Navigation Panel
st.sidebar.title("Internship Selection Menu")
st.sidebar.markdown("Choose a page to explore the data:")
page = st.sidebar.radio(
    "Select Analysis Workspace:",
    [
        "Dataset Glossary and Context",
        "Page 1: Cohort Baseline and Equity",
        "Page 2: Academic Floors and Filters",
        "Page 3: Performance Drivers Matrix",
        "Page 4: Soft Skills and Experience Yield",
        "Page 5: Compensatory Trait Trade-offs"
    ]
)

st.sidebar.markdown("---")
st.sidebar.markdown("### Executive Data Pulse")
st.sidebar.metric("Evaluated Population", f"{len(df):,} Candidates")
st.sidebar.metric("Global Acceptance Rate", f"{(df['selected'].mean() * 100):.1f}%")

# Dataset Glossary Page
if page == "Dataset Glossary and Context":
    st.title("Dataset Overview and Column Reference")
    st.markdown("---")
    st.markdown("### Executive Summary")
    st.write(
        f"This application analyzes a performance tracking matrix containing "
        f"**{len(df):,} student application entries** and **{len(df.columns)} unique data points**. "
        f"The primary goal is to decipher what operational and behavioral characteristics separate selected "
        f"candidates from rejected ones."
    )
    
    st.markdown("---")
    st.markdown("### Data Fields Dictionary")
    
    col_glossary_1, col_glossary_2 = st.columns(2)
    
    with col_glossary_1:
        st.markdown("#### Academic and Background Metrics")
        st.markdown("- **`student_id`**: Unique serial identifier assigned to each tracking candidate record.")
        st.markdown("- **`CGPA`**: Cumulative Grade Point Average scaled continuously between 5.0 and 10.0.")
        st.markdown("- **`college_tier`**: Categorical classification ranking a student's institution (Tier 1, Tier 2, Tier 3).")
        st.markdown("- **`backlogs`**: Integer count of failed or active incomplete academic courses (ranging from 0 to 5).")
        st.markdown("- **`placement_training`**: Binary indicator (Yes/No) representing if the student underwent standard job training.")
        
        st.markdown("#### Technical Capabilities and Scores")
        st.markdown("- **`skills_score`**: Standard core competency technical validation ranking scaled from 1 to 10.")
        st.markdown("- **`coding_test_score`**: Performance grade achieved on timed programming challenges scaled from 1 to 10.")
        st.markdown("- **`github_score`**: Quantitative audit of open-source repository portfolio strength scaled from 1 to 10.")
        st.markdown("- **`projects_count`**: Discrete count of verifiable personal or group engineering projects completed.")
        st.markdown("- **`hackathons_participated`**: Integer total representing how many technical hackathons the student entered.")
        st.markdown("- **`certifications_count`**: Absolute count of certified professional credentials earned by the candidate.")

    with col_glossary_2:
        st.markdown("#### Soft Skills and Behavioral Evaluations")
        st.markdown("- **`communication_score`**: Structural evaluation rating verbal articulation and expression scaled from 1 to 10.")
        st.markdown("- **`soft_skills_score`**: Interpersonal peer interaction rating and emotional intelligence marker scaled from 1 to 10.")
        st.markdown("- **`interview_score`**: Direct performance grade from panels during active technical/behavioral dialogue scaled from 1 to 10.")
        st.markdown("- **`aptitude_score`**: Cognitive screening score validating logical reasoning capabilities scaled from 1 to 10.")
        
        st.markdown("#### Platform Engagement and Portfolio Drivers")
        st.markdown("- **`resume_score`**: Structural parsing grade reflecting automated resume assessment checks scaled from 1 to 10.")
        st.markdown("- **`linkedin_activity_score`**: Algorithmic visibility rating monitoring active professional networking profiles scaled from 1 to 10.")
        st.markdown("- **`consistency_score`**: Tracking indicator logging day-to-day platform study habits scaled from 1 to 10.")
        st.markdown("- **`internships_done`**: Discrete historical total of industry internship cycles completed previously.")
        
        st.markdown("#### Target Variable Outcome")
        st.markdown("- **`selected`**: Final selection choice mapped binary where **`1` indicates Selected** and **`0` indicates Rejected**.")

# Page 1: Cohort Baseline and Equity
elif page == "Page 1: Cohort Baseline and Equity":
    st.title("Cohort Baseline Splits and Institutional Equity Audits")
    st.markdown("---")
    
    st.header("Question 1: Distribution and Selection Balance")
    st.markdown("**Core Analysis Objective:** Check balance of the 'selected' column to evaluate dataset distribution.")
    
    col1_q1, col2_q1 = st.columns([2, 1])
    with col1_q1:
        fig1 = px.pie(
            df, names='selected', 
            title='Balance of Internship Selection (1=Selected, 0=Rejected)',
            hole=0.4,
            color='selected',
            color_discrete_map={0: '#f87171', 1: '#34d399'}
        )
        st.plotly_chart(apply_dark_theme(fig1), use_container_width=True)
    with col2_q1:
        st.markdown("### Analysis:")
        st.write("This shows how competitive the hiring program is by tracking the overall breakdown of selected vs. rejected students.")
        st.markdown("### How to look at it:")
        st.info("Look at the pie chart slices. See if one group is much larger than the other.")
    
    st.markdown("### Conclusion for Question 1:")
    st.write("Out of all the students in our data, around 74% were selected and 26% were rejected. This means selected students make up the clear majority, which is an important pattern to remember when looking at general application trends.")

    st.markdown("---")
    
    st.header("Question 4: College Tier and Institutional Prestige")
    st.markdown("**Core Analysis Objective:** Determine if college tier heavily influences selection or if selection is tier-blind.")
    
    col1_q4, col2_q4 = st.columns([2, 1])
    with col1_q4:
        tier_analysis = df.groupby('college_tier')['selected'].mean().reset_index()
        tier_analysis['selected_pct'] = tier_analysis['selected'] * 100
        fig4 = px.bar(
            tier_analysis, x='college_tier', y='selected_pct',
            title='Selection Rate by College Tier',
            labels={'selected_pct': 'Selection Rate (%)', 'college_tier': 'College Tier'},
            color='college_tier',
            text_auto='.1f',
            color_discrete_sequence=px.colors.qualitative.G10
        )
        st.plotly_chart(apply_dark_theme(fig4), use_container_width=True)
    with col2_q4:
        st.markdown("### Analysis:")
        st.write("This tests fairness. If the bars are roughly the same height, it proves that the company hires based on your individual talent, not just the name of your university.")
        st.markdown("### How to look at it:")
        st.info("Check the tops of the bars. Are the selection percentages across Tier 1, 2, and 3 nearly equal?")
        
    st.markdown("### Conclusion for Question 4:")
    st.write("Students from Tier 1, Tier 2, and Tier 3 universities all have an almost identical chance of being selected. This proves the hiring process is fair and tier-blind; your actual skills and portfolio are what matter, not your school's brand.")

# Page 2: Academic Floors and Filters
elif page == "Page 2: Academic Floors and Filters":
    st.title("Academic Performance Constraints and Disqualification Risks")
    st.markdown("---")
    
    st.header("Question 2: CGPA as a Separator")
    st.markdown("**Core Analysis Objective:** Check average CGPA differences to see how well it separates selected from non-selected students.")
    
    col1_q2, col2_q2 = st.columns([2, 1])
    with col1_q2:
        cgpa_averages = df.groupby('selected')['CGPA'].mean().reset_index()
        cgpa_averages['Status'] = np.where(cgpa_averages['selected'] == 1, 'Selected Candidates', 'Rejected Candidates')
        
        fig2_bar = px.bar(
            cgpa_averages, x='Status', y='CGPA',
            color='Status',
            title='Average College GPA: Selected vs. Rejected Students',
            labels={'CGPA': 'Average CGPA Score'},
            text_auto='.2f',
            color_discrete_map={'Selected Candidates': '#34d399', 'Rejected Candidates': '#f87171'}
        )
        st.plotly_chart(apply_dark_theme(fig2_bar), use_container_width=True)
    with col2_q2:
        st.markdown("### Analysis:")
        st.write("This simple comparison helps discover if accepted students boast a significantly higher academic grading scorecard overall.")
        st.markdown("### How to look at it:")
        st.info("Look at the two bars. Compare the height of the green bar (Selected) directly against the red bar (Rejected) to see the score gap.")
    
    st.markdown("### Conclusion for Question 2:")
    st.write("Selected students maintain a noticeably higher average grade point average compared to rejected students. This confirms that keeping up your academic scores gives you a reliable advantage during the initial recruitment evaluation.")

    st.markdown("---")
    
    st.header("Question 5: The 'Backlog' Threshold Effect")
    st.markdown("**Core Analysis Objective:** Find out if backlogs act as an absolute hard filter for application rejections.")
    
    col1_q5, col2_q5 = st.columns([2, 1])
    with col1_q5:
        backlog_impact = df.groupby('backlogs')['selected'].mean().reset_index()
        backlog_impact['success_rate'] = backlog_impact['selected'] * 100
        fig5 = px.line(
            backlog_impact, x='backlogs', y='success_rate',
            title='The "Backlog Threshold": Success Rate vs. Number of Backlogs',
            markers=True,
            labels={'success_rate': 'Probability of Selection (%)'},
            color_discrete_sequence=['#6366f1']
        )
        st.plotly_chart(apply_dark_theme(fig5), use_container_width=True)
    with col2_q5:
        st.markdown("### Analysis:")
        st.write("This shows how much having failed or active backlogged subjects hurts your job chances.")
        st.markdown("### How to look at it:")
        st.info("Follow the line chart from left to right. Point out where the line drops sharply down toward zero.")
        
    st.markdown("### Conclusion for Question 5:")
    st.write("Having failed subjects is an immediate warning sign for recruiters. As the number of backlogs increases, hiring rates drop dramatically. Having more than 2 backlogs acts as a strict filtering point that makes passing the application stage very unlikely.")

# Page 3: Performance Drivers Matrix
elif page == "Page 3: Performance Drivers Matrix":
    st.title("Statistical Feature Impact and Competency Vectors")
    st.markdown("---")
    
    st.header("Question 3: Linear Predictor Correlation Matrix")
    st.markdown("**Core Analysis Objective:** Run correlation to see which quantitative factors have the highest relationship to selection.")
    
    df_numeric = df.drop(columns=['student_id'], errors='ignore').select_dtypes(include=[np.number])
    corr_matrix = df_numeric.corr()

    fig3 = ff.create_annotated_heatmap(
        z=corr_matrix.values,
        x=list(corr_matrix.columns),
        y=list(corr_matrix.index),
        annotation_text=corr_matrix.round(2).values,
        colorscale='RdBu',
        zmin=-1, zmax=1
    )
    fig3.update_layout(title_text='Correlation Heatmap Matrix', xaxis_tickangle=45)
    st.plotly_chart(apply_dark_theme(fig3), use_container_width=True)
    
    st.markdown("### Analysis:")
    st.write("This ranks every single student trait to see which ones have the biggest statistical impact on final selection.")
    st.markdown("### How to look at it:")
    st.info("Find the 'selected' row or column on the border. Look across to see which boxes show the strongest blue color or highest positive decimals.")
    
    st.markdown("### Conclusion for Question 3:")
    st.write("Practical abilities—like coding test scores and core technical skills—show a much stronger connection to being selected than generic variables like college tier. This tells us that verifiable, real-world skills carry more weight than institutional labels.")

    st.markdown("---")
    
    st.header("Question 6: Success Profiles and Target Performance Averages")
    st.markdown("**Core Analysis Objective:** Group metrics by selected status to evaluate the average performance score differences.")
    
    col1_q6, col2_q6 = st.columns([2, 1])
    with col1_q6:
        metrics_list = ['interview_score', 'coding_test_score', 'skills_score', 'aptitude_score']
        success_profile = df.groupby('selected')[metrics_list].mean().reset_index()
        success_profile_melted = success_profile.melt(id_vars='selected', var_name='Metric', value_name='Average Score')

        fig6 = px.bar(
            success_profile_melted, x='Metric', y='Average Score',
            color='selected', barmode='group',
            title='Average Performance Scores: Selected vs. Rejected Students',
            color_discrete_map={0: '#f87171', 1: '#4f46e5'}
        )
        st.plotly_chart(apply_dark_theme(fig6), use_container_width=True)
    with col2_q6:
        st.markdown("### Analysis:")
        st.write("This highlights the performance gaps. The larger the distance between the two bars, the more that skill matters in deciding who gets hired.")
        st.markdown("### How to look at it:")
        st.info("Compare the heights of the side-by-side purple (Selected) and red (Rejected) bars for each score. See which skill has the largest vertical gap.")
        
    st.markdown("### Conclusion for Question 6:")
    st.write("Selected candidates outperform rejected ones across all test categories. The absolute largest performance gaps appear in interview scores and coding tests, meaning these two specific skills serve as the primary filters for a successful profile.")

# Page 4: Soft Skills and Experience Yield
elif page == "Page 4: Soft Skills and Experience Yield":
    st.title("Behavioral Synergies and Practical Experience Boundaries")
    st.markdown("---")
    
    st.header("Question 7: Qualitative Soft Skill Synergy Distribution")
    st.markdown("**Core Analysis Objective:** Check if the combination of communication and soft skills builds an interactive synergy effect.")
    
    col1_q7, col2_q7 = st.columns([2, 1])
    with col1_q7:
        df['Selection Status'] = np.where(df['selected'] == 1, 'Selected', 'Rejected')
        
        fig7_hist = px.histogram(
            df, x='total_soft_skills', color='Selection Status',
            nbins=15, barmode='group',
            title='Count of Students by Combined Interpersonal and Communication Scores',
            labels={'total_soft_skills': 'Combined Score Threshold (Soft Skills + Communication Score)'},
            color_discrete_map={'Selected': '#6366f1', 'Rejected': '#f87171'}
        )
        st.plotly_chart(apply_dark_theme(fig7_hist), use_container_width=True)
    with col2_q7:
        st.markdown("### Analysis:")
        st.write("This tracks how student populations cluster. It checks if having high total soft-skills volumes completely separates the selected candidates.")
        st.markdown("### How to look at it:")
        st.info("Look at the horizontal axis from left to right. Notice how the purple bars (Selected) grow massive and dominate the high-score side, while red bars are trapped on the left.")
    
    st.markdown("### Conclusion for Question 7:")
    st.write("The chart shows a distinct trend: the selected student groups are heavily bunched up on the upper end of the behavioral score matrix. This proves that high blended communication and soft-skill talent acts as a mandatory baseline floor to succeed.")

    st.markdown("---")
    
    st.header("Question 8: Practical Experience Diminishing Yields")
    st.markdown("**Core Analysis Objective:** Evaluate selection trends across the count of completed internships.")
    
    col1_q8, col2_q8 = st.columns([2, 1])
    with col1_q8:
        exp_impact = df.groupby('internships_done')['selected'].mean().reset_index()
        fig8 = px.bar(
            exp_impact, x='internships_done', y='selected',
            title='Selection Probability based on Internships Completed',
            labels={'selected': 'Selection Rate', 'internships_done': 'Internships Completed'},
            color='selected', color_continuous_scale='Blues'
        )
        st.plotly_chart(apply_dark_theme(fig8), use_container_width=True)
    with col2_q8:
        st.markdown("### Analysis:")
        st.write("This tests if more experience is always better, or if having just 1 or 2 past internships is enough to maximize your chances.")
        st.markdown("### How to look at it:")
        st.info("Look at the steps of the bars from left to right. Identify where the bar heights stop jumping up significantly.")
        
    st.markdown("### Conclusion for Question 8:")
    st.write("Having past internship experience definitely increases selection chances. However, the benefits level off noticeably after completing 1 to 2 internships. This shows that while getting baseline experience is essential, piling on additional internships gives diminishing returns compared to improving your actual skill scores.")

# Page 5: Compensatory Trait Trade-offs
elif page == "Page 5: Compensatory Trait Trade-offs":
    st.title("Strategic Compensatory Frameworks and Decision Boundaries")
    st.markdown("---")
    
    st.header("Question 9: The Grade-Personality 'Offset' Calibration Grid")
    st.markdown("**Core Analysis Objective:** Determine if a high interview score can compensate for a weaker academic CGPA ranking.")
    
    fig9 = px.density_heatmap(
        df, x="CGPA", y="interview_score", z="selected", histfunc="avg",
        nbinsx=10, nbinsy=10,
        title="Selection Probability: CGPA vs. Interview Score Matrix",
        labels={'selected': 'Selection Prob'},
        color_continuous_scale='Viridis', text_auto='.2f'
    )
    st.plotly_chart(apply_dark_theme(fig9), use_container_width=True)
    
    st.markdown("### Analysis:")
    st.write("This checks if a fantastic interview performance can rescue a student who has a below-average GPA.")
    st.markdown("### How to look at it:")
    st.info("Look closely at the upper-left area of the grid grid (High Interview Score matched with Low CGPA scores). Is the color bright and showing high selection numbers?")
    
    st.markdown("### Conclusion for Question 9:")
    st.write("The heatmap shows strong selection rates in the top-left area (representing high interview marks paired with moderate-to-low CGPAs). This proves that an excellent interview can balance out and save an applicant who has weaker academic grades.")

    st.markdown("---")
    
    st.header("Question 10: Behavioral vs. Technical Influence (The Ultimate Tie-Breaker)")
    st.markdown("**Core Analysis Objective:** Group candidates into matrix quadrants to identify the single most impactful category.")
    
    col1_q10, col2_q10 = st.columns([2, 1])
    with col1_q10:
        combination_analysis = df.groupby(['Tech_Category', 'Beh_Category'], observed=False)['selected'].mean().reset_index()
        combination_analysis['success_rate'] = combination_analysis['selected'] * 100

        fig10 = px.bar(
            combination_analysis, x='Tech_Category', y='success_rate',
            color='Beh_Category', barmode='group',
            title='Selection Probability by Technical and Behavioral Skill Level Quadrant Profiles',
            labels={'success_rate': 'Selection Probability (%)'},
            text_auto='.1f',
            color_discrete_sequence=['#4f46e5', '#f43f5e']
        )
        st.plotly_chart(apply_dark_theme(fig10), use_container_width=True)
    with col2_q10:
        st.markdown("### Analysis:")
        st.write("This acts as the ultimate tie-breaker guide. It tells students whether they should focus their short-term preparation on technical coding skills or soft personality traits.")
        st.markdown("### How to look at it:")
        st.info("Compare the 'High Tech / Low Beh' bar directly against the 'Low Tech / High Beh' bar to see which profile wins the tie-breaker.")
        
    st.markdown("### Conclusion for Question 10:")
    st.write("While having high scores in both technical and behavioral categories yields the best absolute chance of getting hired, technical mastery holds a small advantage over behavioral soft skills when resolving tie-breakers. To achieve the highest chances of success, students should build a solid technical foundation first, and then refine their behavioral interview prep.")