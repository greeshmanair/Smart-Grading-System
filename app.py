import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Load datasets
mcq_df = pd.read_excel("mcq_machine_learning.xlsx", sheet_name="mcq_machine_learning")
subjective_df = pd.read_excel("subjective_questions_ml.xlsx", sheet_name="Sheet1")
performance_df = pd.read_excel("student_performance_dataset_large.xlsx", sheet_name="Sheet1")

# Session state to track progress
if 'mcq_score' not in st.session_state:
    st.session_state['mcq_score'] = None
if 'subjective_scores' not in st.session_state:
    st.session_state['subjective_scores'] = None
if 'coding_score' not in st.session_state:
    st.session_state['coding_score'] = None
if 'current_page' not in st.session_state:
    st.session_state['current_page'] = "Home"

# Function to navigate pages
def navigate(page):
    st.session_state['current_page'] = page

# Function to grade MCQs
def grade_mcq(student_answers):
    correct_answers = mcq_df['Correct Answer'].values
    score = sum([1 for i in range(len(correct_answers)) if student_answers[i] == correct_answers[i]]) / len(correct_answers) * 100
    return score

# Function to grade Subjective answers
def grade_subjective(student_answers):
    sample_answers = subjective_df['Sample Answer'].dropna().values
    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform(sample_answers.tolist() + student_answers)
    similarity_scores = cosine_similarity(tfidf_matrix[-len(student_answers):], tfidf_matrix[:-len(student_answers)])
    scores = [max(sim) * 10 for sim in similarity_scores]  # Scale to 10
    return scores

# UI Navigation
st.sidebar.button("MCQ", on_click=navigate, args=("MCQ",))
st.sidebar.button("Subjective", on_click=navigate, args=("Subjective",))
st.sidebar.button("Coding", on_click=navigate, args=("Coding",))
st.sidebar.button("Scorecard", on_click=navigate, args=("Scorecard",))

# MCQ Page
if st.session_state['current_page'] == "MCQ":
    st.header("MCQ Section")
    student_mcq_answers = []
    for i, row in mcq_df.iterrows():
        options = [row['Option A'], row['Option B'], row['Option C'], row['Option D']]
        answer = st.radio(row['Question'], options, key=f"mcq_{i}")
        student_mcq_answers.append(chr(97 + options.index(answer)))
    if st.button("Submit MCQs"):
        st.session_state['mcq_score'] = grade_mcq(student_mcq_answers)
        st.success(f"Your MCQ Score: {st.session_state['mcq_score']:.2f}%")

# Subjective Page
elif st.session_state['current_page'] == "Subjective":
    st.header("Subjective Questions")
    responses = []
    for index, row in subjective_df.iterrows():
        response = st.text_area(f"{index+1}. {row['Question']}", "")
        responses.append(response)
    if st.button("Submit Subjective Answers"):
        st.session_state['subjective_scores'] = grade_subjective(responses)
        st.success("Subjective answers graded successfully!")

# Coding Page
elif st.session_state['current_page'] == "Coding":
    st.header("Coding Assessment")
    uploaded_file = st.file_uploader("Upload Python Code", type=["py", "ipynb"])
    if uploaded_file and st.button("Submit Code"):
        # Simulate grading (replace with actual function)
        st.session_state['coding_score'] = 85  # Dummy score
        st.success("Code evaluated successfully!")

# Scorecard Page
elif st.session_state['current_page'] == "Scorecard":
    st.header("Scorecard")
    if None in [st.session_state['mcq_score'], st.session_state['subjective_scores'], st.session_state['coding_score']]:
        st.warning("Complete all sections to view the scorecard!")
    else:
        scores = {
            "MCQ": st.session_state['mcq_score'],
            "Subjective": sum(st.session_state['subjective_scores']) / len(st.session_state['subjective_scores']),
            "Coding": st.session_state['coding_score']
        }
        df_scores = pd.DataFrame(scores.items(), columns=["Section", "Score"])
        st.table(df_scores)
        
        # Visualization
        fig, ax = plt.subplots()
        sns.barplot(x=df_scores["Section"], y=df_scores["Score"], palette="coolwarm", ax=ax)
        plt.ylim(0, 100)
        st.pyplot(fig)

st.sidebar.write("Complete all sections to unlock Scorecard!")
