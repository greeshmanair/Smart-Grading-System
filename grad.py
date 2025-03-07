import streamlit as st
import pandas as pd
import torch
from transformers import BertTokenizer, BertModel
import torch.nn.functional as F
import subprocess
import tempfile
import matplotlib.pyplot as plt
import seaborn as sns

# Load BERT tokenizer and model only once
@st.cache_resource
def load_model():
    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
    model = BertModel.from_pretrained("bert-base-uncased")
    return tokenizer, model

tokenizer, model = load_model()

# Load subjective questions from file
def load_subjective_questions():
    return pd.read_excel("subjective_questions_ml.xlsx")

subjective_df = load_subjective_questions()

# Load MCQ questions from file
def load_mcq_questions():
    return pd.read_excel("mcq_machine_learning.xlsx")

mcq_df = load_mcq_questions()

# Compute similarity score
def compute_similarity(answer, reference_answer):
    if not answer.strip():
        return 0.0  # If empty response, return lowest score

    inputs = tokenizer([answer, reference_answer], padding=True, truncation=True, return_tensors="pt")
    with torch.no_grad():
        outputs = model(**inputs)

    answer_embedding = outputs.last_hidden_state.mean(dim=1)[0]
    reference_embedding = outputs.last_hidden_state.mean(dim=1)[1]

    similarity_score = F.cosine_similarity(answer_embedding.unsqueeze(0), reference_embedding.unsqueeze(0)).item()
    return round(similarity_score, 2)

# Generate feedback
def generate_feedback(similarity_score):
    if similarity_score > 0.85:
        return "✅ Excellent answer! Well-structured and relevant."
    elif similarity_score > 0.65:
        return "🟡 Good attempt, but needs more clarity and depth."
    else:
        return "❌ Needs improvement. Focus on key points and provide a structured response."

# Store section-wise results
st.session_state.setdefault("mcq_score", None)
st.session_state.setdefault("subjective_scores", [])
st.session_state.setdefault("coding_score", None)

# Streamlit UI
st.title("📚 Smart Grading & Feedback System")

# Section navigation
section = st.sidebar.radio("Select Section", ["MCQ", "Subjective", "Coding", "Scorecard"])

if section == "MCQ":
    st.write("### Answer the MCQs Below:")
    mcq_responses = {}

    for index, row in mcq_df.iterrows():
        question = row["Question"]
        options = [row["Option A"], row["Option B"], row["Option C"], row["Option D"]]
        correct_answer = row["Correct Answer"]
        selected_option = st.radio(f"{index+1}. {question}", options, key=f"mcq_{index}")
        mcq_responses[question] = (selected_option, correct_answer)

    if st.button("🔍 Evaluate MCQs"):
        mcq_score = sum(1 for q, (s, c) in mcq_responses.items() if s == c) / len(mcq_responses) * 100 if mcq_responses else 0
        st.session_state["mcq_score"] = mcq_score
        st.success(f"MCQ Score: {mcq_score:.2f}%")

if section == "Subjective":
    st.write("### Answer the Questions Below:")
    responses = {}

    for index, row in subjective_df.iterrows():
        question = row["Question"]
        reference_answer = row["Sample Answer"]
        student_response = st.text_area(f"{index+1}. {question}", "")
        responses[question] = (student_response, reference_answer)

    if st.button("🔍 Evaluate Answers"):
        subjective_scores = [compute_similarity(s, r) * 100 for s, r in responses.values()]
        st.session_state["subjective_scores"] = subjective_scores
        st.success("Subjective answers evaluated!")

if section == "Coding":
    st.write("### Write a code to find the factorial of 5")
    uploaded_file = st.file_uploader("Upload Python Code", type=["py", "ipynb"])
    if uploaded_file and st.button("Submit Code"):
        with tempfile.NamedTemporaryFile(delete=False, suffix=".py") as temp_file:
            temp_file.write(uploaded_file.read())
            temp_file_path = temp_file.name

        try:
            result = subprocess.run(["python", temp_file_path], capture_output=True, text=True, timeout=5)
            if result.returncode == 0 and result.stdout.strip() == "120":
                coding_score = 100
                feedback = "✅ Correct output! Well done."
            else:
                coding_score = 50
                feedback = f"❌ Incorrect output. Expected 120, but got {result.stdout.strip()}"
        except Exception as e:
            coding_score = 0
            feedback = f"❌ Failed to execute. Error: {str(e)}"

        st.session_state["coding_score"] = coding_score
        st.success(feedback)

if section == "Scorecard":
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

        fig, ax = plt.subplots()
        sns.barplot(x=df_scores["Section"], y=df_scores["Score"], palette="coolwarm", ax=ax)
        plt.ylim(0, 100)
        st.pyplot(fig)

st.sidebar.write("Complete all sections to unlock Scorecard!")

def add_bg_from_url():
    st.markdown(
        f"""
        <style>
        .stApp {{
            background-image: url("https://source.unsplash.com/1600x900/?education,technology");
            background-size: cover;
            background-position: center;
            background-attachment: fixed;
        }}
        </style>
        """,
        unsafe_allow_html=True
    )

add_bg_from_url()
