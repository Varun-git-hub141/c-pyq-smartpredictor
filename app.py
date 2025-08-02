import streamlit as st
import joblib
from streamlit_lottie import st_lottie
import requests

# -------------------- Function to load Lottie -------------------- #
def load_lottieurl(url: str):
    r = requests.get(url)
    if r.status_code != 200:
        return None
    return r.json()

# -------------------- Load models -------------------- #
nlp_model = joblib.load("question_nlp_model_v2.pkl")
topic_model = joblib.load("topic_classifier.pkl")
topic_encoder = joblib.load("topic_encoder.pkl")
syllabus_model = joblib.load("syllabus_topics.pkl")
non_c_classifier = joblib.load("c_non_c_classifier.joblib")
non_c_vectorizer = joblib.load("tfidf_vectorizer.joblib")

# -------------------- Page setup -------------------- #
st.set_page_config(page_title="C PYQ Smart Predictor", page_icon="📘", layout="centered")

# -------------------- Custom CSS -------------------- #
st.markdown("""
    <style>
        body {
            background: linear-gradient(120deg, #dfe9f3 0%, #ffffff 100%);
        }
        .main {
            backdrop-filter: blur(10px);
            background-color: rgba(255, 255, 255, 0.7);
            border-radius: 15px;
            padding: 30px;
            box-shadow: 0 8px 32px 0 rgba(31, 38, 135, 0.2);
            margin-top: 20px;
        }
        h1, h2, h3, h4 {
            font-family: 'Poppins', sans-serif;
            color: #333333;
        }
        .stButton > button {
            background-color: #2F80ED;
            color: white;
            border-radius: 12px;
            padding: 0.75em 2em;
            font-size: 18px;
            font-weight: 600;
            transition: all 0.3s ease-in-out;
        }
        .stButton > button:hover {
            background-color: #1c5db5;
            transform: scale(1.05);
        }
        hr {
            border: none;
            height: 2px;
            background-color: #ddd;
            margin: 20px 0;
        }
    </style>
    """, unsafe_allow_html=True)

# -------------------- Lottie Animation -------------------- #
lottie_study = load_lottieurl("https://assets10.lottiefiles.com/packages/lf20_tijmpv.json")  # You can replace URL

# -------------------- Header -------------------- #
st_lottie(lottie_study, speed=1, width=250)
st.markdown("<h1 style='text-align: center;'>📘 C PYQ Smart Predictor</h1>", unsafe_allow_html=True)
st.markdown("<h4 style='text-align: center;'>Your Smart AI Exam Assistant by teamalris 🚀</h4>", unsafe_allow_html=True)

st.markdown("<hr>", unsafe_allow_html=True)

# -------------------- Input -------------------- #
st.markdown("### ✍️ Enter your C programming question below:")
user_question = st.text_input("", placeholder="e.g., Explain the use of pointers in arrays in C...")

# -------------------- Predict -------------------- #
if st.button("🎯 Predict"):
    if user_question:
        with st.spinner("Analyzing your question..."):
            cleaned = user_question.strip().lower()

            fallback_keywords = [
                "what is c", "define c", "c language", "who invented c", "c programming",
                "why c language", "uses of c", "advantages of c", "features of c"
            ]
            is_obviously_c = any(keyword in cleaned for keyword in fallback_keywords)
            non_c_vector = non_c_vectorizer.transform([cleaned])
            is_c_model = bool(non_c_classifier.predict(non_c_vector)[0])
            is_c = is_obviously_c or is_c_model

            st.markdown("<hr>", unsafe_allow_html=True)
            st.markdown("## 📘 C Syllabus Check")

            if is_c:
                st.success("✅ This question is related to the C programming syllabus.")

                topic_encoded = topic_model.predict([cleaned])[0]
                predicted_topic = topic_encoder.inverse_transform([topic_encoded])[0]
                prob = nlp_model.predict_proba([cleaned])[0][1]
                syllabus_match = "✅ Yes, it's in syllabus"

                st.markdown("<hr>", unsafe_allow_html=True)
                st.markdown("## 🎉 Prediction Results")

                # -------------------- Results Cards -------------------- #
                col1, col2, col3 = st.columns(3)

                with col1:
                    st.markdown(f"""
                        <div style="backdrop-filter: blur(6px); background-color: rgba(255,255,255,0.6); 
                        padding:20px; border-radius:15px; box-shadow: 0 4px 20px rgba(0,0,0,0.1);">
                            <h3>📑 Predicted Topic</h3>
                            <p style="font-size:24px; color:#2F80ED;"><b>{predicted_topic}</b></p>
                        </div>
                    """, unsafe_allow_html=True)

                with col2:
                    appearance_text = ""
                    color = ""
                    if prob >= 0.6:
                        appearance_text = f"High ({prob * 100:.2f}%)"
                        color = "#27AE60"
                    elif prob >= 0.4:
                        appearance_text = f"Medium ({prob * 100:.2f}%)"
                        color = "#F2994A"
                    else:
                        appearance_text = f"Low ({prob * 100:.2f}%)"
                        color = "#EB5757"

                    st.markdown(f"""
                        <div style="backdrop-filter: blur(6px); background-color: rgba(255,255,255,0.6); 
                        padding:20px; border-radius:15px; box-shadow: 0 4px 20px rgba(0,0,0,0.1);">
                            <h3>📊 Exam Probability</h3>
                            <p style="font-size:24px; color:{color};"><b>{appearance_text}</b></p>
                        </div>
                    """, unsafe_allow_html=True)

                with col3:
                    st.markdown(f"""
                        <div style="backdrop-filter: blur(6px); background-color: rgba(255,255,255,0.6); 
                        padding:20px; border-radius:15px; box-shadow: 0 4px 20px rgba(0,0,0,0.1);">
                            <h3>✅ Syllabus Match</h3>
                            <p style="font-size:24px; color:#27AE60;"><b>{syllabus_match}</b></p>
                        </div>
                    """, unsafe_allow_html=True)

            else:
                st.error("🚫 This question is **not related** to the C programming syllabus.")
    else:
        st.warning("⚠️ Please enter a question before clicking Predict!")

# -------------------- Footer -------------------- #
st.markdown("<hr>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center;'>Built with ❤️ by <b>teamairis</b> | Hackathon 2025</p>", unsafe_allow_html=True)

