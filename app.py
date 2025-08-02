import streamlit as st
import joblib
import requests

# -------------------- Optional: Lottie Animation Loader -------------------- #
def load_lottieurl(url: str):
    try:
        r = requests.get(url)
        if r.status_code != 200:
            return None
        return r.json()
    except:
        return None

# -------------------- Load joblib models -------------------- #
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
            background-color: #f4f7fb;
        }
        .main {
            padding: 0;
        }
        h1, h2, h3, h4 {
            font-family: 'Poppins', sans-serif;
            color: #222831;
        }
        .hero-section {
            background-color: #e3f2fd;
            padding: 50px 20px;
            text-align: center;
            border-radius: 20px;
            margin-bottom: 30px;
        }
        .hero-section img {
            max-width: 100%;
            height: auto;
            border-radius: 12px;
            margin-bottom: 20px;
        }
        .input-card {
            background-color: #ffffff;
            border-radius: 15px;
            padding: 30px;
            box-shadow: 0 8px 32px rgba(0, 0, 0, 0.1);
            margin-bottom: 30px;
        }
        .stButton > button {
            background-color: #4f46e5;
            color: white;
            border-radius: 12px;
            padding: 0.75em 2em;
            font-size: 18px;
            font-weight: 600;
            transition: all 0.3s ease-in-out;
        }
        .stButton > button:hover {
            background-color: #3730a3;
            transform: scale(1.05);
        }
    </style>
""", unsafe_allow_html=True)

# -------------------- Hero Section -------------------- #
st.markdown("""
    <div class='hero-section'>
        <img src='https://i.postimg.cc/FFbh7Lzn/undraw-ideas-vn7a.png' alt='Team Working'>
        <h1>📘 C PYQ Smart Predictor</h1>
        <h4>Your Smart AI Exam Assistant by teamalris 🚀</h4>
    </div>
""", unsafe_allow_html=True)

# -------------------- Input Card -------------------- #
st.markdown("<div class='input-card'>", unsafe_allow_html=True)
st.markdown("### 🐇 Enter your C programming question below:")
user_question = st.text_input("", placeholder="e.g., Explain the use of pointers in arrays in C...")
st.markdown("</div>", unsafe_allow_html=True)

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

            # C Syllabus Check card
            st.markdown("""
                <div class='input-card'>
                <h2>📘 C Syllabus Check</h2>
            """, unsafe_allow_html=True)

            if is_c:
                st.success("✅ This question is related to the C programming syllabus.")

                topic_encoded = topic_model.predict([cleaned])[0]
                predicted_topic = topic_encoder.inverse_transform([topic_encoded])[0]
                prob = nlp_model.predict_proba([cleaned])[0][1]
                syllabus_match = "✅ Yes, it's in syllabus"

                st.markdown("### 🎉 Prediction Results")

                col1, col2, col3 = st.columns(3)

                with col1:
                    st.metric(label="📑 Predicted Topic", value=predicted_topic)

                with col2:
                    if prob >= 0.6:
                        appearance_text = f"High ({prob * 100:.2f}%)"
                    elif prob >= 0.4:
                        appearance_text = f"Medium ({prob * 100:.2f}%)"
                    else:
                        appearance_text = f"Low ({prob * 100:.2f}%)"
                    st.metric(label="📊 Exam Probability", value=appearance_text)

                with col3:
                    st.metric(label="✅ Syllabus Match", value=syllabus_match)

            else:
                st.error("🚫 This question is **not related** to the C programming syllabus.")

            st.markdown("</div>", unsafe_allow_html=True)

    else:
        st.warning("⚠️ Please enter a question before clicking Predict!")

# -------------------- Footer -------------------- #
st.markdown("<hr>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center;'>Built with ❤️ by <b>teamalris</b> | Hackathon 2025</p>", unsafe_allow_html=True)


