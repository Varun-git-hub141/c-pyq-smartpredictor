import streamlit as st
import joblib

# -------------------- Load models -------------------- #
nlp_model = joblib.load("question_nlp_model_v2.pkl")
topic_model = joblib.load("topic_classifier.pkl")
topic_encoder = joblib.load("topic_encoder.pkl")
syllabus_model = joblib.load("syllabus_topics.pkl")
non_c_classifier = joblib.load("c_non_c_classifier.joblib")
non_c_vectorizer = joblib.load("tfidf_vectorizer.joblib")

# -------------------- Page setup -------------------- #
st.set_page_config(page_title="C PYQ Smart Predictor", page_icon="📘", layout="centered")

st.markdown("""
    <style>
        body {
            background: linear-gradient(135deg, #74ebd5 0%, #ACB6E5 100%);
        }
        .main {
            background-color: #ffffff;
            border-radius: 15px;
            padding: 30px;
            box-shadow: 0 4px 12px rgba(0, 0, 0, 0.1);
            margin-top: 20px;
        }
        h1, h2, h3 {
            font-family: 'Poppins', sans-serif;
            color: #333333;
        }
        .stButton > button {
            background-color: #2F80ED;
            color: white;
            border-radius: 10px;
            padding: 0.75em 2em;
            font-size: 18px;
            transition: 0.3s;
        }
        .stButton > button:hover {
            background-color: #1c5db5;
        }
        hr {
            border: none;
            height: 2px;
            background-color: #ddd;
            margin: 20px 0;
        }
    </style>
    """, unsafe_allow_html=True)

# -------------------- Header -------------------- #
st.image("https://cdn-icons-png.flaticon.com/512/5968/5968267.png", width=100)
st.markdown("<h1 style='text-align: center;'>📘 C PYQ Smart Predictor</h1>", unsafe_allow_html=True)
st.markdown("<h3 style='text-align: center;'>Your Exam Assistant Powered by AI 🚀</h3>", unsafe_allow_html=True)

st.markdown("<hr>", unsafe_allow_html=True)
st.markdown("### ✍️ Enter a C programming question to predict:")

# -------------------- Input -------------------- #
user_question = st.text_input("", placeholder="e.g., Explain recursion with example in C...")

# -------------------- Predict button -------------------- #
if st.button("🎯 Predict"):
    if user_question:
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

            # Predict topic + probability
            topic_encoded = topic_model.predict([cleaned])[0]
            predicted_topic = topic_encoder.inverse_transform([topic_encoded])[0]
            prob = nlp_model.predict_proba([cleaned])[0][1]
            syllabus_match = "✅ Yes, it's in syllabus"

            st.markdown("<hr>", unsafe_allow_html=True)
            st.markdown("## 🎉 Prediction Results")

            # -------------------- Neumorphic Results -------------------- #
            col1, col2, col3 = st.columns(3)

            with col1:
                st.markdown("""
                    <div style="background-color:#f7f9fc; padding:20px; border-radius:15px; box-shadow: 8px 8px 15px #d1d9e6, -8px -8px 15px #ffffff;">
                        <h3>📑 Predicted Topic</h3>
                        <p style="font-size:24px; color:#2F80ED;"><b>{}</b></p>
                    </div>
                """.format(predicted_topic), unsafe_allow_html=True)

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

                st.markdown("""
                    <div style="background-color:#f7f9fc; padding:20px; border-radius:15px; box-shadow: 8px 8px 15px #d1d9e6, -8px -8px 15px #ffffff;">
                        <h3>📊 Exam Probability</h3>
                        <p style="font-size:24px; color:{};"><b>{}</b></p>
                    </div>
                """.format(color, appearance_text), unsafe_allow_html=True)

            with col3:
                st.markdown("""
                    <div style="background-color:#f7f9fc; padding:20px; border-radius:15px; box-shadow: 8px 8px 15px #d1d9e6, -8px -8px 15px #ffffff;">
                        <h3>✅ Syllabus Match</h3>
                        <p style="font-size:24px; color:#27AE60;"><b>{}</b></p>
                    </div>
                """.format(syllabus_match), unsafe_allow_html=True)

        else:
            st.error("🚫 This question is **not related** to the C programming syllabus.")
    else:
        st.warning("⚠️ Please enter a question before clicking Predict!")

# -------------------- Footer -------------------- #
st.markdown("<hr>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center;'>Made with ❤️ by Team XYZ | Hackathon 2025</p>", unsafe_allow_html=True)
