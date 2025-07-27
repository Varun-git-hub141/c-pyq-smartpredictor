import streamlit as st
import joblib

# Load models
nlp_model = joblib.load("question_nlp_model_v2.pkl")
topic_model = joblib.load("topic_classifier.pkl")
topic_encoder = joblib.load("topic_encoder.pkl")
syllabus_model = joblib.load("syllabus_topics.pkl")
non_c_classifier = joblib.load("c_non_c_classifier.joblib")
non_c_vectorizer = joblib.load("tfidf_vectorizer.joblib")

# App UI
st.set_page_config(page_title="C PYQ Smart Predictor", layout="centered")
st.title("📘 C PYQ Smart Predictor")
st.markdown("Enter a C programming question to know:")
st.markdown("- 📚 **Predicted Topic**")
st.markdown("- 🎯 **Exam Appearance Probability**")
st.markdown("- 🧾 **Syllabus Match Check**")

# Text input and button
user_question = st.text_input("📝 Type your question here:")
play = st.button("▶️ Play / Predict")

if play and user_question:
    cleaned = user_question.strip().lower()

    # Step 1: Fallback keyword match
    fallback_keywords = [
        "what is c", "define c", "c language", "who invented c", "c programming",
        "why c language", "uses of c", "advantages of c", "features of c"
    ]
    is_obviously_c = any(keyword in cleaned for keyword in fallback_keywords)

    # Step 2: ML syllabus check
    is_c_model = bool(syllabus_model.predict([cleaned])[0])

    # Final decision
    is_c = is_obviously_c or is_c_model

    st.subheader("📘 C Syllabus Check")
    if is_c:
        st.success("✅ This question is related to the C programming syllabus.")

        # Predict topic
        topic_encoded = topic_model.predict([cleaned])[0]
        predicted_topic = topic_encoder.inverse_transform([topic_encoded])[0]

        # Predict probability
        prob = nlp_model.predict_proba([cleaned])[0][1]

        st.subheader("🔍 Prediction Result")
        st.markdown(f"📚 **Predicted Topic:** `{predicted_topic}`")

        if prob >= 0.6:
            st.success(f"✅ High Probability to Appear ({prob * 100:.2f}%)")
        elif prob >= 0.4:
            st.warning(f"⚠️ Medium Probability ({prob * 100:.2f}%)")
        else:
            st.error(f"❌ Low Probability ({prob * 100:.2f}%)")
    else:
        st.error("🚫 This question is **not related** to the C programming syllabus.")
