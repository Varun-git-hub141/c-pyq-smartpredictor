import streamlit as st
import joblib

# Load all models
nlp_model = joblib.load("question_nlp_model_v2.pkl")
topic_model = joblib.load("topic_classifier.pkl")
topic_encoder = joblib.load("topic_encoder.pkl")
non_c_model = joblib.load("c_non_c_classifier.joblib")
non_c_vectorizer = joblib.load("c_non_c_vectorizer.joblib")

# Streamlit page setup
st.set_page_config(page_title="C PYQ Smart Predictor", layout="centered")
st.title("📘 C PYQ Smart Predictor")
st.markdown("Enter a question to check:")
st.markdown("- 📚 Is it a C programming question?")
st.markdown("- 🎯 If yes: predicted topic and probability of appearing in exams")

# Input box
user_question = st.text_area("📝 Enter your question below:", height=100)

if user_question:
    cleaned = user_question.strip().lower()

    # Step 1: Check if question is C-related or not
    vectorized = non_c_vectorizer.transform([cleaned])
    is_c_related = bool(non_c_model.predict(vectorized)[0])

    st.subheader("📘 C Syllabus Check")
    if is_c_related:
        st.success("✅ This question is related to the C programming syllabus.")

        # Step 2: Predict Topic and Exam Probability
        topic_encoded = topic_model.predict([cleaned])[0]
        predicted_topic = topic_encoder.inverse_transform([topic_encoded])[0]
        prob = nlp_model.predict_proba([cleaned])[0][1]

        st.subheader("🔍 Prediction Result")
        st.markdown(f"📚 **Predicted Topic:** `{predicted_topic}`")

        if prob >= 0.6:
            st.success(f"✅ High Probability to Appear ({prob * 100:.2f}%)")
        elif prob >= 0.4:
            st.warning(f"⚠️ Medium Probability ({prob * 100:.2f}%)")
        else:
            st.error(f"❌ Low Probability to Appear ({prob * 100:.2f}%)")

    else:
        st.error("🚫 This question is **not related to the C programming syllabus.**")

# Footer
st.markdown("---")
st.caption("🔍 Powered by custom-trained ML models on previous year C questions and topics.")
