import streamlit as st
import joblib

# Load models
nlp_model = joblib.load("question_nlp_model_v2.pkl")
topic_model = joblib.load("topic_classifier.pkl")
topic_encoder = joblib.load("topic_encoder.pkl")
syllabus_model = joblib.load("syllabus_topics.pkl")  # ✅ This is the trained pipeline model

# App UI
st.set_page_config(page_title="C PYQ Smart Predictor", layout="centered")
st.title("📘 C PYQ Smart Predictor")
st.markdown("Enter a C programming question to know:")
st.markdown("- 📚 **Predicted Topic**")
st.markdown("- 🎯 **Exam Appearance Probability**")
st.markdown("- 🧾 **Syllabus Match Check**")

# Input
user_question = st.text_area("📝 Enter your C programming question below:", height=100)

if user_question:
    cleaned = user_question.strip().lower()

    # ✅ Syllabus check directly using the trained pipeline
    is_c = bool(syllabus_model.predict([cleaned])[0])

    st.subheader("📘 C Syllabus Check")
    if is_c:
        st.success("✅ This question is related to the **C programming syllabus**.")

        # Predict Topic
        topic_encoded = topic_model.predict([cleaned])[0]
        predicted_topic = topic_encoder.inverse_transform([topic_encoded])[0]

        # Predict Probability
        prob = nlp_model.predict_proba([cleaned])[0][1]

        # Display
        st.subheader("🔍 Prediction Result")
        st.markdown(f"📚 **Predicted Topic:** `{predicted_topic}`")

        if prob >= 0.6:
            st.success(f"✅ **High Probability** to Appear in Exam ({prob * 100:.2f}%)")
        elif prob >= 0.4:
            st.warning(f"⚠️ **Medium Probability** to Appear ({prob * 100:.2f}%)")
        else:
            st.error(f"❌ **Low Probability** to Appear ({prob * 100:.2f}%)")

    else:
        st.error("🚫 This question is **not related** to the C programming syllabus.")

# Footer
st.markdown("---")
st.caption("🔍 Trained using previous year questions, C syllabus, and NLP analysis.")
