import streamlit as st
import joblib

# Load models
nlp_model = joblib.load("question_nlp_model_v2.pkl")          # Exam probability
topic_model = joblib.load("topic_classifier.pkl")              # Topic classifier
topic_encoder = joblib.load("topic_encoder.pkl")               # Label encoder
syllabus_model = joblib.load("syllabus_topics.pkl")            # TF-IDF + classifier pipeline for C-syllabus

# Streamlit UI
st.set_page_config(page_title="C PYQ Smart Predictor", layout="centered")
st.title("📘 C PYQ Smart Predictor")
st.markdown("Enter a C programming question to know:")
st.markdown("- 📚 **Predicted Topic**")
st.markdown("- 🎯 **Exam Appearance Probability**")
st.markdown("- 🧾 **Syllabus Match Check**")

# User input
user_question = st.text_area("📝 Enter your C programming question below:", height=100)

if user_question:
    cleaned = user_question.strip().lower()

    # Step 1: Check if question belongs to C syllabus
    try:
        is_c_related = bool(syllabus_model.predict([cleaned])[0])
    except Exception as e:
        st.error("⚠️ Error checking syllabus relevance.")
        st.exception(e)
        st.stop()

    # Step 2: Block non-C questions
    if not is_c_related:
        st.error("🚫 This question is **not related to the C programming syllabus**.")
    else:
        st.success("✅ This question is related to the **C programming syllabus**.")

        # Predict topic
        topic_encoded = topic_model.predict([cleaned])[0]
        predicted_topic = topic_encoder.inverse_transform([topic_encoded])[0]

        # Predict exam probability
        prob = nlp_model.predict_proba([cleaned])[0][1]

        # Show results
        st.subheader("🔍 Prediction Result")
        st.markdown(f"📚 **Predicted Topic:** `{predicted_topic}`")

        if prob >= 0.6:
            st.success(f"✅ High Probability to Appear ({prob * 100:.2f}%)")
        elif prob >= 0.4:
            st.warning(f"⚠️ Medium Probability ({prob * 100:.2f}%)")
        else:
            st.error(f"❌ Low Probability ({prob * 100:.2f}%)")

# Footer
st.markdown("---")
st.caption("🧠 Trained using real C PYQs, NLP pipelines, and syllabus filtering.")
