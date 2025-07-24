import streamlit as st
import joblib
import re

# Load models
nlp_model = joblib.load("question_nlp_model_v2.pkl")
topic_model = joblib.load("topic_classifier.pkl")
topic_encoder = joblib.load("topic_encoder.pkl")
syllabus_vectorizer = joblib.load("syllabus_vectorizer.pkl")
syllabus_model = joblib.load("syllabus_topics.pkl")  # Binary classifier: is it C syllabus or not?

# Text cleaning
def preprocess(text):
    text = text.lower().strip()
    text = re.sub(r'[^\w\s]', '', text)
    return text

# App Title
st.set_page_config(page_title="C PYQ Smart Predictor", page_icon="📘")
st.title("📘 C PYQ Smart Predictor")
st.markdown("Get predictions for:")
st.markdown("- 🔍 If your question is from **C Syllabus**")
st.markdown("- 📚 Its **Topic**")
st.markdown("- 🎯 Its **Probability to Appear** in the Exam")

# User Input
question = st.text_area("📝 Enter your question:", height=100)

if question:
    cleaned = preprocess(question)

    # Check if question belongs to C Syllabus
    is_c = syllabus_model.predict(syllabus_vectorizer.transform([cleaned]))[0]

    st.subheader("📘 C Syllabus Check")
    if is_c:
        st.success("✅ This question is related to the C programming syllabus.")
        
        # Predict Topic
        topic_encoded = topic_model.predict([cleaned])[0]
        predicted_topic = topic_encoder.inverse_transform([topic_encoded])[0]
        
        # Predict Probability
        prob = nlp_model.predict_proba([cleaned])[0][1]

        # Display Results
        st.subheader("🔍 Prediction Result")
        st.markdown(f"📚 **Predicted Topic:** `{predicted_topic}`")

        if prob >= 0.6:
            st.success(f"✅ **High Probability to Appear** ({prob*100:.2f}%)")
        elif prob >= 0.4:
            st.warning(f"⚠️ **Medium Probability** ({prob*100:.2f}%)")
        else:
            st.error(f"❌ **Low Probability** ({prob*100:.2f}%)")

    else:
        st.error("🚫 This question does **not** seem to belong to the C programming syllabus.")
        st.markdown("💡 Try asking a question like `What is recursion in C?` or `Explain arrays in C.`")

# Footer
st.markdown("---")
st.caption("🧠 Trained on C PYQs using Logistic Regression, TF-IDF and syllabus validation.")
    
