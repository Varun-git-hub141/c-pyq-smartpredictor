import streamlit as st
import joblib

# Load both models
nlp_model = joblib.load("question_nlp_model_v2.pkl")
topic_model = joblib.load("topic_classifier.pkl")
topic_encoder = joblib.load("topic_encoder.pkl")

st.title("📘 C PYQ Smart Predictor")
st.markdown("Enter a C programming question and get predictions for **exam probability** and **topic**.")

# Input box
user_question = st.text_area("📝 Enter your question below:", height=100)

if user_question:
    # Preprocess question
    cleaned_question = user_question.strip().lower()

    # Predict topic
    topic_encoded = topic_model.predict([cleaned_question])[0]
    predicted_topic = topic_encoder.inverse_transform([topic_encoded])[0]

    # Predict exam probability
    prediction = nlp_model.predict([cleaned_question])[0]

    # Show results
    st.subheader("🔍 Prediction Result")
    st.markdown(f"📚 **Predicted Topic:** `{predicted_topic}`")
    
    if prediction == 1:
        st.success("✅ This question has a **high probability** of appearing in exams.")
    else:
        st.warning("⚠️ This question has **low probability** of appearing.")

# Footer
st.markdown("---")
st.caption("Trained using real PYQs and NLP trend analysis (Logistic Regression + TF-IDF)")
import streamlit as st
import joblib

# Load the trained NLP model
model = joblib.load("question_nlp_model_v2.pkl")

# App UI
st.title("📘 C PYQ Probability Predictor")
st.markdown("Ask any C programming question and find out if it has a high probability to appear in exams based on past trends.")

# User input
question = st.text_input("🔍 Enter your C programming question:")

if question:
    # Preprocess the input
    processed_question = question.lower()

    # Predict probability
    prob = model.predict_proba([processed_question])[0][1]  # Probability for class 1

    # Display results
    st.subheader("🧠 Prediction Result")
    if prob >= 0.6:
        st.success(f"✅ High Probability to Appear ({prob*100:.2f}%)")
    elif prob >= 0.4:
        st.warning(f"⚠️ Medium Probability to Appear ({prob*100:.2f}%)")
    else:
        st.error(f"❌ Low Probability to Appear ({prob*100:.2f}%)")

    # For debugging
    st.caption("This result is based on trends in previous years' C programming questions.")