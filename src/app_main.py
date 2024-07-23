import time
import json
import pandas as pd
import streamlit as st
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Define questions and answers arrays
# questions = [
#     "Cough", "Cold", "Headache", "Indigestion", "Acidity", "Constipation",
#     "Joint Pain", "Insomnia", "Anxiety", "Skin Rash", "Hair Loss", "Sore Throat",
#     "Fatigue", "Back Pain", "High Blood Pressure", "Diabetes", "Allergies",
#     "Menstrual Cramps", "Weight Loss", "Depression"
# ]

# answers = [
#     "Mix honey and ginger juice, take twice daily.",
#     "Boil tulsi leaves and ginger in water, drink as tea.",
#     "Apply a paste of ginger powder and water on the forehead.",
#     "Drink warm water with a teaspoon of cumin seeds.",
#     "Consume a mixture of aloe vera juice and honey.",
#     "Drink warm water with lemon juice and honey in the morning.",
#     "Apply a paste of turmeric and ginger on the affected area.",
#     "Drink warm milk with a pinch of nutmeg powder before bed.",
#     "Drink a tea made from ashwagandha and brahmi leaves.",
#     "Apply a paste of neem and turmeric on the rash.",
#     "Massage the scalp with warm coconut oil mixed with hibiscus powder.",
#     "Gargle with warm salt water mixed with turmeric.",
#     "Drink ashwagandha tea to boost energy levels.",
#     "Apply a paste of ginger powder and eucalyptus oil on the back.",
#     "Drink a tea made from holy basil and lemon balm.",
#     "Consume a mixture of bitter gourd juice and turmeric.",
#     "Drink a tea made from licorice root and turmeric.",
#     "Drink a tea made from fennel seeds and ginger.",
#     "Drink a mixture of honey and lemon juice in warm water.",
#     "Consume a tea made from saffron and turmeric."
# ]



def load_data(file_path):
    data = pd.read_csv(file_path)
    questions = []
    answers = []

    for index, row in data.iterrows():
        disease = row['Disease']
        remedies = row['Remedies']
        symptoms = row['Symptoms'].split(',')  # Split symptoms by comma

        for symptom in symptoms:
            symptom = symptom.strip()  # Remove any leading/trailing whitespace
            question = symptom
            answer = f"Based on the problem you are facing, you have {disease}.\n The Remedies to cure the disease are as follows: {remedies}.\n   {row['How to Apply Remedies']}"
            questions.append(question)
            answers.append(answer)

    return questions, answers

# Example usage
file_path = "data/ayurvedic_QnA.csv"                # Replace data path with your file path
questions, answers = load_data(file_path)





# Initialize TF-IDF Vectorizer
try:
    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform(questions)
except Exception as e:
    st.write(f"Error initializing or fitting TF-IDF Vectorizer: {e}")

# Function to get response from Gemini Pro using genai
def get_gemini_response(query):
    try:
        str = ""
        # Check if response contains an answer
        if query in questions:
            index = questions.index(query)
            return answers[index]
        if str in query:
            # Simulated external API call placeholder
            return "Please enter valid prompt."
            
        else: 
            return "  Sorry! I am not sure about that."
    except Exception as e:
        return f"Error contacting Gemini Pro: {str(e)}"




def main():
    # Streamlit UI
    st.title(" 🤖 Welcome to Ayurmind Bot 🤖")

    if 'conversation' not in st.session_state:
            st.session_state.conversation = []


        # Greeting responses
    greetings = ["hi", "hello", "hey"]
    greeting_responses = ["Hello! How can I help you?", "Hi there! How can I help you?", "Hey! How can I help you?"]

    identity = ["What is your name?", "Who are you?"]
    identity_responses = ["Hello! My name is  Ayurmind Bot, here to help you understand how Ayurveda can assist in managing and treating various health conditions. I will try to predict your diseases and their Ayurvedic remedies based on your symptoms.  Thank you for chosing Ayurmind BOT.", "Hello! My name is Ayurmind Bot, here to help you understand how Ayurveda can assist in managing and treating various health conditions. I will try to predict your diseases and their Ayurvedic remedies based on your symptoms.  Thank you for chosing Ayurmind BOT."]

        # User input
    user_input = st.text_input("You: ", "", key="user_input")

    if st.button("Send"):
            try:
                # Determine bot response
                user_input_lower = user_input.lower()
                if user_input_lower in greetings:
                    bot_response = greeting_responses[greetings.index(user_input_lower)]
                    
                elif user_input_lower in identity:
                    bot_response = identity_responses[0]
                
                
                else:
                    
                
                        user_query_vector = vectorizer.transform([user_input])
                        similarity = cosine_similarity(user_query_vector, tfidf_matrix).flatten()
                        max_similarity_index = similarity.argmax()

                        st.write(f"Max similarity index: {max_similarity_index}")
                        st.write(f"Similarity score: {similarity[max_similarity_index]}")

                        if similarity[max_similarity_index] > 0.5:
                            bot_response = answers[max_similarity_index]
                        else:
                            bot_response = get_gemini_response(user_input)

                # Add to conversation history
                st.session_state.conversation.append((user_input, bot_response))
            except Exception as e:
                st.write(f"Error processing input: {e}")

        # Display conversation
    for user_msg, bot_msg in st.session_state.conversation:
            st.write(f"<div style='border: 1px solid #ddd; border-radius: 8px; padding: 10px; margin-bottom: 10px; background-color: #ffffff;'> <b> You:  </b> {user_msg}</div>", unsafe_allow_html=True)
            st.write(f"<div style='border: 1px solid #ddd; border-radius: 8px; padding: 10px; margin-bottom: 10px; background-color: #ffffff;'><b> 🤖:  </b> {bot_msg}</div>", unsafe_allow_html=True)

        # Add a bot icon to the text input
    st.markdown("""
            <style>
            *{
                background: wheat;
                color: black;
            }
            .stTextInput {
                padding-left: 40px;
                background: url('https://cdn.pixabay.com/photo/2023/03/05/21/11/ai-generated-7832244_640.jpg') no-repeat 10px center;
                background-size: 20px;
            }
            .stButton {
                text-align: center;
                width: auto;
                color: black;s
                font-size: 16px;
                font-weight: bold;
                border-radius: 5px;
                padding: 10px;
            }
            .stButton:hover {
                font-size: 20px;
                font-weight: 650;
            }
            </style>
            """, unsafe_allow_html=True)




if __name__ == "__main__":
    main()
