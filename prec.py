from dotenv import load_dotenv
import streamlit as st
import os
from PIL import Image
import google.generativeai as genai

load_dotenv()  # Take environment variables from .env.

# Configure the generative AI with the API key from environment variables
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

# Function to get the response from the Gemini model
def get_gemini_response(input, image, prompt):
    model = genai.GenerativeModel('gemini-pro-vision')
    response = model.generate_content([input, image[0], prompt])
    return response.text

# Function to process the uploaded image
def input_image_setup(uploaded_file):
    if uploaded_file is not None:
        bytes_data = uploaded_file.getvalue()
        image_parts = [
            {
                "mime_type": uploaded_file.type,
                "data": bytes_data
            }
        ]
        return image_parts
    else:
        raise FileNotFoundError("No file uploaded")

# Define a dictionary of prompts in different languages
prompts = {
    "English": {
        "Basic Information": {
            "What medication is prescribed in this image?": "What medication is prescribed in this image?",
            "Can you list the medications mentioned in this prescription?": "Can you list the medications mentioned in this prescription?",
            "What is the name of the doctor who wrote this prescription?": "What is the name of the doctor who wrote this prescription?"
        },
        "Dosage and Instructions": {
            "What are the dosage instructions for the medications listed?": "What are the dosage instructions for the medications listed?",
            "How often should the patient take the prescribed medication?": "How often should the patient take the prescribed medication?",
            "Are there any specific instructions for taking these medications?": "Are there any specific instructions for taking these medications?"
        },
        "Medication Purpose": {
            "What conditions are these medications prescribed for?": "What conditions are these medications prescribed for?",
            "Can you explain the purpose of each medication in the prescription?": "Can you explain the purpose of each medication in the prescription?"
        },
        "Warnings and Side Effects": {
            "Are there any warnings or precautions mentioned in this prescription?": "Are there any warnings or precautions mentioned in this prescription?",
            "What are the potential side effects of the prescribed medications?": "What are the potential side effects of the prescribed medications?"
        },
        "Additional Details": {
            "Is there any follow-up or additional tests required as per the prescription?": "Is there any follow-up or additional tests required as per the prescription?",
            "What is the patient's name and age as per the prescription?": "What is the patient's name and age as per the prescription?",
            "Are there any special dietary instructions mentioned?": "Are there any special dietary instructions mentioned?"
        },
        "Clarification and Translation": {
            "Can you translate this prescription into [specific language]?": "Can you translate this prescription into [specific language]?",
            "Can you clarify any unclear parts of this prescription?": "Can you clarify any unclear parts of this prescription?"
        }
    },
    "Hindi": {
        "मूल जानकारी": {
            "इस छवि में कौन सी दवा लिखी गई है?": "इस छवि में कौन सी दवा लिखी गई है?",
            "क्या आप इस प्रिस्क्रिप्शन में उल्लिखित दवाओं की सूची बना सकते हैं?": "क्या आप इस प्रिस्क्रिप्शन में उल्लिखित दवाओं की सूची बना सकते हैं?",
            "इस प्रिस्क्रिप्शन को लिखने वाले डॉक्टर का नाम क्या है?": "इस प्रिस्क्रिप्शन को लिखने वाले डॉक्टर का नाम क्या है?"
        },
        "खुराक और निर्देश": {
            "उल्लिखित दवाओं के लिए खुराक निर्देश क्या हैं?": "उल्लिखित दवाओं के लिए खुराक निर्देश क्या हैं?",
            "रोगी को निर्देशित दवा कितनी बार लेनी चाहिए?": "रोगी को निर्देशित दवा कितनी बार लेनी चाहिए?",
            "इन दवाओं को लेने के लिए कोई विशिष्ट निर्देश हैं?": "इन दवाओं को लेने के लिए कोई विशिष्ट निर्देश हैं?"
        },
        "दवा का उद्देश्य": {
            "इन दवाओं को किन परिस्थितियों के लिए निर्देशित किया गया है?": "इन दवाओं को किन परिस्थितियों के लिए निर्देशित किया गया है?",
            "क्या आप प्रिस्क्रिप्शन में प्रत्येक दवा का उद्देश्य स्पष्ट कर सकते हैं?": "क्या आप प्रिस्क्रिप्शन में प्रत्येक दवा का उद्देश्य स्पष्ट कर सकते हैं?"
        },
        "चेतावनी और दुष्प्रभाव": {
            "क्या इस प्रिस्क्रिप्शन में कोई चेतावनी या सावधानी है?": "क्या इस प्रिस्क्रिप्शन में कोई चेतावनी या सावधानी है?",
            "निर्देशित दवाओं के संभावित दुष्प्रभाव क्या हैं?": "निर्देशित दवाओं के संभावित दुष्प्रभाव क्या हैं?"
        },
        "अतिरिक्त जानकारी": {
            "क्या प्रिस्क्रिप्शन के अनुसार कोई फॉलो-अप या अतिरिक्त परीक्षण की आवश्यकता है?": "क्या प्रिस्क्रिप्शन के अनुसार कोई फॉलो-अप या अतिरिक्त परीक्षण की आवश्यकता है?",
            "प्रिस्क्रिप्शन के अनुसार रोगी का नाम और उम्र क्या है?": "प्रिस्क्रिप्शन के अनुसार रोगी का नाम और उम्र क्या है?",
            "क्या कोई विशेष आहार निर्देश उल्लिखित हैं?": "क्या कोई विशेष आहार निर्देश उल्लिखित हैं?"
        },
        "स्पष्टीकरण और अनुवाद": {
            "क्या आप इस प्रिस्क्रिप्शन का [विशिष्ट भाषा] में अनुवाद कर सकते हैं?": "क्या आप इस प्रिस्क्रिप्शन का [विशिष्ट भाषा] में अनुवाद कर सकते हैं?",
            "क्या आप इस प्रिस्क्रिप्शन के किसी अस्पष्ट भाग को स्पष्ट कर सकते हैं?": "क्या आप इस प्रिस्क्रिप्शन के किसी अस्पष्ट भाग को स्पष्ट कर सकते हैं?"
        }
    },
    "Kannada": {
        "ಮೂಲ ಮಾಹಿತಿ": {
            "ಈ ಚಿತ್ರದಲ್ಲಿ ಯಾವ ಔಷಧವನ್ನು ನಿರ್ದೇಶಿಸಲಾಗಿದೆ?": "ಈ ಚಿತ್ರದಲ್ಲಿ ಯಾವ ಔಷಧವನ್ನು ನಿರ್ದೇಶಿಸಲಾಗಿದೆ?",
            "ನೀವು ಈ ನುಸ್ಖೆಯಲ್ಲಿ ಉಲ್ಲೇಖಿಸಲಾದ ಔಷಧಗಳ ಪಟ್ಟಿ ನೀಡಬಹುದೇ?": "ನೀವು ಈ ನುಸ್ಖೆಯಲ್ಲಿ ಉಲ್ಲೇಖಿಸಲಾದ ಔಷಧಗಳ ಪಟ್ಟಿ ನೀಡಬಹುದೇ?",
            "ಈ ನುಸ್ಖೆಯನ್ನು ಬರೆದ ವೈದ್ಯರ ಹೆಸರೇನು?": "ಈ ನುಸ್ಖೆಯನ್ನು ಬರೆದ ವೈದ್ಯರ ಹೆಸರೇನು?"
        },
        "ಮಾತ್ರೆ ಮತ್ತು ಸೂಚನೆಗಳು": {
            "ಪಟ್ಟಿಮಾಡಲಾದ ಔಷಧಗಳಿಗೆ ಖುರಾಕಿನ ಸೂಚನೆಗಳು ಯಾವುವು?": "ಪಟ್ಟಿಮಾಡಲಾದ ಔಷಧಗಳಿಗೆ ಖುರಾಕಿನ ಸೂಚನೆಗಳು ಯಾವುವು?",
            "ರೋಗಿಯು ನಿರ್ದೇಶಿತ ಔಷಧವನ್ನು ಎಷ್ಟು ಬಾರಿ ತೆಗೆದುಕೊಳ್ಳಬೇಕು?": "ರೋಗಿಯು ನಿರ್ದೇಶಿತ ಔಷಧವನ್ನು ಎಷ್ಟು ಬಾರಿ ತೆಗೆದುಕೊಳ್ಳಬೇಕು?",
            "ಈ ಔಷಧಗಳನ್ನು ತೆಗೆದುಕೊಳ್ಳಲು ಯಾವುದೇ ವಿಶಿಷ್ಟ ಸೂಚನೆಗಳಿವೆಯೇ?": "ಈ ಔಷಧಗಳನ್ನು ತೆಗೆದುಕೊಳ್ಳಲು ಯಾವುದೇ ವಿಶಿಷ್ಟ ಸೂಚನೆಗಳಿವೆಯೇ?"
        },
        "ಔಷಧದ ಉದ್ದೇಶ": {
            "ಈ ಔಷಧಗಳನ್ನು ಯಾವ ಪರಿಸ್ಥಿತಿಗಳಿಗಾಗಿ ನಿರ್ದೇಶಿಸಲಾಗಿದೆ?": "ಈ ಔಷಧಗಳನ್ನು ಯಾವ ಪರಿಸ್ಥಿತಿಗಳಿಗಾಗಿ ನಿರ್ದೇಶಿಸಲಾಗಿದೆ?",
            "ನುಸ್ಖೆಯಲ್ಲಿನ ಪ್ರತಿಯೊಂದು ಔಷಧದ ಉದ್ದೇಶವನ್ನು ನೀವು ವಿವರಿಸಬಹುದೇ?": "ನುಸ್ಖೆಯಲ್ಲಿನ ಪ್ರತಿಯೊಂದು ಔಷಧದ ಉದ್ದೇಶವನ್ನು ನೀವು ವಿವರಿಸಬಹುದೇ?"
        },
        "ಎಚ್ಚರಿಕೆ ಮತ್ತು ದೋಷ ಪರಿಣಾಮಗಳು": {
            "ಈ ನುಸ್ಖೆಯಲ್ಲಿ ಯಾವುದೇ ಎಚ್ಚರಿಕೆಗಳನ್ನು ಅಥವಾ ಮುನ್ನೆಚ್ಚರಿಕೆಗಳನ್ನು ನಮೂದಿಸಲಾಗಿದೆಯೇ?": "ಈ ನುಸ್ಖೆಯಲ್ಲಿ ಯಾವುದೇ ಎಚ್ಚರಿಕೆಗಳನ್ನು ಅಥವಾ ಮುನ್ನೆಚ್ಚರಿಕೆಗಳನ್ನು ನಮೂದಿಸಲಾಗಿದೆಯೇ?",
            "ನಿರ್ದೇಶಿತ ಔಷಧಗಳ ಸಂಭಾವ್ಯ ತಾರತಮ್ಯಗಳೇನು?": "ನಿರ್ದೇಶಿತ ಔಷಧಗಳ ಸಂಭಾವ್ಯ ತಾರತಮ್ಯಗಳೇನು?"
        },
        "ಹೆಚ್ಚಿನ ವಿವರಗಳು": {
            "ನುಸ್ಖೆಯ ಪ್ರಕಾರ ಯಾವುದೇ ಅನುಯಾಯಿತ್ವ ಅಥವಾ ಹೆಚ್ಚಿನ ಪರೀಕ್ಷೆಗಳ ಅಗತ್ಯವಿದೆಯೇ?": "ನುಸ್ಖೆಯ ಪ್ರಕಾರ ಯಾವುದೇ ಅನುಯಾಯಿತ್ವ ಅಥವಾ ಹೆಚ್ಚಿನ ಪರೀಕ್ಷೆಗಳ ಅಗತ್ಯವಿದೆಯೇ?",
            "ನುಸ್ಖೆಯ ಪ್ರಕಾರ ರೋಗಿಯ ಹೆಸರು ಮತ್ತು ವಯಸ್ಸು ಏನು?": "ನುಸ್ಖೆಯ ಪ್ರಕಾರ ರೋಗಿಯ ಹೆಸರು ಮತ್ತು ವಯಸ್ಸು ಏನು?",
            "ನಿಗದಿತ ಆಹಾರ ಅನುಸೂಚನೆಗಳಿರುತ್ತವೆಯೇ?": "ನಿಗದಿತ ಆಹಾರ ಅನುಸೂಚನೆಗಳಿರುತ್ತವೆಯೇ?"
        },
        "ವ್ಯಾಖ್ಯಾನ ಮತ್ತು ಅನುವಾದ": {
            "ಈ ನುಸ್ಖೆಯನ್ನು [ನಿರ್ದಿಷ್ಟ ಭಾಷೆಗೆ] ಭಾಷಾಂತರಿಸಬಹುದೇ?": "ಈ ನುಸ್ಖೆಯನ್ನು [ನಿರ್ದಿಷ್ಟ ಭಾಷೆಗೆ] ಭಾಷಾಂತರಿಸಬಹುದೇ?",
            "ಈ ನುಸ್ಖೆಯ ಯಾವುದೇ ಸ್ಪಷ್ಟವಲ್ಲದ ಭಾಗಗಳನ್ನು ವಿವರಿಸಬಹುದೇ?": "ಈ ನುಸ್ಖೆಯ ಯಾವುದೇ ಸ್ಪಷ್ಟವಲ್ಲದ ಭಾಗಗಳನ್ನು ವಿವರಿಸಬಹುದೇ?"
        }
    }
}

# Initialize Streamlit app
st.set_page_config(page_title="Multi-Language Prescription Extractor")

st.header("Medical Prescription Application")

# Language selection
language = st.selectbox("Select Language", list(prompts.keys()))

# Category selection
category = st.selectbox("Select Category", list(prompts[language].keys()))

# Prompt selection
prompt_text = st.selectbox("Select Prompt", list(prompts[language][category].keys()))

# Input text is the selected prompt
input_text = prompts[language][category][prompt_text]

uploaded_file = st.file_uploader("Choose an image of the prescription...", type=["jpg", "jpeg", "png"])

image = ""
if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image.", use_column_width=True)

# Custom prompt input section
st.subheader("Custom Prompt")
custom_prompt = st.text_area("Enter your custom prompt")

# Submit button
submit = st.button("Explain the Prescription")

input_prompt = """
               You are an expert in understanding medical prescriptions.
               You will receive input images as medical prescriptions &
               you will have to answer questions based on the input image.
               """

# If the submit button is clicked
if submit:
    if custom_prompt:  # If custom prompt is provided, use it
        input_text = custom_prompt

    if input_text:
        image_data = input_image_setup(uploaded_file)
        response = get_gemini_response(input_prompt, image_data, input_text)
        st.subheader("The Response is")
        st.write(response)
    else:
        st.write("Please provide a text input first.")
