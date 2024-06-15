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
    
    # Check if the response contains a valid 'Part'
    if not response or not hasattr(response, 'text'):
        raise ValueError("The response does not contain valid text data.")
    
    # Check safety ratings to ensure the response was not blocked
    if hasattr(response, 'candidate') and hasattr(response.candidate, 'safety_ratings'):
        safety_ratings = response.candidate.safety_ratings
        if safety_ratings and any(rating.blocked for rating in safety_ratings):
            raise ValueError("The response was blocked due to safety ratings.")
    
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
            "What is the name of the doctor who wrote this prescription?": "What is the name of the doctor who wrote this prescription?",
            "What is the prescription date?": "What is the prescription date?",
            "What is the prescription number?": "What is the prescription number?"
        },
        "Dosage and Instructions": {
            "What are the dosage instructions for the medications listed?": "What are the dosage instructions for the medications listed?",
            "How often should the patient take the prescribed medication?": "How often should the patient take the prescribed medication?",
            "Are there any specific instructions for taking these medications?": "Are there any specific instructions for taking these medications?",
            "Is there a maximum dosage limit mentioned?": "Is there a maximum dosage limit mentioned?"
        },
        "Medication Purpose": {
            "What conditions are these medications prescribed for?": "What conditions are these medications prescribed for?",
            "What is the expected duration of the treatment?": "What is the expected duration of the treatment?"
        },
        "Warnings and Side Effects": {
            "Are there any warnings or precautions mentioned in this prescription?": "Are there any warnings or precautions mentioned in this prescription?",
            "What are the potential side effects of the prescribed medications?": "What are the potential side effects of the prescribed medications?",
            "Should the patient avoid any activities while taking these medications?": "Should the patient avoid any activities while taking these medications?"
        },
        "Additional Details": {
            "Is there any follow-up or additional tests required as per the prescription?": "Is there any follow-up or additional tests required as per the prescription?",
            "What is the patient's name and age as per the prescription?": "What is the patient's name and age as per the prescription?",
            "Are there any special dietary instructions mentioned?": "Are there any special dietary instructions mentioned?",
            "Is there an emergency contact number provided?": "Is there an emergency contact number provided?"
        },
        "Clarification and Translation": {
            "Can you translate this prescription into [specific language]?": "Can you translate this prescription into [specific language]?",
            "Can you clarify any unclear parts of this prescription?": "Can you clarify any unclear parts of this prescription?",
            "Is there any information written in shorthand or abbreviations?": "Is there any information written in shorthand or abbreviations?"
        }
    },
    "Hindi": {
        "मूल जानकारी": {
            "इस छवि में कौन सी दवा लिखी गई है?": "इस छवि में कौन सी दवा लिखी गई है?",
            "इस प्रिस्क्रिप्शन को लिखने वाले डॉक्टर का नाम क्या है?": "इस प्रिस्क्रिप्शन को लिखने वाले डॉक्टर का नाम क्या है?",
            "प्रिस्क्रिप्शन की तारीख क्या है?": "प्रिस्क्रिप्शन की तारीख क्या है?",
            "प्रिस्क्रिप्शन संख्या क्या है?": "प्रिस्क्रिप्शन संख्या क्या है?"
        },
        "खुराक और निर्देश": {
            "उल्लिखित दवाओं के लिए खुराक निर्देश क्या हैं?": "उल्लिखित दवाओं के लिए खुराक निर्देश क्या हैं?",
            "रोगी को निर्देशित दवा कितनी बार लेनी चाहिए?": "रोगी को निर्देशित दवा कितनी बार लेनी चाहिए?",
            "इन दवाओं को लेने के लिए कोई विशिष्ट निर्देश हैं?": "इन दवाओं को लेने के लिए कोई विशिष्ट निर्देश हैं?",
            "क्या अधिकतम खुराक सीमा उल्लिखित है?": "क्या अधिकतम खुराक सीमा उल्लिखित है?"
        },
        "दवा का उद्देश्य": {
            "इन दवाओं को किन परिस्थितियों के लिए निर्देशित किया गया है?": "इन दवाओं को किन परिस्थितियों के लिए निर्देशित किया गया है?",
            "उपचार की अपेक्षित अवधि क्या है?": "उपचार की अपेक्षित अवधि क्या है?"
        },
        "चेतावनी और दुष्प्रभाव": {
            "क्या इस प्रिस्क्रिप्शन में कोई चेतावनी या सावधानी है?": "क्या इस प्रिस्क्रिप्शन में कोई चेतावनी या सावधानी है?",
            "निर्देशित दवाओं के संभावित दुष्प्रभाव क्या हैं?": "निर्देशित दवाओं के संभावित दुष्प्रभाव क्या हैं?",
            "क्या रोगी को इन दवाओं को लेते समय किसी गतिविधि से बचना चाहिए?": "क्या रोगी को इन दवाओं को लेते समय किसी गतिविधि से बचना चाहिए?"
        },
        "अतिरिक्त जानकारी": {
            "क्या प्रिस्क्रिप्शन के अनुसार कोई फॉलो-अप या अतिरिक्त परीक्षण की आवश्यकता है?": "क्या प्रिस्क्रिप्शन के अनुसार कोई फॉलो-अप या अतिरिक्त परीक्षण की आवश्यकता है?",
            "प्रिस्क्रिप्शन के अनुसार रोगी का नाम और उम्र क्या है?": "प्रिस्क्रिप्शन के अनुसार रोगी का नाम और उम्र क्या है?",
            "क्या कोई विशेष आहार निर्देश उल्लिखित हैं?": "क्या कोई विशेष आहार निर्देश उल्लिखित हैं?",
            "क्या कोई आपातकालीन संपर्क नंबर प्रदान किया गया है?": "क्या कोई आपातकालीन संपर्क नंबर प्रदान किया गया है?"
        },
        "स्पष्टीकरण और अनुवाद": {
            "क्या आप इस प्रिस्क्रिप्शन का [विशिष्ट भाषा] में अनुवाद कर सकते हैं?": "क्या आप इस प्रिस्क्रिप्शन का [विशिष्ट भाषा] में अनुवाद कर सकते हैं?",
            "क्या आप इस प्रिस्क्रिप्शन के किसी अस्पष्ट भाग को स्पष्ट कर सकते हैं?": "क्या आप इस प्रिस्क्रिप्शन के किसी अस्पष्ट भाग को स्पष्ट कर सकते हैं?",
            "क्या कोई जानकारी संक्षेप या संक्षिप्त रूप में लिखी गई है?": "क्या कोई जानकारी संक्षेप या संक्षिप्त रूप में लिखी गई है?"
        }
    },
    "Kannada": {
        "ಮೂಲ ಮಾಹಿತಿ": {
            "ಈ ಚಿತ್ರದಲ್ಲಿ ಯಾವ ಔಷಧವನ್ನು ನಿರ್ದೇಶಿಸಲಾಗಿದೆ?": "ಈ ಚಿತ್ರದಲ್ಲಿ ಯಾವ ಔಷಧವನ್ನು ನಿರ್ದೇಶಿಸಲಾಗಿದೆ?",
            "ಈ ನುಸ್ಖೆಯನ್ನು ಬರೆದ ವೈದ್ಯರ ಹೆಸರೇನು?": "ಈ ನುಸ್ಖೆಯನ್ನು ಬರೆದ ವೈದ್ಯರ ಹೆಸರೇನು?",
            "ನುಸ್ಖೆಯ ದಿನಾಂಕವೇನು?": "ನುಸ್ಖೆಯ ದಿನಾಂಕವೇನು?",
            "ನುಸ್ಖೆಯ ಸಂಖ್ಯೆ ಏನು?": "ನುಸ್ಖೆಯ ಸಂಖ್ಯೆ ಏನು?"
        },
        "ಮಾತ್ರೆ ಮತ್ತು ಸೂಚನೆಗಳು": {
            "ಪಟ್ಟಿಮಾಡಲಾದ ಔಷಧಗಳಿಗೆ ಖುರಾಕಿನ ಸೂಚನೆಗಳು ಯಾವುವು?": "ಪಟ್ಟಿಮಾಡಲಾದ ಔಷಧಗಳಿಗೆ ಖುರಾಕಿನ ಸೂಚನೆಗಳು ಯಾವುವು?",
            "ರೋಗಿಯು ನಿರ್ದೇಶಿತ ಔಷಧವನ್ನು ಎಷ್ಟು ಬಾರಿ ತೆಗೆದುಕೊಳ್ಳಬೇಕು?": "ರೋಗಿಯು ನಿರ್ದೇಶಿತ ಔಷಧವನ್ನು ಎಷ್ಟು ಬಾರಿ ತೆಗೆದುಕೊಳ್ಳಬೇಕು?",
            "ಈ ಔಷಧಗಳನ್ನು ತೆಗೆದುಕೊಳ್ಳಲು ಯಾವುದೇ ವಿಶಿಷ್ಟ ಸೂಚನೆಗಳಿವೆಯೇ?": "ಈ ಔಷಧಗಳನ್ನು ತೆಗೆದುಕೊಳ್ಳಲು ಯಾವುದೇ ವಿಶಿಷ್ಟ ಸೂಚನೆಗಳಿವೆಯೇ?",
            "ಅಧಿಕಾಗಿದಲ್ಲದ ಪ್ರಮಾಣದ ಮಿತಿಯು ಸೂಚಿಸಲ್ಪಟ್ಟಿದೆಯೇ?": "ಅಧಿಕಾಗಿದಲ್ಲದ ಪ್ರಮಾಣದ ಮಿತಿಯು ಸೂಚಿಸಲ್ಪಟ್ಟಿದೆಯೇ?"
        },
        "ಔಷಧದ ಉದ್ದೇಶ": {
            "ಈ ಔಷಧಗಳನ್ನು ಯಾವ ಪರಿಸ್ಥಿತಿಗಳಿಗಾಗಿ ನಿರ್ದೇಶಿಸಲಾಗಿದೆ?": "ಈ ಔಷಧಗಳನ್ನು ಯಾವ ಪರಿಸ್ಥಿತಿಗಳಿಗಾಗಿ ನಿರ್ದೇಶಿಸಲಾಗಿದೆ?",
            "ಚಿಕಿತ್ಸೆಯ ನಿರೀಕ್ಷಿತ ಅವಧಿಯು ಏನು?": "ಚಿಕಿತ್ಸೆಯ ನಿರೀಕ್ಷಿತ ಅವಧಿಯು ಏನು?"
        },
        "ಎಚ್ಚರಿಕೆ ಮತ್ತು ದೋಷ ಪರಿಣಾಮಗಳು": {
            "ಈ ನುಸ್ಖೆಯಲ್ಲಿ ಯಾವುದೇ ಎಚ್ಚರಿಕೆಗಳನ್ನು ಅಥವಾ ಮುನ್ನೆಚ್ಚರಿಕೆಗಳನ್ನು ನಮೂದಿಸಲಾಗಿದೆಯೇ?": "ಈ ನುಸ್ಖೆಯಲ್ಲಿ ಯಾವುದೇ ಎಚ್ಚರಿಕೆಗಳನ್ನು ಅಥವಾ ಮುನ್ನೆಚ್ಚರಿಕೆಗಳನ್ನು ನಮೂದಿಸಲಾಗಿದೆಯೇ?",
            "ನಿರ್ದೇಶಿತ ಔಷಧಗಳ ಸಂಭಾವ್ಯ ತಾರತಮ್ಯಗಳೇನು?": "ನಿರ್ದೇಶಿತ ಔಷಧಗಳ ಸಂಭಾವ್ಯ ತಾರತಮ್ಯಗಳೇನು?",
            "ರೋಗಿಯು ಈ ಔಷಧಗಳನ್ನು ತೆಗೆದುಕೊಳ್ಳುವಾಗ ಯಾವುದೇ ಚಟುವಟಿಕೆಗಳನ್ನು ತಪ್ಪಿಸಿಕೊಳ್ಳಬೇಕೆ?": "ರೋಗಿಯು ಈ ಔಷಧಗಳನ್ನು ತೆಗೆದುಕೊಳ್ಳುವಾಗ ಯಾವುದೇ ಚಟುವಟಿಕೆಗಳನ್ನು ತಪ್ಪಿಸಿಕೊಳ್ಳಬೇಕೆ?"
        },
        "ಹೆಚ್ಚಿನ ವಿವರಗಳು": {
            "ನುಸ್ಖೆಯ ಪ್ರಕಾರ ಯಾವುದೇ ಅನುಯಾಯಿತ್ವ ಅಥವಾ ಹೆಚ್ಚಿನ ಪರೀಕ್ಷೆಗಳ ಅಗತ್ಯವಿದೆಯೇ?": "ನುಸ್ಖೆಯ ಪ್ರಕಾರ ಯಾವುದೇ ಅನುಯಾಯಿತ್ವ ಅಥವಾ ಹೆಚ್ಚಿನ ಪರೀಕ್ಷೆಗಳ ಅಗತ್ಯವಿದೆಯೇ?",
            "ನುಸ್ಖೆಯ ಪ್ರಕಾರ ರೋಗಿಯ ಹೆಸರು ಮತ್ತು ವಯಸ್ಸು ಏನು?": "ನುಸ್ಖೆಯ ಪ್ರಕಾರ ರೋಗಿಯ ಹೆಸರು ಮತ್ತು ವಯಸ್ಸು ಏನು?",
            "ನಿಗದಿತ ಆಹಾರ ಅನುಸೂಚನೆಗಳಿರುತ್ತವೆಯೇ?": "ನಿಗದಿತ ಆಹಾರ ಅನುಸೂಚನೆಗಳಿರುತ್ತವೆಯೇ?",
            "ತುರ್ತು ಸಂಪರ್ಕ ಸಂಖ್ಯೆ ಒದಗಿಸಲಾಗಿದೆ?": "ತುರ್ತು ಸಂಪರ್ಕ ಸಂಖ್ಯೆ ಒದಗಿಸಲಾಗಿದೆ?"
        },
        "ವ್ಯಾಖ್ಯಾನ ಮತ್ತು ಅನುವಾದ": {
            "ಈ ನುಸ್ಖೆಯನ್ನು [ನಿರ್ದಿಷ್ಟ ಭಾಷೆಗೆ] ಭಾಷಾಂತರಿಸಬಹುದೇ?": "ಈ ನುಸ್ಖೆಯನ್ನು [ನಿರ್ದಿಷ್ಟ ಭಾಷೆಗೆ] ಭಾಷಾಂತರಿಸಬಹುದೇ?",
            "ಈ ನುಸ್ಖೆಯ ಯಾವುದೇ ಸ್ಪಷ್ಟವಲ್ಲದ ಭಾಗಗಳನ್ನು ವಿವರಿಸಬಹುದೇ?": "ಈ ನುಸ್ಖೆಯ ಯಾವುದೇ ಸ್ಪಷ್ಟವಲ್ಲದ ಭಾಗಗಳನ್ನು ವಿವರಿಸಬಹುದೇ?",
            "ಯಾವುದೇ ಮಾಹಿತಿ ಶಾರ್ಟ್‌ಹ್ಯಾಂಡ್ ಅಥವಾ ಸಂಕ್ಷಿಪ್ತ ರೂಪದಲ್ಲಿ ಬರೆದಿರುವುದೇ?": "ಯಾವುದೇ ಮಾಹಿತಿ ಶಾರ್ಟ್‌ಹ್ಯಾಂಡ್ ಅಥವಾ ಸಂಕ್ಷಿಪ್ತ ರೂಪದಲ್ಲಿ ಬರೆದಿರುವುದೇ?"
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
        try:
            image_data = input_image_setup(uploaded_file)
            response = get_gemini_response(input_prompt, image_data, input_text)
            st.subheader("The Response is")
            st.write(response)
        except Exception as e:
            st.error(f"An error occurred: {e}")
    else:
        st.write("Please provide a text input first.")
