import streamlit as st
import os
from PIL import Image
import google.generativeai as genai
from gtts import gTTS
import tempfile
import plotly.graph_objects as go
from datetime import datetime, timedelta
import pandas as pd
from pymongo import MongoClient
import uuid
from dotenv import load_dotenv
from googletrans import Translator
import pytesseract
import cv2
import numpy as np
import re

# Load environment variables
load_dotenv()

# Configure the generative AI with the API key from environment variables
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

# MongoDB setup
client = MongoClient(os.getenv("MONGODB_CONNECTION_STRING"))
db = client.prescription_app
users = db.users
prescriptions = db.prescriptions
reminders = db.reminders
feedback_collection = db.feedback

# Initialize translator
translator = Translator()

# Define supported languages
LANGUAGES = {
    'en': 'English',
    'es': 'Español',
    'fr': 'Français',
    'de': 'Deutsch',
    'hi': 'हिन्दी',
    'zh-cn': '中文(简体)'
}

# Function to translate text
def translate_text(text, dest_language):
    if dest_language == 'en':
        return text
    try:
        return translator.translate(text, dest=dest_language).text
    except Exception as e:
        st.error(f"Translation error: {str(e)}")
        return text
def text_to_speech(text, lang='en'):
    try:
        tts = gTTS(text=text, lang=lang)
        with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as fp:
            tts.save(fp.name)
            return fp.name, fp.read()
    except Exception as e:
        st.error(f"Error in text-to-speech conversion: {str(e)}")
        return None, None    

# Function to perform OCR
def perform_ocr(image):
    # Convert image to grayscale
    gray = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2GRAY)
    # Apply adaptive thresholding for better text extraction
    threshold = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                      cv2.THRESH_BINARY, 11, 2)
    # Perform OCR
    text = pytesseract.image_to_string(threshold)
    return text

# Function to analyze prescription image
def analyze_prescription_image(image):
    # Convert PIL Image to OpenCV format
    img_cv = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
    
    # Preprocess the image
    gray = cv2.cvtColor(img_cv, cv2.COLOR_BGR2GRAY)
    thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]

    # Perform OCR
    custom_config = r'--oem 3 --psm 6'
    extracted_text = pytesseract.image_to_string(thresh, config=custom_config)

    # Process the extracted text
    lines = extracted_text.split('\n')
    medications = []
    current_med = {}

    for line in lines:
        # Look for medication names (assuming they start with capital letters)
        med_match = re.match(r'^([A-Z][a-z]+ ?)+', line)
        if med_match:
            if current_med:
                medications.append(current_med)
            current_med = {'name': med_match.group().strip()}
        
        # Look for dosage information
        dosage_match = re.search(r'(\d+)\s*(mg|ml|mcg)', line, re.IGNORECASE)
        if dosage_match and current_med:
            current_med['dosage'] = f"{dosage_match.group(1)} {dosage_match.group(2)}"
        
        # Look for frequency information
        freq_match = re.search(r'(\d+)\s*times?\s*(daily|a day|per day)|every\s*(\d+)\s*hours?', line, re.IGNORECASE)
        if freq_match and current_med:
            if freq_match.group(1):
                current_med['frequency'] = f"{freq_match.group(1)} times a day"
            elif freq_match.group(3):
                current_med['frequency'] = f"Every {freq_match.group(3)} hours"

    # Add the last medication if it exists
    if current_med:
        medications.append(current_med)

    return medications

# Function to get the response from the Gemini model for custom questions
def get_custom_question_response(question, image_data):
    model = genai.GenerativeModel('gemini-pro-vision')
    input_prompt = """
    You are an expert in understanding medical prescriptions.
    You will receive input images as medical prescriptions &
    you will have to answer questions based on the input image.
    """
    try:
        # Combine the input prompt, question, and image data
        full_prompt = f"{input_prompt}\n\nQuestion: {question}"
        response = model.generate_content([full_prompt, image_data[0]])
        
        # Check if the response was blocked
        if response.parts:
            return response.text
        else:
            # Check safety ratings
            safety_ratings = response.prompt_feedback.safety_ratings
            blocked_reasons = [rating.category for rating in safety_ratings if rating.blocked]
            if blocked_reasons:
                return f"The response was blocked due to: {', '.join(blocked_reasons)}. Please rephrase your question or try a different image."
            else:
                return "The response was empty. Please try rephrasing your question or using a different image."
    
    except Exception as e:
        return f"An error occurred: {str(e)}"

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

# Function to process custom question
def process_custom_question(uploaded_file, question):
    if uploaded_file is not None:
        image_data = input_image_setup(uploaded_file)
        response = get_custom_question_response(question, image_data)
        return response
    else:
        return "No file uploaded"

# Function to analyze prescription history
def parse_prescription(prescription_text):
    # Use regex for better text parsing
    medicines = {}
    lines = prescription_text.split('\n')
    for line in lines:
        med_match = re.search(r'Medication:\s*(.*)', line)
        if med_match:
            med_name = med_match.group(1).strip()
            dosage = next((re.search(r'Dosage:\s*(.*)', l).group(1).strip() 
                           for l in lines[lines.index(line)+1:] if 'Dosage:' in l), None)
            if dosage:
                medicines[med_name] = dosage
    return medicines

def analyze_prescription_history(user_id):
    user_prescriptions = list(prescriptions.find({"user_id": user_id}).sort("date", -1))
    
    if not user_prescriptions:
        return "No prescription history found."

    analysis = []
    medication_history = {}

    for i, prescription in enumerate(user_prescriptions):
        date = prescription['date'].strftime("%Y-%m-%d")
        medications = parse_prescription(prescription['data'])

        analysis.append(f"\nPrescription Date: {date}")
        
        for med, info in medications.items():
            if med not in medication_history:
                medication_history[med] = []
            medication_history[med].append((date, info))

            if i > 0 and med in parse_prescription(user_prescriptions[i-1]['data']):
                prev_info = parse_prescription(user_prescriptions[i-1]['data'])[med]
                if info != prev_info:
                    analysis.append(f"- Change in {med}: {prev_info} -> {info}")
            else:
                analysis.append(f"- New medication: {med} ({info})")

    analysis.append("\nMedication History:")
    for med, history in medication_history.items():
        analysis.append(f"\n{med}:")
        for date, info in history:
            analysis.append(f"- {date}: {info}")

    return "\n".join(analysis)

# User authentication functions
def create_user(username, password):
    if users.find_one({"username": username}):
        return False
    user_id = str(uuid.uuid4())
    users.insert_one({"_id": user_id, "username": username, "password": password})
    return user_id

def authenticate_user(username, password):
    user = users.find_one({"username": username, "password": password})
    return user["_id"] if user else None

def save_prescription(user_id, prescription_data):
    prescriptions.insert_one({"user_id": user_id, "data": prescription_data, "date": datetime.now()})

def get_user_prescriptions(user_id):
    return list(prescriptions.find({"user_id": user_id}))

# Reminder functions
def set_reminder(user_id, medication, dosage, time):
    reminders.insert_one({
        "user_id": user_id,
        "medication": medication,
        "dosage": dosage,
        "time": time
    })

def get_user_reminders(user_id):
    return list(reminders.find({"user_id": user_id}))

# Function to display medication schedule
def display_medication_schedule(medications):
    fig = go.Figure()
    times = [datetime.now() + timedelta(hours=i) for i in range(24)]  # 24-hour schedule
    colors = ['blue', 'red', 'green', 'purple', 'orange', 'brown', 'pink', 'gray', 'olive', 'cyan']

    for i, med in enumerate(medications):
        med_name = med['name']
        frequency = med.get('frequency', '')
        doses = [0] * 24  # Initialize 24 hours with 0

        # Parse frequency and set doses
        if 'times a day' in frequency:
            times_per_day = int(frequency.split()[0])
            interval = 24 // times_per_day
            for j in range(0, 24, interval):
                doses[j] = 1
        elif 'Every' in frequency and 'hours' in frequency:
            interval = int(frequency.split()[1])
            for j in range(0, 24, interval):
                doses[j] = 1

        fig.add_trace(go.Scatter(x=times, y=doses, mode='lines+markers', name=med_name, 
                                 line=dict(color=colors[i % len(colors)], width=2),
                                 marker=dict(size=8)))

    fig.update_layout(title=translate_text('Medication Schedule', selected_language),
                      xaxis_title=translate_text('Time', selected_language),
                      yaxis_title=translate_text('Medication', selected_language),
                      legend_title=translate_text('Medications', selected_language),
                      height=500,
                      xaxis=dict(tickformat='%H:%M'),
                      yaxis=dict(tickmode='array', tickvals=[0, 1], 
                                 ticktext=['', translate_text('Take', selected_language)]))

    st.plotly_chart(fig)

    st.subheader(translate_text("Medication Details", selected_language))
    for med in medications:
        st.write(f"**{med['name']}**")
        st.write(f"Dosage: {med.get('dosage', 'Not specified')}")
        st.write(f"Frequency: {med.get('frequency', 'Not specified')}")
        st.write("---")

# Streamlit app
st.set_page_config(page_title="MediRead")

# Sidebar for language selection
st.sidebar.subheader("Language / Idioma / Langue / Sprache / भाषा / 语言")
selected_language = st.sidebar.selectbox("", options=list(LANGUAGES.keys()), format_func=lambda x: LANGUAGES[x])

# Header
st.header(translate_text("MediRead", selected_language))

# User Authentication
if 'user_id' not in st.session_state:
    st.session_state.user_id = None

if not st.session_state.user_id:
    auth_option = st.radio(translate_text("Choose an option", selected_language), 
                           [translate_text("Login", selected_language), 
                            translate_text("Sign Up", selected_language)])
    
    if auth_option == translate_text("Login", selected_language):
        username = st.text_input(translate_text("Username", selected_language))
        password = st.text_input(translate_text("Password", selected_language), type="password")
        if st.button(translate_text("Login", selected_language)):
            user_id = authenticate_user(username, password)
            if user_id:
                st.session_state.user_id = user_id
                st.success(translate_text("Logged in successfully!", selected_language))
            else:
                st.error(translate_text("Invalid credentials", selected_language))
    else:
        username = st.text_input(translate_text("Choose a username", selected_language))
        password = st.text_input(translate_text("Choose a password", selected_language), type="password")
        if st.button(translate_text("Sign Up", selected_language)):
            if create_user(username, password):
                st.success(translate_text("Account created successfully! Please log in.", selected_language))
            else:
                st.error(translate_text("Username already exists", selected_language))

if st.session_state.user_id:
    st.write(translate_text(f"Welcome, {users.find_one({'_id': st.session_state.user_id})['username']}!", selected_language))

    # Sidebar for navigation
    page = st.sidebar.radio(translate_text("Go to", selected_language), 
                            [translate_text("Custom Question", selected_language),
                             translate_text("Medication Schedule", selected_language),
                             translate_text("Reminders", selected_language),
                             translate_text("Prescription History", selected_language)])

    if page == translate_text("Custom Question", selected_language):
        st.subheader(translate_text("Custom Question", selected_language))
        
        uploaded_file = st.file_uploader(translate_text("Upload prescription image...", selected_language), type=["jpg", "jpeg", "png"])
        custom_question = st.text_area(translate_text("Enter your question about the prescription", selected_language))

        if uploaded_file is not None:
            image = Image.open(uploaded_file)
            st.image(image, caption=translate_text("Uploaded Prescription", selected_language), use_column_width=True)

        if st.button(translate_text("Get Answer", selected_language)):
            if uploaded_file is not None and custom_question:
                with st.spinner(translate_text("Processing...", selected_language)):
                    response = process_custom_question(uploaded_file, custom_question)
                    
                    st.subheader(translate_text("Answer", selected_language))
                    st.write(translate_text(response, selected_language))
                    
                    st.download_button(
                        translate_text("Download Answer", selected_language),
                        response,
                        file_name="custom_question_answer.txt"
                    )
                    
                    # Add audio output option with player
                    audio_file, audio_data = text_to_speech(response, lang=selected_language[:2])
                    if audio_file and audio_data:
                        st.subheader(translate_text("Audio Answer", selected_language))
                        st.audio(audio_data, format='audio/mp3')
                        st.download_button(
                            translate_text("Download Audio", selected_language),
                            audio_data,
                            file_name="audio_response.mp3",
                            mime="audio/mp3"
                        )
                    
                    save_prescription(st.session_state.user_id, response)
            else:
                st.write(translate_text("Please upload an image and enter a question.", selected_language))

   

    elif page == translate_text("Medication Schedule", selected_language):
        st.subheader(translate_text("Medication Schedule", selected_language))
        user_prescriptions = get_user_prescriptions(st.session_state.user_id)
        if user_prescriptions:
            selected_prescription = st.selectbox(
            translate_text("Select a prescription", selected_language), 
            [p['date'].strftime("%Y-%m-%d %H:%M") for p in user_prescriptions]
        )
            prescription_data = next(p for p in user_prescriptions if p['date'].strftime("%Y-%m-%d %H:%M") == selected_prescription)
    
            st.subheader(translate_text("Prescription Details", selected_language))
            st.write(translate_text(prescription_data['data'], selected_language))
    
            medications = parse_prescription(prescription_data['data'])
            if medications:
                fig = go.Figure()
                times = [datetime.now() + timedelta(hours=i*4) for i in range(6)]
                colors = ['blue', 'red', 'green', 'purple', 'orange', 'brown', 'pink', 'gray', 'olive', 'cyan']
        
                for i, (med, info) in enumerate(medications.items()):
                    dosage = int(info.split()[0])  # Assuming dosage is the first number in the string
                    doses = [1 if i % (24 // dosage) == 0 else 0 for i in range(6)]  # Changed to 1 for presence
                    fig.add_trace(go.Scatter(x=times, y=doses, mode='lines+markers', name=med, 
                                     line=dict(color=colors[i % len(colors)], width=2),
                                     marker=dict(size=8)))
        
                fig.update_layout(title=translate_text('Medication Schedule', selected_language),
                          xaxis_title=translate_text('Time', selected_language),
                          yaxis_title=translate_text('Medication', selected_language),
                          legend_title=translate_text('Medications', selected_language),
                          height=500,
                          xaxis=dict(tickformat='%H:%M'),
                          yaxis=dict(tickmode='array', tickvals=[0, 1], 
                                     ticktext=['', translate_text('Take', selected_language)]))
        
                st.plotly_chart(fig)
        
                st.subheader(translate_text("Medication Details", selected_language))
                for med, info in medications.items():
                    st.write(translate_text(f"{med}: {info}", selected_language))
            else:
                st.write(translate_text("No structured medication information found in this prescription.", selected_language))
        else:
            st.write(translate_text("No prescriptions found. Please upload a prescription first.", selected_language))

    elif page == translate_text("Reminders", selected_language):
        st.subheader(translate_text("Medication Reminders", selected_language))
        user_reminders = get_user_reminders(st.session_state.user_id)
        if user_reminders:
            for reminder in user_reminders:
                st.write(translate_text(f"Take {reminder['dosage']} of {reminder['medication']} at {reminder['time']}", selected_language))
        
        st.subheader(translate_text("Set New Reminder", selected_language))
        medication = st.text_input(translate_text("Medication Name", selected_language))
        dosage = st.number_input(translate_text("Dosage", selected_language), min_value=1, step=1)
        time = st.time_input(translate_text("Time", selected_language))
        if st.button(translate_text("Set Reminder", selected_language)):
            set_reminder(st.session_state.user_id, medication, dosage, time)
            st.success(translate_text("Reminder set successfully!", selected_language))

    elif page == translate_text("Prescription History", selected_language):
        st.subheader(translate_text("Prescription History", selected_language))
        user_prescriptions = get_user_prescriptions(st.session_state.user_id)
        for prescription in user_prescriptions:
            st.write(translate_text(f"Date: {prescription['date']}", selected_language))
            st.write(translate_text(prescription['data'], selected_language))
            st.write("---")

        st.subheader(translate_text("Prescription History Analysis", selected_language))
        if st.button(translate_text("Analyze Prescription History", selected_language)):
            analysis = analyze_prescription_history(st.session_state.user_id)
            st.text(translate_text(analysis, selected_language))

        if st.button(translate_text("Visualize Medication Changes", selected_language)):
            user_prescriptions = list(prescriptions.find({"user_id": st.session_state.user_id}).sort("date", 1))
            if user_prescriptions:
                dates = [p['date'] for p in user_prescriptions]
                medications = set()
                for p in user_prescriptions:
                    medications.update(parse_prescription(p['data']).keys())

                fig = go.Figure()
                for med in medications:
                    y_values = []
                    for p in user_prescriptions:
                        parsed = parse_prescription(p['data'])
                        if med in parsed:
                            y_values.append(1)  # Medication present
                        else:
                            y_values.append(0)  # Medication not present
                    fig.add_trace(go.Scatter(x=dates, y=y_values, mode='lines+markers', name=med))

                fig.update_layout(title=translate_text('Medication Changes Over Time', selected_language),
                                  xaxis_title=translate_text('Date', selected_language),
                                  yaxis_title=translate_text('Medication Presence', selected_language),
                                  yaxis=dict(tickmode='array', tickvals=[0, 1], 
                                             ticktext=[translate_text('Not Prescribed', selected_language), 
                                                       translate_text('Prescribed', selected_language)]))
                st.plotly_chart(fig)
            else:
                st.write(translate_text("No prescription history available for visualization.", selected_language))

# Sidebar feedback and help desk section
st.sidebar.header(translate_text("Feedback & Help", selected_language))

st.sidebar.subheader(translate_text("Feedback", selected_language))
feedback_text = st.sidebar.text_area(translate_text("Enter your feedback", selected_language))
feedback_rating = st.sidebar.slider(translate_text("Rate your experience", selected_language), 1, 5, 3)
if st.sidebar.button(translate_text("Submit Feedback", selected_language)):
    if feedback_text:
        feedback_data = {
            "user_id": st.session_state.user_id,
            "text": feedback_text,
            "rating": feedback_rating,
            "timestamp": datetime.now()
        }
        feedback_collection.insert_one(feedback_data)
        st.sidebar.success(translate_text("Thank you for your feedback!", selected_language))
    else:
        st.sidebar.warning(translate_text("Please enter some feedback before submitting.", selected_language))

st.sidebar.subheader(translate_text("Help & Documentation", selected_language))
if st.sidebar.button(translate_text("Show Help", selected_language)):
    st.sidebar.write(translate_text("""
    This application allows you to extract information from medical prescriptions.
    1. Select a language from the dropdown menu at the top of the sidebar.
    2. Upload an image of a prescription in the Prescription Analysis section.
    3. The app will use OCR to extract text from the image and analyze it.
    4. View your medication schedule and set reminders in the respective sections.
    5. Check your prescription history and visualize medication changes over time.
    6. Provide feedback to help us improve the application.
    """, selected_language))

# Function to improve Gemini responses based on feedback
def get_improved_gemini_response(input_prompt, image_data, feedback):
    recent_feedback = feedback_collection.find().sort("timestamp", -1).limit(5)
    feedback_text = "\n".join([f.get('text', '') for f in recent_feedback])
    improved_prompt = f"{input_prompt}\n\nRecent user feedback: {feedback_text}\n\nPlease consider this feedback and provide an improved response."
    
    model = genai.GenerativeModel('gemini-pro-vision')
    try:
        response = model.generate_content([improved_prompt, image_data[0]])
        return response.text
    except Exception as e:
        return f"An error occurred: {str(e)}"

# Function to analyze feedback
def analyze_feedback():
    st.subheader(translate_text("Feedback Analysis", selected_language))
    
    total_feedback = feedback_collection.count_documents({})
    avg_rating = feedback_collection.aggregate([
        {"$group": {"_id": None, "avg_rating": {"$avg": "$rating"}}}
    ]).next()['avg_rating']
    
    st.write(translate_text(f"Total Feedback: {total_feedback}", selected_language))
    st.write(translate_text(f"Average Rating: {avg_rating:.2f}", selected_language))
    
    st.subheader(translate_text("Recent Feedback", selected_language))
    recent_feedback = feedback_collection.find().sort("timestamp", -1).limit(10)
    for feedback in recent_feedback:
        feedback_date = feedback['timestamp'].strftime("%Y-%m-%d %H:%M")
        st.write(f"{translate_text('Date:', selected_language)} {feedback_date}")
        st.write(f"{translate_text('Rating:', selected_language)} {feedback['rating']}")
        st.write(f"{translate_text('Feedback:', selected_language)} {translate_text(feedback['text'], selected_language)}")
        st.write("---")
if st.sidebar.button(translate_text("Logout", selected_language)):
    st.session_state.user_id = None
    st.sidebar.success(translate_text("Logged out successfully!", selected_language))

# Feedback analysis section
if 'is_admin' in st.session_state and st.session_state['is_admin']:
    st.sidebar.subheader(translate_text("Admin Panel", selected_language))
    if st.sidebar.button(translate_text("Analyze Feedback", selected_language)):
        analyze_feedback()        

# Add feedback analysis to sidebar
if st.sidebar.button(translate_text("Analyze Feedback", selected_language)):
    analyze_feedback()

# Main content area closing bracket
if st.session_state.user_id:
    st.write(translate_text("Thank you for using MediRead!", selected_language))

# Closing message
st.write(translate_text("Powered by Streamlit and Gemini", selected_language))
