import streamlit as st
import requests
import pandas as pd

# 1. CONFIGURATION & CUSTOM CSS (For Visuals & Animations)
st.set_page_config(page_title="Fake News Detector", layout="wide", page_icon="📰")


st.markdown(
    """
    <style>
    /* Import Google Fonts */
    @import url('https://fonts.googleapis.com/css2?family=Poppins:wght@300;400;600&display=swap');

    /* Global Styles */
    html, body, [class*="css"]  {
        font-family: 'Poppins', sans-serif;
    }

    /* Animated Background */
    .stApp {
        background: linear-gradient(-45deg, #ee7752, #e73c7e, #23a6d5, #23d5ab);
        background-size: 400% 400%;
        animation: gradient 8s ease infinite;
        color: white;
    }

    @keyframes gradient {
        0% { background-position: 0% 50%; }
        50% { background-position: 100% 50%; }
        100% { background-position: 0% 50%; }
    }

    /* Glassmorphism Container Effect */
    .css-1d391kg { /* Main container padding */
        padding-top: 2rem;
    }

    /* Style the Input Container */
    div[data-testid="stVerticalBlock"] > div[style*="flex-direction"] {
        background-color: rgba(255, 255, 255, 0.15);
        backdrop-filter: blur(10px);
        -webkit-backdrop-filter: blur(10px);
        border-radius: 5px;
        padding: 2rem;
        border: 1px solid rgba(255, 255, 255, 0.2);
        box-shadow: 0 8px 32px 0 rgba(31, 38, 135, 0.37);
        color: white;
    }

    /* Input Fields Styling */
    label {
        font-weight: 600;
        color: #fff !important;
    }
    
    /* Dropdown & Number Inputs background */
    div[data-baseweb="select"] span, 
    div[data-baseweb="input"] {
        background-color: rgba(255, 255, 255, 0.2) !important;
        color: white !important;
    }
    
    /* Button Styling */
    .stButton > button {
        width: 100%;
        border-radius: 50px;
        height: 5em;
        font-weight: bold;
        transition: all 0.9s ease;
        background-color: #ffffff;
        color: #23a6d5;
        border: none;
        box-shadow: 0 4px 15px rgba(0,0,0,0.2);
    }

    .stButton > button:hover {
        transform: translateY(-3px);
        box-shadow: 0 6px 20px rgba(0,0,0,0.3);
        background-color: #b3b3b3;
    }

    /* Success/Error Message Styling */
    .stSuccess, .stError {
        border-radius: 10px;
        text-align: center;
        font-weight: bold;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# --- MAIN APP LOGIC ---

# Header with Icon
st.markdown("<h1 style='text-align: center; color: white; text-shadow: 2px 2px 4px rgba(0,0,0,0.3);'>📰 Fake News Detector</h1>", unsafe_allow_html=True)



# User Input Section
with st.container(border=True):
    
        st.markdown("### Article Text")
        article_text = st.text_area("Enter the news article : ", height=200)
        

# Predict Button with Animation
st.markdown("<br>", unsafe_allow_html=True) # Spacing
if st.button("✨ Predict Fake News", use_container_width=True):
    # Prepare JSON
    input_data = {
        "Article": article_text
    }

    # Simulate loading for effect (optional)
    with st.spinner('Checking for news...'):
        # Send request to backend
        try:
            # response = requests.post("https://your-backend-url.onrender.com/predict", json=input_data)
            # response = requests.post("http://127.0.0.1:8001/predict", json=input_data)
            response = requests.post("http://backend:8000/predict", json=input_data)  # due to docker 
            
            if response.status_code == 200:
                result = response.json()

                if "Predicted_Class" in result.keys():
                    predicted_class = result['Predicted_Class']
                    st.markdown(
                                f"""
                                <div style="
                                    background: linear-gradient(90deg, #00c6ff, #0072ff);
                                    padding: 30px;
                                    border-radius: 15px;
                                    text-align: center;
                                    font-size: 37px;
                                    font-weight: bold;
                                    color: black;
                                    box-shadow: 0 20px 20px rgba(0,0,0,0.3);
                                ">
                                    Predicted Class: {predicted_class}
                                </div>
                                """,  unsafe_allow_html=True
                            )
                else:
                    st.error(f"Backend Error: {result}")
            else:
                st.error(f"HTTP Error {response.status_code}")
                st.write(response.text)
            
        except requests.exceptions.RequestException as e:
            st.error(f"Connection Error: {e}")