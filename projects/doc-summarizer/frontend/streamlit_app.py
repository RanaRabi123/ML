import streamlit as st
import requests
import pandas as pd
from docx import Document
from pypdf import PdfReader

# 1. CONFIGURATION & CUSTOM CSS (For Visuals & Animations)
st.set_page_config(page_title="Fake News Detector", layout="wide", page_icon="📰")

# Header with Icon
st.markdown("<h1 style='text-align: center; color: white; text-shadow: 2px 2px 4px rgba(0,0,0,0.3);'>📰 Summarizer</h1>", unsafe_allow_html=True)



def data_preprocessing(document, file_name):
    file_path = document
    if file_name.endswith('.txt'):
        with open(file_path, 'r', encoding='utf-8') as file:
            document = file.read()

    elif file_name.endswith('.docx'):
        doc = Document(file_path)
        file = ''
        for para in doc.paragraphs:
            file+=para.text
        document = file
            

    elif file_name.endswith('.pdf'):
        file = PdfReader(file_path)
        file_content=''
        for index, page in enumerate(file.pages):
            file_content += page.extract_text()
        document = file_content

    else:
        raise ValueError("Unsupported document type. Please provide a string, docx.Document, or pdfplumber.PDF.")

    return document


# User Input Section
with st.container(border=True):
    
        st.markdown("### Article Text")
        article_text = st.text_area("Enter the news article : ", height=100)
        uploaded_file = st.file_uploader("Or upload a document (txt, docx, pdf)", type=["txt", "docx", "pdf"], accept_multiple_files=False, max_upload_size=5)  
        

st.markdown("<br>", unsafe_allow_html=True) # Spacing
if st.button("✨ Summarize Article", use_container_width=True):
    # Prepare JSON
    if uploaded_file and not article_text:
        file_name = st.session_state.get("uploaded_file_name", uploaded_file.name)
        file  = data_preprocessing(uploaded_file, file_name)
        input_data = {"Article": file}
        st.write("File uploaded successfully!")

    elif article_text and not uploaded_file and article_text.strip() != "":
        input_data = {
            "Article": article_text}
    else:
        st.error("Please provide either text input or upload a file.")
        st.stop()

    with st.spinner('Summarizing article...'):
        try:
            # response = requests.post("https://your-backend-url.onrender.com/predict", json=input_data)
            response = requests.post("http://127.0.0.1:8000/predict", json=input_data)
            # response = requests.post("http://backend:8000/predict", json=input_data)  # due to docker 
            
            if response.status_code == 200:
                result = response.json()

                if "Summarize_text" in result.keys():
                    summary = result['Summarize_text']
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
                                    Summary: {summary}
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