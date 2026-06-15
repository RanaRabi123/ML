import streamlit as st
import requests


# 1. CONFIGURATION & CUSTOM CSS (For Visuals & Animations)
st.set_page_config(page_title="Tumor Detector ", layout="wide", page_icon="🧠")


# Header with Icon
st.markdown("<h1 style='text-align: center; color: white; text-shadow: 2px 2px 4px rgba(0,0,0,0.3);'> 🧠 Tumor Detector</h1>", unsafe_allow_html=True)
st.markdown("<br>", unsafe_allow_html=True) # Spacing


col1, col2 = st.columns(2)
# User Input Section
with col1:
    with st.container(border=True, width=550, height=200):
    
        st.markdown("### Tumor Detection from MRI Scans")
        st._upload_button_label = "Upload MRI Image"
        uploaded_file = st.file_uploader("Choose an MRI image : ", type=["jpg", "jpeg", "png"], accept_multiple_files=False, max_upload_size=10)     

    if st.button("Check Tumor", width=550):
        if uploaded_file is not None:
            files = {
            "file": (
                uploaded_file.name,
                uploaded_file.getvalue(),
                uploaded_file.type
            )
        }
        else:
            st.warning("Please upload an MRI image.")
            st.stop()

        with st.spinner('Checking for tumors...'):
            try:
                # response = requests.post("https://your-backend-url.onrender.com/predict", json=input_data)
                response = requests.post("http://127.0.0.1:8000/predict", files=files)
                # response = requests.post("http://backend:8000/predict", json=input_data)  # due to docker 
                
                if response.status_code == 200:
                    result = response.json()

                    if result:
                        model_result = result['Prediction']
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
                                        Prediction: {model_result}
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


st.markdown("<br>", unsafe_allow_html=True) # Spacing
    # Prepare JSON
with col2:
        with st.container(border=True, width=550):
            if uploaded_file is not None:
                st.image(uploaded_file, caption='Uploaded MRI Image', )
