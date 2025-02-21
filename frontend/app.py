import streamlit as st
import pandas as pd
import requests

st.title("GBD Optimization Dashboard - File Upload")

# File Upload
uploaded_file = st.file_uploader("Upload a CSV or Excel file", type=["csv", "xlsx"])

if uploaded_file:
    st.write("File uploaded successfully.")

    # Send file to FastAPI
    files = {"file": (uploaded_file.name, uploaded_file.getvalue(), uploaded_file.type)}  # ✅ Fix here
    response = requests.post("http://127.0.0.1:8000/upload/", files=files)

    if response.status_code == 200:
        result = response.json()
        st.success(result["message"])

        # Display data preview
        if "data" in result:  # ✅ Fix key name (FastAPI returns "data", not "data_preview")
            df = pd.DataFrame(result["data"])
            st.write("Processed Data:")
            st.write(df)

    else:
        st.error(f"Error in API request: {response.text}")  # ✅ Show error details
