import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import requests
from datetime import datetime


def plot_q_value_regression(df):
    """
    Plots regression graph for q-value calculation.
    """
    if "Log_D/Dmax" not in df.columns or "Log_pct_cpft" not in df.columns:
        st.warning("⚠️ Required columns for regression are missing.")
        return

    plt.figure(figsize=(6, 4))
    plt.scatter(df["Log_D/Dmax"], df["Log_pct_cpft"], label="Data Points", color="blue")

    # ✅ Perform regression and plot line
    slope, intercept = np.polyfit(df["Log_D/Dmax"], df["Log_pct_cpft"], 1)
    reg_line = slope * df["Log_D/Dmax"] + intercept
    plt.plot(df["Log_D/Dmax"], reg_line, label=f"Regression Line (q = {slope:.4f})", color="red")

    plt.xlabel("Log(D/Dmax)")
    plt.ylabel("Log(% CPFT)")
    plt.title("q-Value Regression")
    plt.legend()
    plt.grid()
    st.pyplot(plt)

# def plot_double_modified_q_regression(df):
#     """
#     Plots regression graph for Double Modified q-value calculation.
#     """
#     if "x_value" not in df.columns or "y_value" not in df.columns:
#         st.warning("⚠️ Required columns for regression are missing.")
#         return

#     plt.figure(figsize=(6, 4))
#     plt.scatter(df["x_value"], df["y_value"], label="Data Points", color="blue")

#     # ✅ Perform regression and plot line
#     slope, intercept = np.polyfit(df["x_value"], df["y_value"], 1)
#     reg_line = slope * df["x_value"] + intercept
#     plt.plot(df["x_value"], reg_line, label=f"Regression Line (q = {slope:.4f})", color="red")

#     plt.xlabel("ln(D - D_min) - ln(D_max - D_min)")
#     plt.ylabel("ln(% CPFT)")
#     plt.title("Double Modified q-Value Regression")
#     plt.legend()
#     plt.grid()
#     st.pyplot(plt)


st.title("GBD Optimization Dashboard - File Upload")

uploaded_file = st.file_uploader("Upload a CSV or Excel file", type=["csv", "xlsx"])

available_dates = None
selected_date = None
sample_data = None

if uploaded_file:
    st.write("File uploaded successfully.")
    
    files = {"file": (uploaded_file.name, uploaded_file.getvalue(), uploaded_file.type)}
    response = requests.post("http://127.0.0.1:8000/upload/", files=files)
    
    if response.status_code == 200:
        result = response.json()
        date_range = result.get("date_range", [])

        if date_range and len(date_range) == 2:
            min_date = datetime.strptime(date_range[0], "%Y-%m-%d")
            max_date = datetime.strptime(date_range[1], "%Y-%m-%d")

            formatted_date_range = [min_date.strftime("%d-%m-%Y"), max_date.strftime("%d-%m-%Y")]
            st.write(f"Available date range: **{formatted_date_range[0]} to {formatted_date_range[1]}**")

            selected_date = st.date_input("Select a date:", value=None, format="DD-MM-YYYY")

            if selected_date:
                selected_date_dt = datetime.combine(selected_date, datetime.min.time())

                if selected_date_dt < min_date:
                    st.error(f"⚠️ Please select a date from the available range: {formatted_date_range[0]} to {formatted_date_range[1]}")
                else:
                    formatted_selected_date = selected_date_dt.strftime("%d-%m-%Y")

                    if st.button("🔍 Verify Sample Data"):
                        sample_response = requests.get("http://127.0.0.1:8000/get_sample_data/", params={"selected_date": formatted_selected_date})

                        if sample_response.status_code == 200:
                            sample_data = sample_response.json().get("sample_data", {})
                            if sample_data:
                                st.success(f"✅ Sample Data for {formatted_selected_date}:")
                                for sheet, df_data in sample_data.items():
                                    st.write(f"**📄 Sheet: {sheet}**")
                                    df = pd.DataFrame(df_data)

                                    if 'Received Date' in df.columns:
                                        df['Received Date'] = pd.to_datetime(df['Received Date']).dt.strftime("%d-%m-%Y")

                                    st.dataframe(df)
                            else:
                                st.warning(f"No sample data found for {formatted_selected_date}")
                        else:
                            st.error("Error retrieving sample data.")

                    calculation_type = st.selectbox("Select Calculation Type:", ["Select", "GBD Values", "q-Values"])

                    calculate_button_label = None
                    packing_density = None

                    if calculation_type == "GBD Values":
                        packing_density_input = st.text_input("Enter Packing Density (single value or comma-separated list):")
                        
                        if packing_density_input:
                            try:
                                packing_density = [float(x.strip()) for x in packing_density_input.split(",")]
                                calculate_button_label = f"Calculate {calculation_type} with Packing Density: {packing_density}"
                            except ValueError:
                                st.error("❌ Please enter valid numeric values for packing density.")

                    elif calculation_type == "q-Values":
                        q_type = st.selectbox(
                            "Select q-Value Calculation Method:",
                            ["Select", "q-value using Andreasen Eq.", "q-value using Modified Andreasen Eq.", "q-value using Double Modified Andreasen Eq."]
                        )

                        if q_type == "q-value using Modified Andreasen Eq.":
                            packing_density_input = st.text_input("Enter Packing Density for Modified Andreasen Eq.:")
                            
                            if packing_density_input:
                                try:
                                    packing_density = [float(x.strip()) for x in packing_density_input.split(",")]
                                    calculate_button_label = f"Calculate {q_type} with Packing Density: {packing_density}"
                                except ValueError:
                                    st.error("❌ Please enter valid numeric values for packing density.")
                        else:
                            calculate_button_label = f"Calculate {q_type}"

                    # ✅ Button to trigger calculations
                     
                    if calculate_button_label and st.button(calculate_button_label):
                        st.info(f"🔄 Processing: {calculate_button_label}")

                        try:
                            payload = {"selected_date": formatted_selected_date}

                            if calculation_type == "GBD Values":
                                if packing_density:
                                    payload["packing_density"] = ",".join(map(str, packing_density))
                                response = requests.get("http://127.0.0.1:8000/calculate_gbd/", params=payload)

                            elif q_type == "q-value using Andreasen Eq.":
                                response = requests.get("http://127.0.0.1:8000/calculate_q_value/", params={"selected_date": formatted_selected_date})
                            
                            elif q_type == "q-value using Modified Andreasen Eq.":
                                payload["packing_density"] = ",".join(map(str, packing_density))
                                response = requests.get("http://127.0.0.1:8000/calculate_q_value_modified_andreason/", params=payload)
                            
                            elif q_type == "q-value using Double Modified Andreasen Eq.":
                                response = requests.get("http://127.0.0.1:8000/calculate_q_value_double_modified/", params = {"selected_date": formatted_selected_date})

                            if response.status_code == 200:
                                result = response.json()

                                if calculation_type == "GBD Values":
                                    total_volume = result.get("total_volume")
                                    specific_gravity = result.get("specific_gravity")
                                    gbd_values = result.get("gbd_values")

                                    st.write("## 📌 GBD Calculation Results")
                                    st.write(f"**🔹 Total Volume of the Mix:** `{total_volume:.4f}`")
                                    st.write(f"**🔹 Specific Gravity of the Mix:** `{specific_gravity:.4f}`")

                                    st.write("### ✅ **GBD Values**")
                                    for density, gbd in result["gbd_values"].items():
                                        st.write(f"- **GBD for {density}% Packing Density:** `{gbd:.4f} g/cc`")

                                elif q_type == "q-value using Andreasen Eq.":
                                    st.write("## 📌 q-Value Calculation Results")

                                    # ✅ Display q-values **first**
                                    q_values = result.get("q_values", [])
                                    if q_values:
                                        
                                        for q_data in q_values:
                                            st.markdown(f"#### ✅ q-value for {q_data['Date']}: **`{q_data['q-value']:.4f}`**", unsafe_allow_html=True)

                                    # ✅ Collapse the intermediate table and regression plot for cleaner UI
                                    with st.expander("📊 Show Processed Data Table & Regression Graph", expanded=False):
                                        df_intermediate = pd.DataFrame(result.get("intermediate_table"))
                                        if not df_intermediate.empty:
                                            st.write("### 📄 **Processed Data Table**")
                                            st.dataframe(df_intermediate)
                                        else:
                                            st.warning("⚠️ No intermediate data available.")

                                        # ✅ Plot Regression Graph
                                        st.write("### 📈 **Regression Graph**")
                                        plot_q_value_regression(df_intermediate)

                                elif q_type == "q-value using Modified Andreasen Eq.":
                                    st.write("## 📌 q-Value Calculation Results")

                                    q_values = result.get("q_values", [])
                                    if q_values:
                                        for q_data in q_values:
                                            for density, q_value in q_data.items():
                                                if density != "Date":
                                                    st.markdown(f"#### ✅ q-value for {q_data['Date']} at {density}% Packing Density: **`{q_value:.4f}`**", unsafe_allow_html=True)

                                    # ✅ Display Intermediate CPFT Error Table
                                    cpft_error_table = result.get("cpft_error_table", [])
                                    if cpft_error_table:
                                        st.write("### 📊 **CPFT Error Table**")
                                        df_cpft_error = pd.DataFrame(cpft_error_table)
                                        st.dataframe(df_cpft_error)

                                elif q_type == "q-value using Double Modified Andreasen Eq.":
                                    st.write("## 📌 Double Modified q-Value Calculation Results")

                                    double_mod_q_values = result.get("double_modified_q_values", [])
                                    if double_mod_q_values:
                                        for q_data in double_mod_q_values:
                                            st.markdown(f"#### ✅ Double Modified q-value for {q_data['Date']}: **`{q_data['Double_modified_q']:.4f}`**", unsafe_allow_html=True)

                                     # ✅ Plot Regression Graph (No Intermediate Table)
                            #         df_regression = pd.DataFrame(result.get("intermediate_table", []))
                            #         if not df_regression.empty and "Log_D/Dmax" in df_regression.columns and "Log_pct_cpft" in df_regression.columns:
                            #             with st.expander("📊 Show Processed Data Table & Regression Graph", expanded=False):
                            #                 st.write("### 📄 **Processed Data Table for Regression**")
                            #                 st.dataframe(df_regression)
                                            
                            #                 # ✅ Show Regression Graph
                            #                 st.write("### 📈 **Regression Graph**")
                            #                 plot_q_value_regression(df_regression)
                            #         else:
                            #             st.warning("⚠️ No data available for regression graph.")
                            else:
                                st.error(f"❌ Error calculating {calculation_type}. Backend response: {response.text}")

                        except Exception as e:
                            st.error(f"❌ Exception: {str(e)}")

                            
                