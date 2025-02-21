from fastapi import FastAPI, File, UploadFile, HTTPException
import pandas as pd
import numpy as np
from io import BytesIO
from app.model import calculate_gbd_from_excel  # ✅ Ensure correct import

app = FastAPI()

# ✅ Define these variables at the top (before any function)
required_sheets = ["7-12", "14-30", "36-70", "80-180", "220F"]
column_to_drop = ["Samples No."]
proportions = {'7-12': 0.35, '14-30': 0.20, '36-70': 0.15, '80-180': 0.10, '220F': 0.20}
packing_densities = [0.85, 0.82]

@app.post("/upload/")
async def process_file(file: UploadFile = File(...)):
    try:
        # Ensure file type is valid
        if not file.filename.endswith((".xlsx", ".csv")):
            raise HTTPException(status_code=400, detail="Invalid file type. Please upload a CSV or Excel file.")

        # Read file into memory
        contents = await file.read()
        file_obj = BytesIO(contents)

        # Check if the file is empty
        if len(contents) == 0:
            raise HTTPException(status_code=400, detail="Uploaded file is empty. Please upload a valid file.")


        # Ensure required_sheets is passed correctly
        gbd_df = calculate_gbd_from_excel(file_obj, required_sheets, column_to_drop, proportions, packing_densities)

        # Check if GBD data is empty
        if gbd_df.empty:
            raise HTTPException(status_code=400, detail="Processed file does not contain valid data.")


        return {"message": "File processed successfully", "data": gbd_df.to_dict(orient="records")}

    except ValueError as ve:
        raise HTTPException(status_code=400, detail=f"Value Error: {str(ve)}")
    except pd.errors.ExcelFileError:
        raise HTTPException(status_code=400, detail="Error reading the uploaded Excel file. Please check the file format.")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Internal Server Error: {str(e)}")