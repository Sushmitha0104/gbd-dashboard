from fastapi import FastAPI, File, UploadFile, HTTPException
import pandas as pd
import numpy as np
from scipy.interpolate import interp1d
from scipy.stats import linregress
from io import BytesIO
from app.model import calculate_gbd_and_q_values  # ✅ Ensure correct import

app = FastAPI()

# ✅ Define these variables at the top
required_sheets = ["7-12", "14-30", "36-70", "80-180", "220F"]
column_to_drop = ["Samples No."]
proportions = {'7-12': 0.35, '14-30': 0.20, '36-70': 0.15, '80-180': 0.10, '220F': 0.20}
packing_densities = [0.85, 0.82]
sheet_constants = {'7-12': 65, '14-30': 45, '36-70': 30, '80-180': 20, '220F': 0}
mesh_size_to_particle_size = {'+6': 3360, '+8': 2380, '+10': 2000, '+14': 1410, '+16': 1190, '+12': 1680, '+18': 1000, '+30': 595, '+40': 420, '+50': 297, '+70': 210, '+100': 149, '+80': 177, '+120': 125, '+140': 105, '+200': 74, '+230': 63, '+270': 53, '+325': 44}


@app.post("/upload/")
async def process_file(file: UploadFile = File(...)):
    try:
        # ✅ Validate file type
        if file.content_type not in ["application/vnd.openxmlformats-officedocument.spreadsheetml.sheet", "text/csv"]:
            raise HTTPException(status_code=400, detail="Invalid file type. Please upload a CSV or Excel file.")

        # ✅ Read file into memory
        contents = await file.read()

        # ✅ Check if the file is empty
        if len(contents) == 0:
            raise HTTPException(status_code=400, detail="Uploaded file is empty. Please upload a valid file.")

        # ✅ Convert file contents to a file-like object
        file_obj = BytesIO(contents)
        file_obj.seek(0)  # Reset file pointer before reading

        # ✅ Ensure required_sheets is passed correctly
        combined_df = calculate_gbd_and_q_values(file_obj, required_sheets, column_to_drop, proportions, packing_densities, sheet_constants, mesh_size_to_particle_size)

        # ✅ Check if the DataFrame is empty
        if combined_df.empty:
            raise HTTPException(status_code=400, detail="Processed file does not contain valid data.")

        return {"message": "File processed successfully", "data": combined_df.to_dict(orient="records")}

    # except pd.errors.EmptyDataError:
    #     raise HTTPException(status_code=400, detail="Uploaded file is empty or unreadable.")

    except FileNotFoundError:
        raise HTTPException(status_code=400, detail="Error reading the uploaded file.")

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Internal Server Error: {str(e)}")
