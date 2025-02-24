import pandas as pd
import numpy as np
from scipy.interpolate import interp1d
from scipy.stats import linregress
from scipy.optimize import curve_fit, minimize_scalar, differential_evolution

# Step 1: Reading Excel files and defining sheets

def read_excel_file(file, required_sheets):
    """
    Reads the Excel file and returns the required sheets as DataFrames.

    Parameters:
        file_path (str): Path to the Excel file.
        required_sheets (list): List of sheet names to be read.

    Returns:
        dict: Dictionary of DataFrames for each required sheet.
    """
    xls = pd.ExcelFile(file)
    sheets = {sheet: pd.read_excel(xls, sheet_name=sheet) for sheet in required_sheets if sheet in xls.sheet_names}
    return sheets

# Step 2: Cleaning data

def clean_data(sheets, column_to_drop):
    """
    Cleans the data by removing headers, dropping unnecessary rows and columns.

    Parameters:
        sheets (dict): Dictionary of DataFrames for each sheet.
        column_to_drop (list): List of column names to be dropped if present.

    Returns:
        dict: Cleaned DataFrames for each sheet.
    """
    clean_data = {}
    for sheet_name, df in sheets.items():
        # Drop the first row (header)
        df = df.drop(index=0)

        # Set the column names using the first row (now the header)
        df.columns = df.iloc[0].astype(str)

        # Drop the first row again and reset the index
        df = df[1:].reset_index(drop=True)

        # Drop rows containing "Specification"
        df = df[~df.iloc[:, 0].astype(str).str.contains("Specification", na=False)]

        # Drop unnecessary columns
        df = df.drop(columns=[col for col in column_to_drop if col in df.columns], errors="ignore")

        clean_data[sheet_name] = df
        
    return clean_data

# Step 3: Standardizing column names and date conversion

def standardize_column_names_and_convert_dates(clean_data):
    """
    Standardizes column names and converts date strings to datetime objects.

    Parameters:
        clean_data (dict): Dictionary of cleaned DataFrames for each sheet.

    Returns:
        dict: DataFrames with standardized column names and converted date columns.
    """
    for sheet_name, df in clean_data.items():
        df.columns = df.columns.astype(str)  # Convert all column names to strings
        for col in df.columns:
            if "Received" in col and "Date" in col:
                df.rename(columns={col: "Received Date"}, inplace=True)
        if "Received Date" in df.columns:
            df["Received Date"] = pd.to_datetime(df["Received Date"], format="%d.%m.%y", errors="coerce")
    return clean_data

# Step 4: Matching dates across sheets

def match_dates_across_sheets(standardized_sheets, required_sheets, dates_list1, dates_list2):
    """
    Matches dates across sheets or finds the nearest past date if an exact match is unavailable.

    Parameters:
        cleaned_sheets (dict): Dictionary of cleaned DataFrames for each sheet.
        required_sheets (list): List of sheet names to be matched.

    Returns:
        dict: Dictionary with main dates as keys and matched dates for each sheet as values.
    """

    all_dates = {}
    for sheet_name in required_sheets:
        dates = standardized_sheets[sheet_name]["Received Date"].dropna().unique()
        all_dates[sheet_name] = pd.to_datetime(dates)

    main_dates = all_dates[required_sheets[0]]  # Use dates from the first sheet as main reference
    matched_dates = {}
    for date in main_dates:
        matched_dates[date] = {}
        for sheet_name, dates in all_dates.items():
            possible_dates = dates[dates <= date]
            matched_dates[date][sheet_name] = possible_dates.max() if len(possible_dates) > 0 else None
    return matched_dates

# Step 5: Extracting sample data
def extract_sample_data(clean_data, matching_dates, unique_dates):
    """
    Extracts sample data for each date using the matching dates for each sheet.

    Parameters:
        clean_data (dict): Dictionary of cleaned DataFrames for each sheet.
        matching_dates (dict): Dictionary with main dates as keys and matched dates for each sheet.
        unique_dates (array-like): List or array of unique dates to extract data for.

    Returns:
        dict: Sample data for each date and sheet.
    """
    sample_data = {}  # Dictionary to store sample data for each date

    for date in unique_dates:
        sample_data[date] = {}  # Sample data for this specific date

        for sheet, df in clean_data.items():
            # Get the target date for this sheet and main date
            target_date = matching_dates.get(date, {}).get(sheet, None)

            if target_date is not None:
                # Extract rows for the target_date
                sample_rows = df[df.iloc[:, 0].dt.date == target_date.date()]
            else:
                sample_rows = None  # No matching rows for this date

            # Store the extracted rows for this date and sheet
            sample_data[date][sheet] = sample_rows

    return sample_data


# Step 6: Converting sample data to numeric and calculating averages

def convert_to_numeric_and_calculate_average(sample_data):
    """
    Converts extracted sample data to numeric and calculates averages.
    
    Parameters:
        sample_data (dict): Dictionary containing sample data for each date and sheet.
        
    Returns:
        dict: Averages of numeric columns for each date and sheet.
    """
    averages = {}
    for date, sheets_data in sample_data.items():
        averages[date] = {}
        
        for sheet_name, df in sheets_data.items():
            if df is None or df.empty:
                averages[date][sheet_name] = None
                continue

            numeric_df = df.iloc[:, 1:].apply(pd.to_numeric, errors="coerce")
            numeric_df = numeric_df.dropna(axis=1, how = "all")

            averages[date][sheet_name] = numeric_df.mean()
    return averages

# Step 7: Calculating volume using weight proportions

def calculate_volume(average_data, proportions):
    """
    Calculates the volume for each date and sheet using specific gravity and predefined proportions.
    
    Parameters:
        average_data (dict): Averages of numeric columns for each date and sheet.
        proportions (dict): Predefined weight proportions for each sheet.
        
    Returns:
        dict: Calculated volume values for each date.
    """
    volume_data = {}

    for date, sheet_data in average_data.items():
        volume_values = []

        for sheet, proportion in proportions.items():  # Loop through predefined sheets
            if sheet in sheet_data and sheet_data[sheet] is not None:
                df = sheet_data[sheet]

                # Get specific gravity (last row)
                sg = df.iloc[-1]

                # Convert to float if it's a Series
                sg_value = sg.values[0] if isinstance(sg, pd.Series) else float(sg)

                # Calculate volume
                volume = proportion / sg_value
                volume_values.append(volume)
            else:
                # If sheet data is missing, append None or some placeholder
                volume_values.append(None)

        # Store the volume values for each date
        volume_data[str(date.date())] = volume_values

    return volume_data


# Step 8: Summing volumes to get total volume

def sum_volumes(volumes):
    """
    Sums the total volume of the mix from all sheets for each date.

    Parameters:
        volumes (dict): Calculated volume values for each date.
        
    Returns:
        dict: Total volume for each date.
    """
    return {date : sum(volume) for date, volume in volumes.items()}

# Step 9: Calculating specific gravity of the mix

def calculate_sg_mix(volume_sum):
    """
    Calculates the specific gravity (SG) of the mix for each date.

    Parameters:
        volume_sum (dict): Total volume for each date.

    Returns:
        dict: Specific gravity of the mix for each date.
    """
    
    sg_mix_data = {}
    for date, volume in volume_sum.items():
        sg_mix = np.round(1 / volume, 2)  # Specific gravity calculation
        sg_mix_data[date] = sg_mix
    return sg_mix_data


# Step 10: Calculating GBD values
def calculate_gbd_values(sg_mix_data, packing_densities):
    """
    Calculates GBD (Gross Bulk Density) values for each date using specific gravities and packing densities.

    Parameters:
        sg_mix_data (dict): Specific gravity of the mix for each date.
        packing_densities (list): List of two packing densities to be used for calculations.

    Returns:
        list: List of lists containing date, GBD for PD1, and GBD for PD2.
    """
    gbd_data = []
    for date, sg in sg_mix_data.items():
        GBD_pd1 = np.round(sg * packing_densities[0], 4)
        GBD_pd2 = np.round(sg * packing_densities[1], 4)
        gbd_data.append([date, GBD_pd1, GBD_pd2])
    
    gbd_df = pd.DataFrame(gbd_data, columns=["Date", "GBD (g/cc) with 85% packing density", "GBD (g/cc) with 82% packing density"])
    # print(gbd_df)
    return gbd_df

# Step 11: Calculate reverse cumulative sum
def drop_last_3_and_reverse_cumsum(average_data):
    """
    Drops the last 3 items in average data and calculates reverse cumulative sum of remaining items.
    """
    weights = {}
    cum_sum = {}

    for date, sheet_data in average_data.items():
        weights[date] = {}
        cum_sum[date] = {}

        for sheet, df in sheet_data.items():
            if df is None:
                weights[date][sheet] = None
                cum_sum[date][sheet] = None
                continue

            # Drop the last 3 rows (Total, LBD, SG)
            df = df.iloc[:-3]
            weights[date][sheet] = df

            # Calculate reverse cumulative sum
            df = df.to_frame()  # Convert Series to DataFrame
            mesh_sizes = df.index.values
            weight_values = df.iloc[:, 0].values

            reverse_cumsum_weights = np.cumsum(weight_values[::-1])[::-1]
            reverse_cumsum_weights = reverse_cumsum_weights[1:]

            mesh_sizes = mesh_sizes[:-1]  # Sync mesh sizes with cumulative sum values

            df_result = pd.DataFrame({
                'Mesh Size': mesh_sizes,
                'Cumulative Sum': reverse_cumsum_weights
            })

            cum_sum[date][sheet] = df_result

    return weights, cum_sum

# Step 12: Calculate cpft
def calculate_cpft(cum_sum, multipliers):
    """
    Calculates CPFT values for each date and sheet using provided multipliers.
    """
    cpft = {}
    for date, sheet_data in cum_sum.items():
        cpft[date] = {}
        for sheet, df in sheet_data.items():
            if df is not None:
                multiplier = multipliers.get(sheet) 
                if multiplier:
                    df['cpft'] = df['Cumulative Sum'] * multiplier
                    cpft[date][sheet] = df
    return cpft

# Step -13: Calculate pct_cpft values
def calculate_pct_cpft(cum_sum, sheet_constants):
    """
    Calculates Percentage CPFT values for each date and sheet using sheet constants.
    """
    pct_cpft = {}
    for date, sheet_data in cum_sum.items():
        pct_cpft[date] = {}
        for sheet, df in sheet_data.items():
            if df is not None:
                constant = sheet_constants.get(sheet, 0) 
                df['pct_cpft'] = df['cpft'] + constant
                df['Sheet'] = sheet
                df_result = df[['Sheet', 'Mesh Size', 'cpft', 'pct_cpft']]
                pct_cpft[date][sheet] = df_result
    return pct_cpft

# step -14: Merge into single df
def merge_pct_cpft_into_df(mesh_size_to_particle_size, pct_cpft):
    """
    Merges pct_cpft values into a single DataFrame for each date.
    """
    final_df = {}
    for date, sheet_data in pct_cpft.items():
        final_df[date] = pd.concat(sheet_data.values(), ignore_index=True)
        final_df[date]['Particle Size'] = final_df[date]['Mesh Size'].map(mesh_size_to_particle_size)

    return final_df

# Step-15 : Interpolation
def calculate_interpolated_values(final_df, rows_to_interpolate=[5, 15, 18, 19]):
    """
    Calculates interpolated pct_cpft values for specified rows using linear interpolation.
    """
    for date, df in final_df.items():
        

        # Initialize with original values
        df['pct_cpft_interpolated'] = df['pct_cpft']  

        # Remove rows to be interpolated and store the valid data
        valid_rows = df.drop(rows_to_interpolate)

        for i in rows_to_interpolate:
            if i >= len(df):  # Ensure index is within bounds
                continue  

            current_particle_size = df.loc[i, 'Particle Size']

            # Select nearest smaller and larger values from valid_rows
            small_rows = valid_rows[valid_rows['Particle Size'] < current_particle_size]
            large_rows = valid_rows[valid_rows['Particle Size'] > current_particle_size]

            if not small_rows.empty and not large_rows.empty:
                # Find nearest smaller and larger indices
                nearest_small_index = small_rows['Particle Size'].sub(current_particle_size).abs().idxmin()
                nearest_large_index = large_rows['Particle Size'].sub(current_particle_size).abs().idxmin()

                # Extract data for interpolation
                x_vals = [valid_rows.loc[nearest_small_index, 'Particle Size'], 
                          valid_rows.loc[nearest_large_index, 'Particle Size']]
                y_vals = [valid_rows.loc[nearest_small_index, 'pct_cpft'], 
                          valid_rows.loc[nearest_large_index, 'pct_cpft']]

                
                # Create interpolation function
                interp_func = interp1d(x_vals, y_vals, kind='linear', fill_value='extrapolate')

                # Get interpolated value
                df.at[i, 'pct_cpft_interpolated'] = np.round(interp_func(current_particle_size), 3)

        

    return final_df

#  Step-16: Dropping duplicate rows
def drop_and_reset_indices(final_df, rows_to_drop=[9, 13, 14, 17]):
    """
    Drops specified rows and resets index for each date's DataFrame in final_df.
    """
    for date, df in final_df.items():
        df = df.drop(rows_to_drop, axis=0)
        final_df[date] = df.reset_index(drop=True)
    
    return final_df

# Step-17: Normalize particle size
def normalize_particle_size(final_df):
    """
    Normalizes the Particle Size by dividing by the maximum Particle Size for each date.
    """
    for date, df in final_df.items():
      
        
        d_max = df["Particle Size"].max()
        df['Normalized_D'] = df['Particle Size'] / d_max
      
    
    return final_df

# Step-18: q_value prediction

def q_value_prediction(final_df):
    """
    Reads the DataFrame after normalization, calculates logarithmic values,
    performs q-value prediction using linear regression, and returns a DataFrame
    containing logarithm values and q-values.
    """
    # Initialize a list to store q-values and logarithmic data
    log_q_values_data = []

    # Loop through all dates in the normalized DataFrame
    for date, df in final_df.items():
        
        # Ensure required columns exist
        if "Normalized_D" not in df.columns or "pct_cpft_interpolated" not in df.columns:
            print(f"Skipping {date} as required columns are missing.")
            continue

        # Apply logarithm
        df["Log_D/Dmax"] = np.log(df["Normalized_D"])
        df["Log_pct_cpft"] = np.log(df["pct_cpft_interpolated"])

        # Perform linear regression
        slope, intercept, r_value, p_value, std_err = linregress(df["Log_D/Dmax"], df["Log_pct_cpft"])

        # Store logarithmic values and q-value
        log_q_values_data.append({
            "Date": date,
            "q-value": round(slope, 4)  # Storing q-value (slope)
        })

    # Convert list of dictionaries into a DataFrame
    q_values_df = pd.DataFrame(log_q_values_data)
    return q_values_df



# Corrected Modified Andreason Equation
def modified_andreason_eq(q, D, D_min, D_max):
    return ((D ** q - D_min ** q) / (D_max ** q - D_min ** q)) * 100

# Objective Function: Squared Differences for Better Penalization
def objective_diff(q, particle_sizes, target_cpft):
    calculated_cpft = modified_andreason_eq(q, particle_sizes, particle_sizes.min(), particle_sizes.max())
    return np.sum((calculated_cpft - target_cpft) ** 2)  # Squared differences

# Differential Evolution Optimization
def optimize_q_de(particle_sizes, cpft_values):
    result = differential_evolution(
        objective_diff,
        bounds=[(0.20, 0.60)],  # Wider bounds for more flexibility
        args=(particle_sizes, cpft_values),
        strategy='best1bin',
        maxiter=2000,  # Increased iterations for better convergence
        tol=1e-10  # Tighter tolerance for more precision
    )
    return result.x[0]

# Function to Predict q-values for all dates
def predict_q_values(mod_q):
    import pandas as pd  # Make sure pandas is imported

    optimized_q_list = []  # Collect results as a list of dictionaries

    for date, df in mod_q.items():
       

        # Extract relevant data
        particle_sizes = df['Particle Size (μm)'].values
        cpft_85 = df['pct_85_poros_CPFT'].values
        cpft_82 = df['pct_82_poros_CPFT'].values
        
        # Optimize for 85% packing density
        q_85 = optimize_q_de(particle_sizes, cpft_85)
        
        # Optimize for 82% packing density
        q_82 = optimize_q_de(particle_sizes, cpft_82)
        
        # Store results as a dictionary
        optimized_q_list.append({
            'Date': date,
            'q_85': q_85,
            'q_82': q_82
        })
        
        
    
    # Convert the list of dictionaries into a DataFrame
    optimized_q_df = pd.DataFrame(optimized_q_list)
    
    # Sort the DataFrame by date for better visualization
    optimized_q_df = optimized_q_df.sort_values(by='Date').reset_index(drop=True)
    
    return optimized_q_df


def calculate_all_values(file_path, required_sheets, column_to_drop, proportions, packing_densities, sheet_constants, mesh_size_to_particle_size):
    # Step 1: Read and clean data
    sheets = read_excel_file(file_path, required_sheets)
    cleaned_sheets = clean_data(sheets, column_to_drop)
    standardized_sheets = standardize_column_names_and_convert_dates(cleaned_sheets)
    
    # Step 2: Extract Dates and Match Dates
    dates = standardized_sheets[required_sheets[0]]["Received Date"].dropna().unique()
    matched_dates = match_dates_across_sheets(standardized_sheets, required_sheets, dates, dates)
    
    # Step 3: Extract Sample Data and Calculate Averages
    sample_data = extract_sample_data(standardized_sheets, matched_dates, dates)
    averages = convert_to_numeric_and_calculate_average(sample_data)
    
    # Step 4: Calculate GBD Values
    volume_data = calculate_volume(averages, proportions)
    total_volume = sum_volumes(volume_data)
    sg_mix_data = calculate_sg_mix(total_volume)
    gbd_df = calculate_gbd_values(sg_mix_data, packing_densities)
    
    # Step 5: Calculate q-values
    weights, cum_sum = drop_last_3_and_reverse_cumsum(averages)
    cpft = calculate_cpft(cum_sum, proportions)
    pct_cpft = calculate_pct_cpft(cum_sum, sheet_constants)
    final_df = merge_pct_cpft_into_df(mesh_size_to_particle_size, pct_cpft)
    final_df = calculate_interpolated_values(final_df)
    final_df = drop_and_reset_indices(final_df)
    final_df = normalize_particle_size(final_df)
    q_values_df = q_value_prediction(final_df)
    
    # Step 6: Calculate Optimized q-values
    mod_q = {}
    for date, df in final_df.items():
        mod_q[pd.to_datetime(date).date()] = df[["Sheet", "Mesh Size", "pct_cpft_interpolated", "Particle Size"]].rename(
            columns={
                "Sheet": "Sheet",
                "Mesh Size": "Mesh Size",
                "pct_cpft_interpolated": "pct_CPFT",
                "Particle Size": "Particle Size (μm)"
            }
        )

    for date, df in mod_q.items():
        for density in packing_densities:
            df[f"pct_{int(density * 100)}_poros_CPFT"] = df["pct_CPFT"] * density

    optimized_q_values_df = predict_q_values(mod_q)
    
  # Convert Date columns to datetime objects without timestamps
    gbd_df['Date'] = pd.to_datetime(gbd_df['Date']).dt.date
    q_values_df['Date'] = pd.to_datetime(q_values_df['Date']).dt.date
    optimized_q_values_df['Date'] = pd.to_datetime(optimized_q_values_df['Date']).dt.date

    # Step 7: Combine all results into a single DataFrame using pd.concat
    combined_df = pd.concat([gbd_df.set_index('Date'), 
                         q_values_df.set_index('Date'), 
                         optimized_q_values_df.set_index('Date')], 
                        axis=1).reset_index()
    
    return combined_df
