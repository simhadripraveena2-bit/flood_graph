import pandas as pd
import numpy as np

def load_and_process(path):
    """
    Reads Excel file with two sheets:
      Sheet1: rainfall grid
        Row 1 = Latitude values
        Row 2 = Longitude values
        Subsequent rows = year, month, day, followed by intensity grid
      Sheet2: inflow data (columns: date, inflow)
    Returns:
      df_rain (DataFrame): columns [year, month, day, date, latitude, longitude, intensity]
      df_inflow (DataFrame): columns [date, inflow]
    """
    # --- Rainfall sheet ---
    raw = pd.read_excel(path, sheet_name=0, header=None)

    lat_row = raw.iloc[0].dropna().values
    lon_row = raw.iloc[1].dropna().values
    data = raw.iloc[2:].reset_index(drop=True)

    year = data.iloc[:, 0]
    month = data.iloc[:, 1]
    day = data.iloc[:, 2]
    intensity_data = data.iloc[:, 3:].values

    records = []
    for i in range(len(data)):
        for j in range(len(lat_row)):
            records.append([
                year.iloc[i], month.iloc[i], day.iloc[i],
                lat_row[j], lon_row[j],
                intensity_data[i][j] if j < len(intensity_data[i]) else np.nan
            ])

    df_rain = pd.DataFrame(records, columns=["year", "month", "day", "latitude", "longitude", "intensity"])
    df_rain["date"] = pd.to_datetime(df_rain[["year", "month", "day"]])
    df_rain.dropna(subset=["intensity"], inplace=True)

    # --- Inflow sheet ---
    inflow = pd.read_excel(path, sheet_name=1)
    inflow.columns = inflow.columns.str.lower()
    print(inflow.columns)
    if 'date' not in inflow.columns:
        inflow.rename(columns={inflow.columns[0]: 'date'}, inplace=True)
    if 'inflow' not in inflow.columns:
        inflow.rename(columns={inflow.columns[1]: 'inflow'}, inplace=True)
    inflow['date'] = pd.to_datetime(inflow['date'])

    df_inflow = inflow.dropna(subset=['inflow']).reset_index(drop=True)
    return df_rain, df_inflow
