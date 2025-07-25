import json
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional
import pandas as pd
from datetime import datetime
import calendar
from forecasting_function import create_forecast_for_target ,create_city_forecast_for_target  # Ensure this is correctly imported

app = FastAPI()

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://98.83.3.11:3000"],  # Adjust this in production for security
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load service points data
# SERVICE_POINTS_PATH = '/media/test/New Volume/Nabeel/Capacity_Optimizer/Notebooks/Data_preparation/frontend/project-bolt-sb1-sgtyxpyk/project/api/service_point.json'
SERVICE_POINTS_PATH = "testing_sp_json2.json"#"/media/test/New Volume/Nabeel/Capacity_Optimizer/Notebooks/Data_preparation/augmentation_work/Untitled Folder/testing_sp_json2.json"#'/media/test/New Volume/Nabeel/Capacity_Optimizer/Notebooks/Data_preparation/augmentation_work/Untitled Folder/testing_sp_json.json'
city_path ="testing_regionWise2.json"#'/media/test/New Volume/Nabeel/Capacity_Optimizer/Notebooks/Data_preparation/augmentation_work/Untitled Folder/testing_regionWise2.json'#'/media/test/New Volume/Nabeel/Capacity_Optimizer/Notebooks/Data_preparation/augmentation_work/Untitled Folder/testing_regionWise.json'

# SERVICE_POINTS_PATH = '/media/test/New Volume/Nabeel/Capacity_Optimizer/Notebooks/Data_preparation/augmentation_work/Untitled Folder/testing_sp_json.json'
# city_path = '/media/test/New Volume/Nabeel/Capacity_Optimizer/Notebooks/Data_preparation/augmentation_work/Untitled Folder/testing_regionWise.json'


with open(SERVICE_POINTS_PATH, 'r') as f:
    service_points = json.load(f)


with open(city_path, 'r') as f:
    cityes = json.load(f)

# Load DataFrame
# DF_PATH = "/media/test/New Volume/Nabeel/Capacity_Optimizer/Notebooks/Data_preparation/frontend/skroutzDF_sp_with_clustring.csv"
# DF_PATH = "/media/test/New Volume/Nabeel/Capacity_Optimizer/Notebooks/Data_preparation/frontend/PNO_city_sp_data_cleaned.csv"
# DF_PATH_city_level = "/media/test/New Volume/Nabeel/Capacity_Optimizer/Notebooks/Data_preparation/frontend/PNO_city_network_data.csv"

DF_PATH = "PNO_city_sp_data_cleaned2.csv"#"/media/test/New Volume/Nabeel/Capacity_Optimizer/Notebooks/Data_preparation/frontend/PNO_city_sp_data_cleaned2.csv"#"/media/test/New Volume/Nabeel/Capacity_Optimizer/Notebooks/Data_preparation/frontend/PNO_city_sp_data_cleaned.csv"
DF_PATH_city_level = "PNO_city_network_data2.csv"#"/media/test/New Volume/Nabeel/Capacity_Optimizer/Notebooks/Data_preparation/frontend/PNO_city_network_data2.csv"#"/media/test/New Volume/Nabeel/Capacity_Optimizer/Notebooks/Data_preparation/frontend/PNO_city_network_data.csv"

df = pd.read_csv(DF_PATH)
df_city_level = pd.read_csv(DF_PATH_city_level)
df['ds'] = pd.to_datetime(df['ds'])
df_city_level['ds'] = pd.to_datetime(df_city_level['ds'])
df['unique_id'] = df['unique_id'].astype(int)
sp_ids = df['unique_id'].unique()
citys = df_city_level['city'].unique()
print(f"Found sp_ids: {sp_ids}")
print(f"Found sp_ids: {citys}")


class ForecastRequest(BaseModel):
    sp_id: str
    # city: str
    start_date: str  # Expected format: "YYYY-MM"
    end_date: str    # Expected format: "YYYY-MM"
    target_column: str


def get_last_day_of_month(year_month: str) -> str:
    year, month = map(int, year_month.split('-'))
    _, last_day = calendar.monthrange(year, month)
    return f"{year_month}-{str(last_day).zfill(2)}"


@app.get("/service-points")
async def get_service_points():
    return {"service_points": service_points}  # Wrap in a dict for consistency





@app.get("/city_name")
async def get_city():
    return {"service_points": cityes}  # Wrap in a dict for consistency


@app.post("/forecast")
async def get_forecast(request: ForecastRequest):
    try:
        # Convert sp_id to int and validate
        sp_id = int(request.sp_id)
        if sp_id not in sp_ids:
            raise HTTPException(status_code=404, detail=f"Service point ID {sp_id} not found")

        # Prepare dates
        predict_start_date = f"{request.start_date}-01"
        predict_end_date = get_last_day_of_month(request.end_date)

        # Extract month and year from start_date for training
        year_train, month_train = map(int, request.start_date.split('-'))

        # Filter DataFrame for the specific sp_id
        sp_df = df[df['unique_id'] == sp_id]
        if sp_df.empty:
            raise HTTPException(status_code=404, detail=f"No data found for sp_id {sp_id}")

        # Call forecasting function
        fig, data = create_forecast_for_target(
            df=sp_df,
            target_column=request.target_column,
            sp_id=sp_id,
            Month_train=month_train,
            Year_train=year_train,
            predict_start_date=predict_start_date,
            predict_end_date=predict_end_date,
        )

        # Ensure 'ds' is in the DataFrame and convert to string format
        data['ds'] = data.index
        data['ds'] = data['ds'].dt.strftime('%Y-%m-%d')  # Convert timestamps to string

        # Return forecast data in the expected format
        return {
            "actual": data['actual'].tolist(),
            "predicted": data['predicted'].tolist(),
            "timestamps": data['ds'].tolist()
        }

    except ValueError as ve:
        raise HTTPException(status_code=400, detail=f"Invalid input: {str(ve)}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Forecasting error: {str(e)}")
    



@app.post("/forecast_city")
async def get_city_forecast(request: ForecastRequest):
    try:
        # Convert sp_id to int and validate
        city = str(request.sp_id)
        if city not in citys:
            raise HTTPException(status_code=404, detail=f"Service point ID {city} not found")

        # Prepare dates
        predict_start_date = f"{request.start_date}-01"
        predict_end_date = get_last_day_of_month(request.end_date)

        # Extract month and year from start_date for training
        year_train, month_train = map(int, request.start_date.split('-'))

        # Filter DataFrame for the specific sp_id
        city_df = df_city_level[df_city_level['city'] == city]
        if city_df.empty:
            raise HTTPException(status_code=404, detail=f"No data found for sp_id {city}")

        # Call forecasting function
        fig, data = create_city_forecast_for_target(
            df=city_df,
            target_column=request.target_column,
            sp_id=city,
            Month_train=month_train,
            Year_train=year_train,
            predict_start_date=predict_start_date,
            predict_end_date=predict_end_date,
        )

        # Ensure 'ds' is in the DataFrame and convert to string format
        data['ds'] = data.index
        data['ds'] = data['ds'].dt.strftime('%Y-%m-%d')  # Convert timestamps to string

        # Return forecast data in the expected format
        return {
            "actual": data['actual'].tolist(),
            "predicted": data['predicted'].tolist(),
            "timestamps": data['ds'].tolist()
        }

    except ValueError as ve:
        raise HTTPException(status_code=400, detail=f"Invalid input: {str(ve)}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Forecasting error: {str(e)}")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="3.95.74.209", port=8000)