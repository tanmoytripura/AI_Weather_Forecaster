# Importing the libraries
import streamlit as st
import numpy as np
import pickle
from tensorflow.keras.models import load_model
import requests
import pandas as pd
from datetime import datetime
import matplotlib.pyplot as plt

# Load models and scalers
model_forecaster = load_model("model_4.h5", compile=False)
model_forecaster.compile(optimizer="adam", loss="mse")
with open("scaler3.pkl", "rb") as f:
    scaler_forecaster = pickle.load(f)

model_prediction = load_model("model_3.h5", compile=False)
model_prediction.compile(optimizer="adam", loss="mse")
with open("scaler2.pkl", "rb") as f:
    scaler_prediction = pickle.load(f)

with open("last_data.pkl", "rb") as f:
    last_data = pickle.load(f)

# print(last_data)


# API call url
url = "https://api.weatherapi.com/v1/current.json?key=e8ed85de788b416fad5133428250203&q=agartala"

response = requests.get(url)
data = response.json()


# API call function
def get_weather():
    url = "https://api.weatherapi.com/v1/current.json?key=e8ed85de788b416fad5133428250203&q=agartala"
    response = requests.get(url)
    data = response.json()
    return data["current"]["temp_c"]

def get_other():
    url = "https://api.weatherapi.com/v1/current.json?key=e8ed85de788b416fad5133428250203&q=agartala"
    response = requests.get(url)
    data = response.json()
    return data["current"]["temp_f"], data["current"]["wind_kph"]

# Set Streamlit UI
st.set_page_config(page_title="AI Weather Predictor", layout="wide")
st.title("🌦️ AI-Powered Weather Forecasting")

if response.status_code == 200:
    # st.text("Live Weather Data "+str(data["current"]["temp_c"]))
    print(data["current"]["temp_c"])

# Scroll Down Button
st.markdown(
    """
    <style>
        .scroll-down {
            position: fixed;
            top: 20px;
            right: 20px;
            background-color: #FF4B4B;
            color: white;
            padding: 10px 15px;
            border-radius: 10px;
            font-weight: bold;
            cursor: pointer;
            z-index: 1000;
        }
        .scroll-down:hover {
            background-color: #E63946;
        }
    </style>
    <script>
        function scrollToBottom() {
            window.scrollTo({ top: document.body.scrollHeight, behavior: 'smooth' });
        }
    </script>
    <button class="scroll-down" onclick="scrollToBottom()">🔽 Scroll Down</button>
    """,
    unsafe_allow_html=True
)

# Sidebar Menu
st.sidebar.header("🔍 Choose an Option")
menu = st.sidebar.radio("", ["Tripura AI Weather Forecaster", "Weather Prediction"])

# Common feature names
conditions_mapping = {
    'Clear': 0, 'Partially cloudy': 1, 'Overcast': 2,
    'Rain, Overcast': 3, 'Rain, Partially cloudy': 4, 'Rain': 5
}
reverse_conditions_mapping = {v: k for k, v in conditions_mapping.items()}

# Model & Scaler Selection
if menu == "Tripura AI Weather Forecaster":
    st.subheader("🚀 Tripura AI Weather Forecaster")
    model = model_forecaster
    scaler = scaler_forecaster
    feature_names = []

    default_values = {
        "temp": 20.0, "humidity": 65.0, "preciptype": "None", "windspeed": 10.0,
        "sealevelpressure": 1013.0, "cloudcover": 50.0, "visibility": 10.0, "conditions": "Clear"
    }

elif menu == "Weather Prediction":
    st.subheader("🌍 Weather Prediction Using LSTM")
    model = model_prediction
    scaler = scaler_prediction
    feature_names = [
        "Precip Type", "Temperature (C)", "Apparent Temperature (C)", "Humidity",
        "Wind Speed (km/h)", "Wind Bearing (degrees)", "Visibility (km)", "Pressure (millibars)"
    ]

    default_values = {
        "Precip Type": "Rain", "Temperature (C)": 20.0, "Apparent Temperature (C)": 18.0,
        "Humidity": 65.0, "Wind Speed (km/h)": 10.0, "Wind Bearing (degrees)": 180.0,
        "Visibility (km)": 10.0, "Pressure (millibars)": 1013.0
    }

# Sidebar Instructions
with st.sidebar.expander("📌 Instructions (Click to Expand)"):
    st.write("""
    1. Enter **24-hour historical data**.
    2. Click **Predict Weather**.
    3. Model forecasts the next **N hours**.
    """)

# Define input sequence length
n_steps_in = 24

# Collect Input Data
features = []
for i in range(n_steps_in):
    if feature_names != []:
        st.subheader(f"⏳ Hour {i+1}")
        row = []


        # Other Weather Features
        for name in feature_names:
            if name == "conditions":
                condition = st.selectbox(
                    f"Conditions (Hour {i+1})",
                    list(conditions_mapping.keys()),
                    index=list(conditions_mapping.keys()).index(default_values["conditions"]),
                    key=f"Conditions_{i+1}"  # 🔥 Unique key for each hour
                )

                row.append(conditions_mapping[condition])
            
            elif name == "preciptype":
                precip_type = st.selectbox(
                    f"Precip Type (Hour {i+1})",
                    ["None", "Rain"],
                    index=0 if default_values["preciptype"] == "None" else 1,
                    key=f"Precip_{i+1}"  # 🔥 Ensuring unique key for each hour
                )
                row.append(0 if precip_type == "None" else 1)
            
            elif name == "Precip Type":
                precip_type = st.selectbox(
                    f"Precip Type (Hour {i+1})",
                    ["Rain", "Snow"],
                    index=0 if default_values["Precip Type"] == "Rain" else 1,
                    key=f"Precip_{i+1}"
                )
                precip_type_encoded = 0 if precip_type == "Rain" else 1
                row.append(precip_type_encoded)
            
            else:
                if isinstance(default_values[name], (int, float)):  # Ensure only numbers go into number_input
                    value = float(default_values[name])
                else:
                    value = 0.0  # Fallback for categorical values

                value = st.number_input(
                    f"{name} (Hour {i+1})",
                    min_value=-100.0, max_value=10000.0,
                    value=value,
                    step=0.1,
                    key=f"{name}_{i+1}"
                )
                row.append(value)

        features.append(row)

# Convert input to NumPy array
if feature_names != []:
    features = np.array(features)

    # Ensure Correct Shape (24, 8)
    if features.shape != (n_steps_in, len(feature_names)):
        st.error(f"❌ Feature shape mismatch! Expected {(n_steps_in, len(feature_names))}, but got {features.shape}")
    else:
        features = features.reshape(1, n_steps_in, len(feature_names))  # (1, 24, 8)


# print(features)
# print(len(features))
    features = features.reshape(1, n_steps_in, len(feature_names))

# Predict Button
    if st.button("🔮 Predict Weather"):
        try:
            # Scale input
            input_scaled = scaler.transform(features.reshape(-1, len(feature_names))).reshape(1, n_steps_in, len(feature_names))

            # LSTM Prediction
            prediction = model.predict(input_scaled)

            # Inverse transform output
            prediction = scaler.inverse_transform(prediction.reshape(-1, len(feature_names)))

            st.success("✅ Prediction Successful!")


            # Convert Conditions back to text
            condition_predictions = [reverse_conditions_mapping.get(round(val), "Unknown") for val in prediction[:, -1]]

            # Display predictions
            st.subheader("📊 Predicted Weather for the Next Hours")
            prediction_df = {}
            print(feature_names)
            print(prediction)
            for i in range(len(feature_names) - 1):
                print(i)
                print(feature_names[i])
                prediction_df[feature_names[i]] = prediction[:, i]
            prediction_df["Conditions"] = condition_predictions

            if "temp" in feature_names:
                prediction_df['temp'] = (prediction_df['temp'] - 32) * 5.0 / 9.0
            
            if "Precip Type" in feature_names:
                print(data["current"]["temp_c"])
                prediction_df["Precip Type"] =  ["Rain" if abs(val) < abs(val - 1) else "Snow" for val in prediction_df["Precip Type"]]

            print('dsd',feature_names)
            if "preciptype" in feature_names:
                print("here")
                prediction_df["preciptype"] = ["None" if abs(val) < abs(val - 1) else "Rain" for val in prediction_df["preciptype"]]
            

            st.dataframe(prediction_df)

        except Exception as e:
            st.error(f"⚠️ Error: {str(e)}")
else:
    if "weather_report" not in st.session_state:
        st.session_state.weather_report = get_weather()

    # Display weather data
    st.markdown(f"### 🌍 Live Weather Data\n")
    st.markdown(
        f"""
        <div style="text-align: center; font-size: 40px; font-weight: bold; color: #FF4B4B;">
            🌡️ {st.session_state.weather_report}°C
        </div>
        """,
        unsafe_allow_html=True
    )

    # Refresh button
    if st.button("🔄 Refresh Weather"):
        st.session_state.weather_report = get_weather()
        st.rerun()

    
    predicted = scaler_forecaster.inverse_transform(model_forecaster.predict(last_data)[0])
    predicted = predicted[-4:]

    temp_f, wind_spd = get_other()
    print(last_data[0][0])
    cur_data_to_add = [temp_f,last_data[0][-1][1],0,last_data[0][-1][3],last_data[0][-1][4],0,last_data[0][-1][6],0]
    cur_data_to_add = np.array(cur_data_to_add)

    

    print(cur_data_to_add)

    print(last_data.shape)

    df_pred = {}
    df_pred_column = ['temp', 'humidity', 'preciptype', 'windspeed', 'sealevelpressure',
       'cloudcover', 'visibility', 'conditions']

    for i in range(len(df_pred_column)):
        if i==0:
            df_pred[df_pred_column[i]] = [((val - 32)*10)/9 for val in predicted[:,i]]
        elif i==2:
            df_pred[df_pred_column[i]] = ["None" if abs(val) < abs(val - 1) else "Rain" for val in predicted[:,i]]
        elif i==7:

            def closest_number(num):
                choices = [0, 1, 2, 3, 4, 5]
                return min(choices, key=lambda x: abs(x - num))
            
            condition_chk = ['Clear','Partially cloudy','Overcast','Rain, Overcast','Rain, Partially cloudy','Rain']
            
            df_pred[df_pred_column[i]] = [condition_chk[closest_number(val)] for val in predicted[:,i]]
        else:
            df_pred[df_pred_column[i]] = predicted[:,i]

    st.dataframe(df_pred)

    hours = pd.date_range(start=datetime.now(), periods=4, freq="D")
    temperature = df_pred['temp']

    fig, ax = plt.subplots(figsize=(10, 5))  # Enlarged graph
    ax.plot(hours, temperature, marker='o', linestyle='-', color='b', label="Temperature (°C)")

    # Set labels and title
    ax.set_xlabel("Hour of the Day")
    ax.set_ylabel("Temperature (°C)")
    ax.set_title("Temperature Trend Over 24 Hours")

    # Set y-axis range from 15°C to 40°C
    ax.set_ylim(15, 40)

    # Add grid for better readability
    ax.grid(True, linestyle="--", alpha=0.6)

    # Show legend
    ax.legend()

    # Display the plot in Streamlit
    st.pyplot(fig)

    last_data = np.append(last_data[:, 1:, :], cur_data_to_add.reshape(1, 1, 8), axis=1)
    print(last_data)
    
    with open("last_data.pkl", "wb") as f:
        pickle.dump(last_data, f)

st.write("🚀 Made with ❤️ using LSTM & Streamlit")
