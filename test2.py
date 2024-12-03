import streamlit as st
from PIL import Image, ImageOps
import numpy as np
import pandas as pd
import pickle
import folium
from folium.plugins import MarkerCluster
from streamlit_folium import folium_static  # Ensure this import is included

# Import the custom classifier scripts
from deforestation_classifier import classify_deforestation
from rainfall_classifier import classify_rainfall

# Load the saved model from the pickle file
with open('flood_batata.pkl', 'rb') as file:
    loaded_model = pickle.load(file)

df = pd.read_csv('flood_risk_dataset_india.csv')

# Dictionary containing flood risk impact for deforestation classes
flood_risk_impact_deforestation = {
    "Forest": 10,
    "Field": 20,
    "Highway_Residential_Industrial": 35,
    "River": 45,
}

# Dictionary containing flood risk impact for rainfall classes
flood_risk_impact_rainfall = {
    "Heavy": 50,
    "Moderate": 20,
    "No_rainfall": 0,
}

def calculate_flood_probability(rainfall, discharge, slope, vegetation, capacity):
    flood_probability = (
        (rainfall / 500) * 0.3 + 
        (discharge / 2000) * 0.25 + 
        (slope / 10) * 0.1 + 
        (vegetation / 10) * 0.15 + 
        (capacity / 5000) * 0.2
    )
    return flood_probability

def calculate_final_probability(flood_probability, flood_forestation, flood_rainfall):
    totalProbability = (
        (flood_probability) * 0.8 + 
        (flood_forestation) * 0.1 + 
        (flood_rainfall) * 0.1
    )
    return totalProbability * 100

# Streamlit Layout Setup
st.markdown("""<style>
    body {
        background-color: #f4f4f4;
        font-family: 'Arial', sans-serif;
    }
    .header {
        background-color: rgba(41, 128, 185, 0.9);
        padding: 30px;
        border-radius: 15px;
        color: white;
        text-align: center;
        margin-bottom: 30px;
    }
    </style>""", unsafe_allow_html=True)

st.markdown("<div class='header'><h2>Flood Prediction App</h2></div>", unsafe_allow_html=True)

# Initialize variables to hold flood probabilities
flood_prob_deforestation = None
flood_prob_rainfall = None

# Section for Deforestation Prediction
with st.expander("Classification of Area Impacting Flood Risk", expanded=True):
    st.write("Upload an image to classify Deforestation impact on Flood Risk")

    uploaded_file_deforestation = st.file_uploader("Input a Deforestation Image", type=["jpg", "jpeg", "png"], key="deforestation")

    if uploaded_file_deforestation is not None:
        # Open and display the deforestation image
        image_deforestation = Image.open(uploaded_file_deforestation).convert("RGB")
        st.image(image_deforestation, caption="Uploaded Deforestation Image", width=300)

        # Use the classifier function from deforestation_classifier.py
        class_name_deforestation = classify_deforestation(image_deforestation)

        # Flood Risk Impact based on Deforestation Prediction
        percentage_deforestation = flood_risk_impact_deforestation.get(class_name_deforestation, "No data available for this class")
        flood_prob_deforestation = percentage_deforestation / 100 if isinstance(percentage_deforestation, int) else 0

        st.write(f"**Class**: {class_name_deforestation}")

# Section for Rainfall Prediction
with st.expander("Rainfall Image Prediction", expanded=True):
    st.write("Upload an image to classify Rainfall impact on Flood Risk")

    uploaded_file_rainfall = st.file_uploader("Input a Rainfall Image", type=["jpg", "jpeg", "png"], key="rainfall")

    if uploaded_file_rainfall is not None:
        # Open and display the rainfall image
        image_rainfall = Image.open(uploaded_file_rainfall).convert("RGB")
        st.image(image_rainfall, caption="Uploaded Rainfall Image", width=300)

        # Use the classifier function from rainfall_classifier.py
        class_name_rainfall = classify_rainfall(image_rainfall)

        # Flood Risk Impact based on Rainfall Prediction
        percentage_rainfall = flood_risk_impact_rainfall.get(class_name_rainfall, "No data available for this class")
        flood_prob_rainfall = percentage_rainfall / 100 if isinstance(percentage_rainfall, int) else 0

        st.write(f"**Class**: {class_name_rainfall}")

# Average of Images
flood_prob_input = flood_prob_deforestation if flood_prob_deforestation is not None else 0
flood_prob_rainfall_combined = flood_prob_rainfall if flood_prob_rainfall is not None else 0

# Parameter-based Flood Prediction
with st.expander("Parameters for the flood", expanded=True):
    latitude = st.number_input("Latitude:(Enter between 8-37) deg", value=18.32)
    longitude = st.number_input("Longitude:(Enter between 68-97) deg", value=73.51)
    rainfall = st.number_input("Rainfall (mm):(Enter between 0.01-300)", value=33.2)
    temperature = st.number_input("Temperature (°C):(Enter between 15-45)", value=28.144337)
    humidity = st.number_input("Humidity (%):(Enter between 20-100)", value=68.912963)
    river_discharge = st.number_input("River Discharge (m³/s): (Enter between 0.04-5000)", value=3000.182888)
    water_level = st.number_input("Water Level (m):(Enter between 0-10)", value=5.415552)
    elevation = st.number_input("Elevation (m):(Enter between 1.15-8850)", value=560)
    land_cover = st.selectbox("Land Cover:(Choose one of the Land types)", ['Water Body', 'Forest', 'Agricultural', 'Urban', 'Bare Soil', 'Wetland', 'Shrubland', 'Grassland'])
    soil_type = st.selectbox("Soil Type:(Choose one of the soil types)", ['Clay', 'Sand', 'Silt', 'Loam'])
    population_density = st.number_input("Population Density:(Enter between 2.29-10000)", value=350.74)
    infrastructure = st.selectbox("Infrastructure (1 for Yes, 0 for No)(if present (Select 1)):", [0, 1])
    historical_floods = st.selectbox("Historical Floods:(if present (Select 1))", [0, 1])

    if st.button("Predict"):
        new_data = pd.DataFrame({
            'Latitude': [latitude],
            'Longitude': [longitude],
            'Rainfall (mm)': [rainfall],
            'Temperature (°C)': [temperature],
            'Humidity (%)': [humidity],
            'River Discharge (m³/s)': [river_discharge],
            'Water Level (m)': [water_level],
            'Elevation (m)': [elevation],
            'Land Cover': [land_cover],
            'Soil Type': [soil_type],
            'Population Density': [population_density],
            'Infrastructure': [infrastructure],
            'Historical Floods': [historical_floods]
        })

        predicted_probabilities = loaded_model.predict(new_data)
        finalProbability = calculate_final_probability(predicted_probabilities[0], flood_prob_input, flood_prob_rainfall_combined)

        st.write(f"Predicted Flood Probability of overall model: {finalProbability:.2f}%")
        if finalProbability > 0.5:
            st.write("There are chances of Flood occurrence according to the given parameters.")
            
            
            
            
# Filter data
precipitation_threshold = 200
high_precipitation_data = df[(df['Rainfall (mm)'] > precipitation_threshold) & (df['Flood Occurred'] == 1)]

# Create the map
if not high_precipitation_data.empty:
    map_center = [high_precipitation_data['Latitude'].mean(), high_precipitation_data['Longitude'].mean()]
    my_map = folium.Map(location=map_center, zoom_start=5)

    marker_cluster = MarkerCluster().add_to(my_map)

    for _, row in high_precipitation_data.iterrows():
        folium.Marker(
            location=[row['Latitude'], row['Longitude']],
            popup=(
                f"Rainfall: {row['Rainfall (mm)']} mm, "
                f"Temp: {row['Temperature (°C)']} °C, "
                f"Humidity: {row['Humidity (%)']}%, "
                f"Water Level: {row['Water Level (m)']} m, "
                f"Elevation: {row['Elevation (m)']} m"
            )
        ).add_to(marker_cluster)

    # Display map in Streamlit
    folium_static(my_map)
else:
    st.write("No flooding data available for rainfall greater than 200 mm.")  


# Map for places where floods did not occur and rainfall > 200 mm
st.write("Map Of All Places Where Floods Didn't Occur and The Precipitation is Higher Than 200 mm")

precipitation_threshold = 200

# Filter data for places where floods didn't occur
high_precipitation_no_flood_data = df[(df['Rainfall (mm)'] > precipitation_threshold) & (df['Flood Occurred'] == 0)]

# Create the map if there is data available
if not high_precipitation_no_flood_data.empty:
    map_center = [high_precipitation_no_flood_data['Latitude'].mean(), high_precipitation_no_flood_data['Longitude'].mean()]
    my_map_no_flood = folium.Map(location=map_center, zoom_start=5)

    marker_cluster_no_flood = MarkerCluster().add_to(my_map_no_flood)

    for _, row in high_precipitation_no_flood_data.iterrows():
        folium.Marker(
            location=[row['Latitude'], row['Longitude']],
            popup=(
                f"Rainfall: {row['Rainfall (mm)']} mm, "
                f"Temp: {row['Temperature (°C)']} °C, "
                f"Humidity: {row['Humidity (%)']}%, "
                f"Water Level: {row['Water Level (m)']} m, "
                f"Elevation: {row['Elevation (m)']} m"
            )
        ).add_to(marker_cluster_no_flood)

    # Display map in Streamlit
    folium_static(my_map_no_flood)
else:
    st.write("No data available for locations with rainfall greater than 200 mm and no floods.")