# weather_dashboard_final.py
import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier

# ------------------ Load Dataset ------------------
data = pd.read_csv('weather_large.csv')

st.title("🌦 Weather Prediction Dashboard")
st.write("Interactive prediction and data visualization of weather data.")

# ------------------ Preprocessing ------------------
# Fill missing numeric values with mean
for col in ['Temperature','Humidity','WindSpeed']:
    data[col].fillna(data[col].mean(), inplace=True)

# Fill missing categorical values with mode
data['Rain'].fillna(data['Rain'].mode()[0], inplace=True)

# Optional: remove extreme outliers
data = data[(data['Temperature'] > 0) & (data['Temperature'] < 50)]
data = data[(data['Humidity'] >= 0) & (data['Humidity'] <= 100)]
data = data[data['WindSpeed'] >= 0]

# ------------------ Feature Engineering ------------------
data['Rain'] = data['Rain'].map({'Yes':'Rainy','No':'Sunny'})
data['WeatherType'] = data.apply(lambda x: 'Windy' if x['WindSpeed']>15 else x['Rain'], axis=1)

# ------------------ Sidebar Inputs ------------------
st.sidebar.header("Predict Weather")
temp = st.sidebar.slider("Temperature (°C)", int(data['Temperature'].min()), int(data['Temperature'].max()), 30)
humidity = st.sidebar.slider("Humidity (%)", int(data['Humidity'].min()), int(data['Humidity'].max()), 60)
windspeed = st.sidebar.slider("Wind Speed (km/h)", int(data['WindSpeed'].min()), int(data['WindSpeed'].max()), 10)

# ------------------ Train Model ------------------
X = data[['Temperature','Humidity','WindSpeed']]
y = data['WeatherType']
model = DecisionTreeClassifier(max_depth=6, random_state=42)
model.fit(X, y)

# ------------------ Prediction ------------------
prediction = model.predict([[temp, humidity, windspeed]])
st.subheader("🌈 Predicted Weather:")
st.write(f"### {prediction[0]}")

# ------------------ Data Visualizations ------------------
st.subheader("📊 Data Visualizations")

# ---------- 1️⃣ Countplot ----------
st.write("### Countplot: Weather Type Distribution")
plt.figure(figsize=(6,4))
sns.countplot(x='WeatherType', data=data, palette='Set1', order=['Sunny','Rainy','Windy'])
plt.xlabel("Weather Type")
plt.ylabel("Number of Days")
plt.title("Weather Type Distribution")
# Highlight user prediction on top of the bar
pred_count = data['WeatherType'].value_counts()
plt.bar(prediction[0], pred_count[prediction[0]] + 50, color='red', alpha=0.5)
st.pyplot(plt)

# ---------- 2️⃣ Combined Histogram ----------
st.write("### Combined Histogram: Features Distribution with Your Input Highlighted")
plt.figure(figsize=(10,6))
features = ['Temperature','Humidity','WindSpeed']
colors = ['darkorange','dodgerblue','darkgreen']

for col, color in zip(features, colors):
    sns.histplot(data[col], bins=20, kde=True, color=color, label=col, alpha=0.6)
    # Highlight user input
    plt.axvline(x={'Temperature':temp,'Humidity':humidity,'WindSpeed':windspeed}[col],
                color=color, linewidth=2, linestyle='--')

plt.title("Combined Feature Distributions")
plt.xlabel("Value")
plt.ylabel("Number of Days")
plt.legend()
st.pyplot(plt)
