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

# Remove extreme outliers
data = data[(data['Temperature'] > 0) & (data['Temperature'] < 50)]
data = data[(data['Humidity'] >= 0) & (data['Humidity'] <= 100)]
data = data[data['WindSpeed'] >= 0]

# ------------------ Feature Engineering ------------------
data['Rain'] = data['Rain'].map({'Yes': 'Rainy', 'No': 'Sunny'})
data['WeatherType'] = data.apply(lambda x: 'Windy' if x['WindSpeed'] > 15 else x['Rain'], axis=1)
data['Day'] = range(1, len(data) + 1)

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

# ---------- 1️⃣ Countplot: Weather Type Distribution ----------
st.write("### Countplot: Weather Type Distribution (Number of Days vs Weather Type)")
fig1, ax1 = plt.subplots(figsize=(6, 4))
sns.countplot(x='WeatherType', data=data, palette='Set1', order=['Sunny', 'Rainy', 'Windy'], ax=ax1)
ax1.set_xlabel("Weather Type")
ax1.set_ylabel("Number of Days")
ax1.set_title("Weather Type Distribution (Days vs Weather Type)")

# Highlight predicted bar
pred_count = data['WeatherType'].value_counts()
if prediction[0] in pred_count.index:
    ax1.bar(prediction[0], pred_count[prediction[0]] + 50, color='red', alpha=0.5, label="Predicted")
ax1.legend()
st.pyplot(fig1)

# ---------- 2️⃣ Line Plot: Weather Features Over Days (replacement for histogram) ----------
st.write("### Weather Parameters Over Number of Days (Line Plot)")

fig2, ax2 = plt.subplots(figsize=(10, 6))

# Plot raw values for each day (thin lines)
ax2.plot(data['Day'], data['Temperature'], label='Temperature (°C)', color='orange', alpha=0.35)
ax2.plot(data['Day'], data['Humidity'], label='Humidity (%)', color='blue', alpha=0.35)
ax2.plot(data['Day'], data['WindSpeed'], label='Wind Speed (km/h)', color='green', alpha=0.35)

# Plot smoothed (rolling mean) lines for clarity
window = max(3, int(len(data) * 0.02))  # small window relative to data size
ax2.plot(data['Day'], data['Temperature'].rolling(window).mean(), color='orange', linewidth=2.2, label=f'Temp (MA{window})')
ax2.plot(data['Day'], data['Humidity'].rolling(window).mean(), color='blue', linewidth=2.2, label=f'Humidity (MA{window})')
ax2.plot(data['Day'], data['WindSpeed'].rolling(window).mean(), color='green', linewidth=2.2, label=f'WindSpeed (MA{window})')

# Highlight user inputs with horizontal dashed lines
ax2.axhline(y=temp, color='orange', linestyle='--', linewidth=1.8, label='Your Temp')
ax2.axhline(y=humidity, color='blue', linestyle='--', linewidth=1.8, label='Your Humidity')
ax2.axhline(y=windspeed, color='green', linestyle='--', linewidth=1.8, label='Your Wind Speed')

ax2.set_xlabel("Number of Days")
ax2.set_ylabel("Value")
ax2.set_title("Temperature, Humidity & Wind Speed Over Days (with moving average)")
ax2.legend(loc='upper right', fontsize='small')
ax2.grid(alpha=0.3)

st.pyplot(fig2)
