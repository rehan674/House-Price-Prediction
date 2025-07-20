import streamlit as st
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

st.title("🏠 House Price Predictor")

data = {
    'Area (sq ft)': [1000, 1500, 2000, 2500, 3000],
    'Bedrooms': [2, 3, 3, 4, 4],
    'Price (₹ Lakhs)': [50, 70, 90, 110, 130]
}
df = pd.DataFrame(data)

st.subheader("Sample Dataset")
st.write(df)

X = df[['Area (sq ft)', 'Bedrooms']]
y = df['Price (₹ Lakhs)']
model = LinearRegression()
model.fit(X, y)

st.subheader("📋 Predict Your House Price")
area = st.number_input("Enter Area (sq ft)", value=1200)
bedrooms = st.slider("Select Number of Bedrooms", 1, 5, 3)

# Prediction
if st.button("Predict Price"):
    input_data = pd.DataFrame({'Area (sq ft)': [area], 'Bedrooms': [bedrooms]})
    prediction = model.predict(input_data)[0]
    st.success(f"Estimated Price: ₹ {prediction:.2f} Lakhs")
