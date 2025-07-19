import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import plotly.express as px

# Generate synthetic dataset
def generate_house_data(n_samples=100):
    np.random.seed(42)
    size = np.random.normal(1500, 500, n_samples)
    price = size * 100 + np.random.normal(0, 10000, n_samples)
    return pd.DataFrame({'size_sqft': size, 'price': price})

# Train simple linear regression model
def train_model():
    df = generate_house_data()
    X = df[['size_sqft']]
    y = df['price']
    model = LinearRegression()
    model.fit(X, y)
    return model

# Main Streamlit app
def main():
    st.title('🏠 Simple House Pricing Predictor')
    st.write('Enter the house size (in sq. ft.) to predict its price.')

    # Train model
    model = train_model()

    # User input
    size = st.number_input('House size (sq. ft.)', 500, 5000, step=100, value=1500)

    # Predict and display
    if st.button('Predict price'):
        prediction = model.predict([[size]])[0]
        st.success(f"Estimated price: **${prediction:,.2f}**")

        # Visualize prediction
        df = generate_house_data()
        fig = px.scatter(df, x='size_sqft', y='price', title='Size vs Price')
        fig.add_scatter(x=[size], y=[prediction],
                        mode='markers',
                        marker=dict(size=15, color='red'),
                        name='Your Prediction')
        st.plotly_chart(fig)

if __name__ == '__main__':
    main()
