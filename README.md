# Car Price Prediction using KNN

Overview
This project aims to build a Car Price Prediction Model using the K-Nearest Neighbors (KNN) algorithm. The dataset contains various car attributes like brand, model, year, mileage, fuel type, transmission type, and price. The goal is to predict the price of a car based on these features.

Dataset
We are using a dataset that includes:
- Car Brand & Model (Categorical)
- Year of Manufacture (Numerical)
- Fuel Type (Categorical: Petrol/Diesel/Electric/Hybrid)
- Transmission (Categorical: Manual/Automatic)
- Kilometers Driven (Numerical) 
- Owner Type (First, Second, Third, etc.)
- Car Condition (Categorical: Excellent/Good/Average/Poor)
- Location (City-based pricing variation)
- Price (Target Variable)

Project Steps

1. Exploratory Data Analysis (EDA)
- Visualizing distributions of features
- Checking missing values
- Understanding relationships between features

2. Data Cleaning
- Handling missing values
- Removing outliers
- Encoding categorical variables

3. Feature Engineering
- Creating new features from existing ones
- Transforming data for better model performance

4. Correlational Analysis
- Checking feature correlation with the target variable
- Removing highly correlated independent variables

5. Normalization
- Scaling numerical features for better KNN performance
- Using Min-Max Scaling or Standardization

6. Performance Optimization
- Finding the best value of k for KNN
- Hyperparameter tuning using GridSearchCV
- Evaluating RMSE, R² Score, and Mean Absolute Error

7. PCA Utilization
- Reducing dimensionality for better performance
- Visualizing principal components

Setup Instructions
1. Clone this repository:
   ```bash
   git clone https://github.com/Divyansh-13/KNN_car_Price_prediction.git
   ```
2. Run the Google Colab Notebook to preprocess data and train the model.

Colab Notebook
[Open the Colab Notebook](https://colab.research.google.com/drive/158LqQivPjy9FTAX-fBaFkfIzN2GDNP7l?usp=sharing)

Future Enhancements
- Deploying the trained model on a Next.js Web App
- Using Flask/FastAPI to serve predictions
- Experimenting with deep learning models for better accuracy

Contributor
- Divyansh Sharma- Developer & Data Analyst
