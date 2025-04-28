# Car Price Prediction using KNN

Overview
This project aims to build a Car Price Prediction Model using the K-Nearest Neighbors (KNN) algorithm. The dataset contains various car attributes like brand, model, year, mileage, fuel type, transmission type, and price. The goal is to predict the price of a car based on these features.

# Car Price Prediction using K-Nearest Neighbors (KNN)

This project aims to predict car prices using the K-Nearest Neighbors (KNN) algorithm. The dataset used contains information about various car features, including make, model, year, horsepower, engine size, mileage, and price.

## Project Structure

- `CarPriceDataset_Final.csv`: The dataset containing car information.
- `car_price_prediction.ipynb`: The Jupyter Notebook containing the code for data cleaning, EDA, model training, and evaluation.
- `knn_car_price_model.pkl`: The trained KNN model saved using `joblib`.
- `readme.md`: This file providing an overview of the project.

## Steps

1. **Data Preparation and Cleaning:**
   - Load the dataset.
   - Clean column names.
   - Handle missing values (e.g., imputing mileage with the median).
   - Remove duplicate rows.
   - Handle outliers (e.g., removing cars with prices above a certain threshold).
   - Encode categorical variables using Label Encoding.

2. **Exploratory Data Analysis (EDA):**
   - Visualize the distribution of car prices.
   - Analyze relationships between price and key features (horsepower, engine size, year, mileage).
   - Explore car prices by company using box plots.
   - Calculate and visualize the correlation matrix between numerical features.

3. **Model Selection and Training:**
   - Split the data into training and testing sets.
   - Scale numerical features using RobustScaler.
   - Train a KNN model with different values of `k` (number of neighbors).
   - Evaluate model performance using metrics like Mean Squared Error (MSE), Root Mean Squared Error (RMSE), Mean Absolute Error (MAE), and R-squared (R²).
   - Select the best `k` value based on the lowest MAE.
   - Train a final KNN model with the optimal `k` value.
   - Consider using a log-transformed target variable for better model performance.

4. **Model Evaluation and Optimization:**
   - Perform hyperparameter tuning using GridSearchCV to find the best combination of hyperparameters for the KNN model.
   - Evaluate the tuned model on the test set.
   - Visualize predictions vs. actual values.
   - Analyze errors by price range to identify potential areas for improvement.

5. **Prediction and Interpretation:**
   - Create a function to predict car prices for new data points.
   - Provide insights into the model's predictions by showing the nearest neighbors used for prediction.

## Dependencies

- pandas
- numpy
- matplotlib
- seaborn
- scikit-learn
- joblib


## Usage

1. Upload the `CarPriceDataset_Final.csv` file and `car_price_prediction.ipynb` to Google Colab.
2. Execute the cells in the notebook to perform the steps described above.
3. The trained model will be saved as `knn_car_price_model.pkl`.
4. To make predictions, call the `predict_car_price()` function with the index of the car you want to predict the price for.

## Results

The project demonstrates the effectiveness of KNN in predicting car prices. By optimizing the model and handling outliers, reasonable prediction accuracy can be achieved.

## Further Improvements

- Explore other regression algorithms (e.g., Linear Regression, Random Forest) for comparison.
- Experiment with feature engineering to create more informative features.
- Fine-tune the model further by adjusting hyperparameters and considering different scaling techniques.
Setup Instructions
1. Clone this repository:
   ```bash
   git clone https://github.com/Divyansh-13/KNN_car_Price_prediction.git
   ```
2. Run the Google Colab Notebook to preprocess data and train the model.

Colab Notebook
[Open the Colab Notebook](https://colab.research.google.com/drive/1-rFkvBvoJqGOiutzwaP8TmA6dEw2DzTl?usp=sharing)

Future Enhancements
- Deploying the trained model on a Next.js Web App
- Using Flask/FastAPI to serve predictions
- Experimenting with deep learning models for better accuracy

Contributor
- Divyansh Sharma- Developer & Data Analyst
