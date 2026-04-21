# Stock Price Prediction using LSTM

## Project Overview

The objective of this project is to build and evaluate a Deep Learning model capable of predicting the next day's closing price of a stock based on its previous 60-day price history. It includes the full pipeline from data ingestion and normalization to model training and performance visualization.

## Dataset

* Source: Historical stock data for Apple (AAPL).

* Features: The model specifically focuses on the Close price column.

* Size: Over 10,000 daily records starting from December 1980.

## Technical Implementation

## 1. Data Preprocessing
* Normalization: Data is scaled between 0 and 1 using MinMaxScaler to improve the convergence of the neural network.

* Windowing: The data is transformed into a supervised learning format where 60 consecutive days of prices serve as input features to predict the price of the 61st day.

* Splitting: The dataset is split into 80% for training and 20% for testing.

## 2. Model Architecture

The project implements a Sequential LSTM model using TensorFlow/Keras: 

* LSTM Layers: Two LSTM layers with 50 units each. The first layer returns sequences to allow the second layer to process the temporal information.

* Regularization: Dropout layers (20%) are included after each LSTM layer to reduce overfitting.

* Output Layers: A Dense layer with 25 units followed by a final Dense layer with a single output (the predicted price).

* Optimizer: `adam`.

* Loss Function: mean_squared_error.

## 3. Evaluation Metrics
The model's performance is evaluated on the test set using standard regression metrics:

* RMSE (Root Mean Squared Error): Measures the average magnitude of the error.

* MAE (Mean Absolute Error): Measures the average absolute difference between predicted and actual values.

* MAPE (Mean Absolute Percentage Error): Represents the accuracy as a percentage.

## Dependencies
`Python 3.x`

`Pandas & NumPy`

`Matplotlib (for visualization)`

`Scikit-learn (for data scaling)`

`TensorFlow/Keras (for building the LSTM)`

## Usage
* Ensure `AAPL (1).csv` is in the project directory.

* Run the notebook cells to load the data, train the model `(default: 5 epochs)`, and generate the prediction plots.
