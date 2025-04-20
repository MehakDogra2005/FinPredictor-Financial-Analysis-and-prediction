# FinPredictor - Financial Analysis and Prediction

A web application that uses machine learning to identify undervalued stocks based on financial metrics.

## Features

- Analyze stocks using key financial metrics (P/E, P/B, ROE, EBITDA Growth, Sales Growth)
- Predict whether a stock is undervalued using a trained machine learning model
- Visualize feature importance to understand what factors contribute to the prediction
- Modern, responsive UI for easy use on any device

## Installation

1. Clone this repository:
```
git clone <repository-url>
cd financial-analysis-and-prediction-model
```

2. Install the required dependencies:
```
pip install -r requirements.txt
```

3. Make sure your model file (`value stock.pkl`) is in the root directory of the project.

## Usage

1. Start the Flask application:
```
python app.py
```

2. Open your web browser and navigate to:
```
http://127.0.0.1:5000/
```

3. Click on "Analyze a Stock Now" or navigate to the analysis page to input stock metrics and get predictions.

## Model Information

The application uses a machine learning model trained on historical stock data to identify undervalued stocks. The model considers the following metrics:

- P/E Ratio (Price-to-Earnings)
- P/B Ratio (Price-to-Book)
- Return on Equity (ROE)
- EBITDA 1-Year Growth
- Sales 5-Year CAGR

## Project Structure

- `app.py`: Flask application that serves the web interface and handles API requests
- `stock_predictor.py`: Class for loading and using the trained model
- `templates/`: HTML templates for the web interface
  - `homePage.html`: Home page with introduction and navigation
  - `analysis.html`: Page for inputting stock metrics and viewing predictions
- `value stock.pkl`: The trained machine learning model

## License

This project is licensed under the MIT License - see the LICENSE file for details. 