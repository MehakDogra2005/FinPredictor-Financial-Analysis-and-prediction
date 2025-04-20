from flask import Flask, render_template, request, jsonify
import os
import traceback
from stock_predictor import StockPredictor

app = Flask(__name__)

# Initialize the stock predictor
model_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "value stock.pkl")
predictor = StockPredictor(model_path)

@app.route('/')
def home():
    """Render the home page."""
    return render_template('homePage.html')

@app.route('/analysis.html')
def analysis():
    """Render the analysis page."""
    return render_template('analysis.html')

@app.route('/api/predict', methods=['POST'])
def predict():
    """API endpoint to make predictions using the model."""
    try:
        # Get data from request
        data = request.json
        
        # Validate required fields
        required_fields = ['P/E_Ratio', 'P/B_Ratio', 'ROE', 'EBITDA_Growth', 'Sales_Growth']
        missing_fields = [field for field in required_fields if field not in data]
        
        if missing_fields:
            return jsonify({
                'success': False,
                'error': f"Missing required fields: {', '.join(missing_fields)}"
            }), 400
        
        # Make prediction
        prediction = predictor.predict(data)
        
        # Determine if it's a value stock
        is_value_stock = bool(prediction)
        
        # Generate a more detailed response
        response = {
            'success': True,
            'prediction': prediction,
            'is_value_stock': is_value_stock,
            'message': 'Based on our analysis, this stock is considered undervalued and may present a good investment opportunity.' if is_value_stock else 'This stock is not considered a value stock based on our analysis.'
        }
        
        return jsonify(response)
    except Exception as e:
        # Log the full error for debugging
        print(f"Error in prediction: {str(e)}")
        traceback.print_exc()
        
        return jsonify({
            'success': False,
            'error': f"An error occurred during prediction: {str(e)}"
        }), 500

if __name__ == '__main__':
    app.run(debug=True) 