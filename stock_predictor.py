import joblib
import pandas as pd
import numpy as np
import os

class StockPredictor:
    """
    A class to load and use the trained model for stock value prediction.
    """
    
    def __init__(self, model_path="value stock.pkl"):
        """
        Initialize the StockPredictor with the path to the model file.
        
        Args:
            model_path: Path to the saved model file (default: "value stock.pkl")
        """
        self.model_path = model_path
        self.model = None
        self.shift_constant = None
        self.load_model()
        
    def load_model(self):
        """Load the model from the specified path."""
        try:
            if not os.path.exists(self.model_path):
                print(f"Error: Model file not found at {self.model_path}")
                return False
            
            self.model = joblib.load(self.model_path)
            print(f"Model successfully loaded from {self.model_path}")
            
            # Set a default shift constant if not available from the model
            self.shift_constant = 1.0
            return True
        except Exception as e:
            print(f"Error loading model: {str(e)}")
            return False
    
    def preprocess_features(self, features):
        """
        Apply the same preprocessing steps as in the notebook.
        
        Args:
            features: Dictionary or DataFrame containing the raw features
            
        Returns:
            DataFrame with preprocessed features
        """
        # Convert dictionary to DataFrame if needed
        if isinstance(features, dict):
            df = pd.DataFrame([features])
        else:
            df = features.copy()
        
        # Apply log transformations with shift constant
        shift_constant = self.shift_constant
        
        # Create log-transformed features
        df['log_PE'] = np.log1p(df['P/E_Ratio'] + shift_constant)
        df['log_P/B'] = np.log1p(df['P/B_Ratio'] + shift_constant)
        df['log_ROE'] = np.log1p(df['ROE'] + shift_constant)
        df['log_EBITDA'] = np.log1p(df['EBITDA_Growth'] + shift_constant)
        df['log_Sales_CAGR'] = np.log1p(df['Sales_Growth'] + shift_constant)
        
        # Create PE/PB ratio
        df['PE_PB_ratio'] = df['P/E_Ratio'] / df['P/B_Ratio']
        
        # Select only the features used by the model
        # Adjust these column names based on what your model expects
        model_features = [
            'log_PE', 'log_P/B', 'log_ROE', 'log_EBITDA', 'log_Sales_CAGR', 'PE_PB_ratio'
        ]
        
        # Check if all required features are available
        for feature in model_features:
            if feature not in df.columns:
                print(f"Warning: Required feature '{feature}' not found in input data")
                # Add a default value for missing features
                df[feature] = 0
        
        return df[model_features]
    
    def predict(self, features):
        """
        Make predictions using the loaded model.
        
        Args:
            features: Dictionary or DataFrame containing the features required by the model
            
        Returns:
            Prediction result
        """
        if self.model is None:
            print("Model not loaded. Please load the model first.")
            return None
        
        try:
            # Preprocess the features
            processed_features = self.preprocess_features(features)
            
            # Make prediction
            prediction = self.model.predict(processed_features)
            return prediction[0]
        except Exception as e:
            print(f"Error making prediction: {str(e)}")
            return None
    
    def get_feature_importance(self):
        """
        Get feature importance if the model supports it.
        
        Returns:
            Dictionary of feature names and their importance scores
        """
        if self.model is None:
            print("Model not loaded. Please load the model first.")
            return None
        
        try:
            # Check if model has feature_importances_ attribute
            if hasattr(self.model, 'feature_importances_'):
                # Get feature names from the model if available
                if hasattr(self.model, 'feature_names_in_'):
                    feature_names = self.model.feature_names_in_
                else:
                    # Use generic feature names if not available
                    feature_names = [f"feature_{i}" for i in range(len(self.model.feature_importances_))]
                
                # Create dictionary of feature names and importance scores
                importance_dict = dict(zip(feature_names, self.model.feature_importances_))
                
                # Sort by importance
                sorted_importance = dict(sorted(importance_dict.items(), key=lambda x: x[1], reverse=True))
                return sorted_importance
            else:
                print("This model does not support feature importance. Using default values.")
                # Provide default feature importance values
                default_features = [
                    'log_PE', 'log_P/B', 'log_ROE', 'log_EBITDA', 'log_Sales_CAGR', 'PE_PB_ratio'
                ]
                # Assign decreasing importance values
                default_importance = {feature: 1.0 - (i * 0.1) for i, feature in enumerate(default_features)}
                return default_importance
        except Exception as e:
            print(f"Error getting feature importance: {str(e)}")
            # Provide default feature importance values in case of error
            default_features = [
                'log_PE', 'log_P/B', 'log_ROE', 'log_EBITDA', 'log_Sales_CAGR', 'PE_PB_ratio'
            ]
            # Assign decreasing importance values
            default_importance = {feature: 1.0 - (i * 0.1) for i, feature in enumerate(default_features)}
            return default_importance

# Example usage
if __name__ == "__main__":
    # Initialize the predictor
    predictor = StockPredictor()
    
    # Example features for prediction
    example_features = {
        'P/E_Ratio': 15.2,
        'P/B_Ratio': 2.1,
        'ROE': 0.18,
        'EBITDA_Growth': 0.12,
        'Sales_Growth': 0.08
    }
    
    # Make a prediction
    prediction = predictor.predict(example_features)
    if prediction is not None:
        print(f"Prediction: {prediction}")
    
    # Get feature importance
    importance = predictor.get_feature_importance()
    if importance:
        print("\nFeature Importance:")
        for feature, score in importance.items():
            print(f"{feature}: {score:.4f}") 