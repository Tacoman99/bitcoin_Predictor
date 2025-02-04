# Bitcoin Price Predictor

This project is a Bitcoin price prediction tool that uses machine learning techniques to forecast future Bitcoin prices based on historical data.

## Features

- Data collection from Kraken APIs
- Data preprocessing and feature engineering
- Implementation of machine learning models for price prediction
- Visualization of predictions and historical data
- Performance evaluation of the prediction model


3 pipelines 
- Feature Pipeline
    Takes incoming data from Kraken API, transforms it to OHLC candles and stores it to hopsworks featurestore
- Training Pipeline
    Implements model training loading data from featurestore, creating a baseline model, regression and xgboost model
    Saves model in CometML model store
- Inference Pipeline
    Loads model
Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Disclaimer

This project is for educational purposes only. Cryptocurrency investments are volatile and high-risk. Always do your own research before making any investment decisions.
=======



