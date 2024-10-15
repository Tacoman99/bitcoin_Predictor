## BitCoin predictor 

3 pipelines 
- Feature Pipeline
    Takes incoming data from Kraken API, transforms it to OHLC candles and stores it to hopsworks featurestore
- Training Pipeline
    Implements model training loading data from featurestore, creating a baseline model, regression and xgboost model
    Saves model in CometML model store
- Inference Pipeline
    Loads model


Next steps:
- Dockerize the Training/ Inference pipline
- Implement monitoring and incremental learning steps
- Improve feature engineering and model accuaracy 
- Create a REST API for so the model can be call
- Deploy pipeline onto quixcloud
