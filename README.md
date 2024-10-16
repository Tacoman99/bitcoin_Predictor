# Bitcoin Price Predictor

This project is a Bitcoin price prediction tool that uses machine learning techniques to forecast future Bitcoin prices based on historical data.

## Features

- Data collection from cryptocurrency APIs
- Data preprocessing and feature engineering
- Implementation of machine learning models for price prediction
- Visualization of predictions and historical data
- Performance evaluation of the prediction model

## Installation

1. Clone the repository:
   ```
   git clone https://github.com/yourusername/Bitcoin_predictor.git
   cd Bitcoin_predictor
   ```

2. Install the required dependencies:
   ```
   pip install -r requirements.txt
   ```

## Usage

1. Run the data collection script:
   ```
   python collect_data.py
   ```

2. Preprocess the data:
   ```
   python preprocess_data.py
   ```

3. Train the model:
   ```
   python train_model.py
   ```

4. Make predictions:
   ```
   python predict.py
   ```

5. Visualize results:
   ```
   python visualize_results.py
   ```

## Project Structure

- `collect_data.py`: Script for collecting Bitcoin price data from APIs
- `preprocess_data.py`: Data cleaning and feature engineering
- `train_model.py`: Implementation and training of the machine learning model
- `predict.py`: Using the trained model to make predictions
- `visualize_results.py`: Plotting and visualizing the results
- `utils.py`: Helper functions used across the project
- `requirements.txt`: List of Python dependencies

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Disclaimer

This project is for educational purposes only. Cryptocurrency investments are volatile and high-risk. Always do your own research before making any investment decisions.
