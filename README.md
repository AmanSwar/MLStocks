# Stock Predictor API

This project is a stock prediction application that uses several machine learning and deep learning models to forecast stock prices. It provides a user-friendly web interface built with Gradio and is powered by a FastAPI backend.

## Features

*   **Multiple Models:** Compare predictions from various state-of-the-art models:
    *   Hawk
    *   Mamba
    *   xLSTM
    *   Random Forest
*   **Web Interface:** An intuitive Gradio interface to input data and visualize predictions.
*   **Data Processing:** A comprehensive data preparation pipeline that cleans data and engineers a wide range of technical indicator features.
*   **API Backend:** Built with FastAPI, allowing for robust and scalable deployment.

## Project Structure

```
/
├── app.py                  # Main application file with Gradio UI and FastAPI server
├── data/                   # Raw data
├── data_prep/              # Scripts for data preparation and feature engineering
├── deployment/             # Deployment configurations and trained model artifacts
├── model/                  # Source code for the model architectures
├── train/                  # Scripts for training the models
├── requirements.txt        # Python dependencies
└── readme.md               # This README file
```

## Installation

1.  Clone the repository:
    ```bash
    git clone <repository-url>
    cd <repository-name>
    ```

2.  Install the required dependencies:
    ```bash
    pip install -r requirements.txt
    ```

## Usage

To run the application, execute the following command in your terminal:

```bash
python app.py
```

This will start the Gradio web interface, which you can access in your browser at the URL provided in the terminal (usually `http://127.0.0.1:7860`).

## Models

The project implements and compares the following models for stock price prediction:

*   **Hawk:** A novel architecture for time-series forecasting.
*   **Mamba:** A state-space model for sequence modeling.
*   **xLSTM:** An extension of the traditional LSTM with improved memory capabilities.
*   **Random Forest:** A classical ensemble learning method for regression.

The trained model files and their configurations are located in the `deployment/models/` directory.
