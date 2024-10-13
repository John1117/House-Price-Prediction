# House Price Prediction

This project is about the [House Price Prediction Contest](https://tbrain.trendmicro.com.tw/Competitions/Details/30) hosted by Sinopac Holdings and Trend Micro in 2023. They employed a dataset that includes number of rooms, location, size, and other factors that influence house pricing in Taiwan. We then aimed to predict house-prices-per-area (or called unit-price) based on various features using machine learning techniques. In this project, I mainly dedicated myself to three parts:
1. Developed automative geographic coordinate system transformation function.
2. Inspected data distirbution to detemine the type of categrical data encoding.
3. Applied linear regression and deep neural network with different kinds of loss function.

## Table of Contents
- [Introduction](#house-price-prediction)
- [Project Structure](#project-structure)
- [Getting Start](#getting-start)
    - [Prerequisites](#prerequisites)
    - [Installation](#installation)
    - [Usage](#usage)

## Project Structure
- `coordinate.py`: The automative transformation function of geographic coordinate system.
- `data_distribution.py`: Inspected the continuity of numerical data.
- `data_grouped_distribution.py`: Inspected the distribution of categrical data sorted by grouped average of house price.
- `metric.py`: Loss or metric funcitons.
- `training/linear_regression`: Linear regression on data with different kinds of data preprocessing.
- `training/deep_neural_network`: Deep neural network model on data with different kinds of data preprocessing and training data sampling.
- `Wilson`: The models from my teammate, Wilson.

## Getting Start

### Prerequisites

- Python 3.x
- `pandas`
- `numpy`
- `scikit-learn`
- `matplotlib`
- `seaborn`
- `torch`

### Installation
1. Clone the repository:
    ```bash
    git clone https://github.com/John1117/House-Price-Prediction.git
    ```
    
2. Install the dependencies using:
    ```bash
    cd House-Price-Prediction
    pip install -r requirements.txt
    ```
### Usage
If you would like to train regression model, please see the folder `training/` and contact me for the training data.