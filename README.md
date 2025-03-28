## Overview
XMTC is an Explainable AI (XAI) framework for early classification of multivariate time series, specifically applied to reach-to-grasp hand kinematics. This repository contains the full source code.

## Installation
To set up this project locally, follow these steps:

#### Prerequisites
Ensure you have the following installed:
- Python (>=3.8)
- Git
- Virtual environment support (optional, but recommended)
#### Setup Steps
1. Clone this repository:
    ```bash
   git clone https://github.com/Reyhaneh-Sabbagh/XMTC.git
3. Navigate to the project directory:
   ```
   cd XMTC
4. (Optional) Set up a virtual environment (recommended):
   ```
   python -m venv venv
   source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
5. Install the required libraries:
   ```
   pip install -r requirements.txt

## Running the Application

After installation, you can run the application locally using the following command:

`python app.py`

This will start a local web server. Open your browser and go to http://127.0.0.1:8050/ to view the application.

## Train Machine Learning model
If you want to train the ML model from scratch, you can follow the steps:
  
**1. Preprocessing:**  
1.1. Download the Dataset which is publicly available at https://github.com/Reyhaneh-Sabbagh/Grasp-Dataset .  
1.2. Go to `ML_model`directory and run `preprocessing_data_R2G_features.py` to preprocess data. The preprocessed data will be saved in `Data` naming: 'R2G_features_t_normalized_Task1.csv', 'R2G_features_t_normalized_Task2.csv' and 'R2G_features_t_normalized_Task3.csv'. 
      
**2. Train the ML model:**  
2.1. To train the model you should go to `ML_model` directory and run `main_slidingWindow.py`.  
  
**3. Prepare data for XMTC dash tool:**  
3.1. Go to `data_and_preprocessing` directory which includes all precalculations and preprocessing for XMTC dash tool.
3.2. To calculate accuracy plot run `accuracy_drcif.py` in python.  
3.3. To calculate Histogram for both the whole data and test data run `calculate_Histogram.py` in python.
3.4. To prepare the probabilities of each test data for each class label, you should run `calculate_class_probs_test_data_drcif.py` and `preprocessing_probs_test_data_drcif.py` respectively.  
3.5. Finally you should run `calculate_PartialDependencePlot.py` to compute the partial dependence plots values.  
3.6. The results will be saved in either `Task1`, `Task2` or `Task3`.

## Methodology
XMTC employs an ensemble-based approach for early classification of multivariate time series in reach-to-grasp hand kinematics. In this project we developed an incremental sliding window that classifies the multivariate time series over time. Our main contribution can be summarized as follows:
- The design of the XMTC tool for interactive visualization, exploration, and evaluation of classification prediction models for non-synchronized multivariate time series.
- Adoption of an interval-based classifier model to gradually predict the classification results with increasing length of the time series.
- Interactive visual estimation of a trade-off point between early prediction and classification accuracy.
- Interactive visual exploration of the temporal evolution of the classifier models with increasing length.
- Detailed investigation of the classification prediction evolution of individual multivariate time series.
- Detailed analysis of the impact of input features on the individual model’s classification output.
- Application of the methods to R2G data for detecting user intentions, which offers valuable insights for developing interfaces that use everyday objects as haptic proxies and open the door to utilizing predictive algorithms to account for latency and anticipate future actions.

## Algorithm
For Multivariate time series classification we used Diverse Representation Canonical Interval Forest (DrCIF) which is a state-of-the-art ensemble classifier specifically designed for time series data. It builds decision trees by considering intervals of the time series, rather than individual data points. It works by partitioning the data into intervals and learning decision trees based on the distribution of these intervals across the time series, which enhances its ability to capture temporal dependencies.  


## Dependencies
The main dependencies include:
- Dash
- NumPy
- Pandas
- Scikit-learn
- sktime  

See `requirements.txt` for the full list.

## Dataset
Grasp-Dataset is publicly available at https://github.com/Reyhaneh-Sabbagh/Grasp-Dataset.   
DOI information of the dataset: [![DOI](https://zenodo.org/badge/955882786.svg)](https://doi.org/10.5281/zenodo.15096149)

## Citation information  
This project, 'Early and Explainable Prediction of Reach-to-Grasp Hand Kinematics using Multivariate Time Series Classification', is submitted to 'The Visual Computer' journal.

## License
This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for more details.

## DOI information of this repository  
v1.0: [![DOI](https://zenodo.org/badge/949922361.svg)](https://doi.org/10.5281/zenodo.15043955)  


