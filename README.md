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


## License
This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for more details.

## DOI information of this repository  
v1.1: [![DOI](https://zenodo.org/badge/949922361.svg)](https://doi.org/10.5281/zenodo.15100644)   
v1.0: [![DOI](https://zenodo.org/badge/949922361.svg)](https://doi.org/10.5281/zenodo.15043956)    


 


