# Housing Price Prediction Using XGBoost, PSO-XGBoost, and CatBoost

## Overview

This project predicts housing prices using three machine learning
approaches inspired by the parent papers used in the DS340W project:

-   **XGBoost** as the baseline model
-   **PSO-XGBoost** for hyperparameter optimization using Particle Swarm
    Optimization
-   **CatBoost** as the final comparison model

The project combines multiple housing datasets, performs preprocessing and feature engineering, and then trains and evaluates the models using regression metrics.

------------------------------------------------------------------------

## What this project does

The workflow is:

1.  Load and merge multiple housing datasets
2.  Clean missing values and inconsistent formats
3.  Engineer features such as:
    -   `price_per_sqft`
    -   `log_price`
    -   `house_age`
    -   `dataset` source label
4.  Train and evaluate:
    -   XGBoost
    -   PSO-XGBoost
    -   CatBoost
5.  Report performance using:
    -   RMSE
    -   MAE
    -   MAPE
    -   RMSLE
    -   R-squared
6.  Interpret the CatBoost model using feature importance and SHAP values

------------------------------------------------------------------------

## Expected project structure

Put the files in a single project folder using a structure like this:

``` text
Housing-Price-Prediction/
│
├── data/
│   ├── ames_housing.csv
│   ├── king_county_house_data.csv
│   ├── Housing.csv
│   └── world_housing.csv
│
├── Dataset load.py
├── Modelling.py
├── cleaned_housing_data.csv         # generated after preprocessing
├── README_Housing_Price_Prediction.md
└── requirements.txt
```

### Important

If your file names are slightly different, update the names in the Python scripts or rename your files to match the names expected by the code.

------------------------------------------------------------------------

## Datasets used

This project was built from a combined dataset using the following
sources:

1.  **Ames Housing dataset**
2.  **King County housing dataset**
3.  **Housing.csv**
4.  **World housing dataset**

### Notes on reproducibility

To get results that are as close as possible to the original project:

-   use the **same raw datasets**
-   keep the **same column names**
-   use the **same preprocessing logic**
-   use the **same train/test split and random seed**
-   use the **same package versions** whenever possible

If any dataset is updated, renamed, filtered differently, or encoded differently, the final metrics may change.

------------------------------------------------------------------------

## Python version

Use:

``` text
Python 3.12.x
```

A virtual environment is strongly recommended.

------------------------------------------------------------------------

## Step 1: Create and activate a virtual environment

### On macOS / Linux

``` bash
python3 -m venv .venv
source .venv/bin/activate
```

### On Windows (Command Prompt)

``` bash
py -3 -m venv .venv
.venv\Scripts\activate
```

### On Windows (PowerShell)

``` powershell
py -3 -m venv .venv
.venv\Scripts\Activate.ps1
```

------------------------------------------------------------------------

## Step 2: Install dependencies

Install the required Python libraries:

``` bash
pip install numpy pandas scikit-learn xgboost catboost pyswarms matplotlib seaborn shap scipy joblib
```
------------------------------------------------------------------------

## Step 3: Place the dataset files in the correct folder

Create a folder named `data` inside the project directory and place all
raw CSV files there.

Example:

``` text
data/
├── ames_housing.csv
├── king_county_house_data.csv
├── Housing.csv
└── world_housing.csv
```

If your script currently reads files from another location, either:

-   edit the paths in the code, or
-   move the files so they match the code exactly

------------------------------------------------------------------------

## Step 4: Run preprocessing

The preprocessing script is assumed to be:

``` text
Dataset load.py
```

Run it with:

### macOS / Linux

``` bash
python3 "Dataset load.py"
```

### Windows

``` bash
python "Dataset load.py"
```

This script should:

-   load all source datasets
-   clean and standardize columns
-   merge the datasets
-   engineer features
-   remove invalid or extreme values if your code does so
-   save the final processed file as:

``` text
cleaned_housing_data.csv
```

### Expected output

After this step, you should see:

``` text
cleaned_housing_data.csv
```

in your main project folder.

------------------------------------------------------------------------

## Step 5: Run the modeling script

The main modeling script is assumed to be:

``` text
Modelling.py
```

Run it with:

### macOS / Linux

``` bash
python3 "Modelling.py"
```

### Windows

``` bash
python "Modelling.py"
```

This script should:

-   load `cleaned_housing_data.csv`
-   prepare features and target
-   split the data into training and testing sets
-   train XGBoost
-   optimize XGBoost with PSO
-   train CatBoost
-   print performance metrics
-   generate feature importance and SHAP outputs

## Suggested command sequence

After placing the files correctly, a new user should be able to run:

### macOS / Linux

``` bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python3 "Dataset load.py"
python3 "Modelling.py"
```

### Windows

``` bash
py -3 -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
python "Dataset load.py"
python "Modelling.py"
```

------------------------------------------------------------------------

## Reproducibility statement

This project is reproducible **if the same code, raw data, preprocessing
logic, and package versions are used**. Exact numerical equality across
all systems is not always guaranteed, but following this README should
allow another user to reproduce the workflow and obtain substantially
the same findings.

------------------------------------------------------------------------

## Citation context

This project was developed from three parent papers focused on house
price prediction with XGBoost, PSO-XGBoost, and CatBoost-based
comparisons.

------------------------------------------------------------------------

## Author note

This README was prepared for the DS340W housing price prediction project
so another user can set up the environment, run the preprocessing and
modeling pipeline, and reproduce the project results as closely as
possible.
