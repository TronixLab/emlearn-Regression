# emlearn-Regression

Technical documentation for the `TronixLab/emlearn-Regression` repository.

## Project background

This repository demonstrates how classical machine-learning regression models can be trained in Python and then deployed to embedded targets with [emlearn](https://emlearn.readthedocs.io/). The project uses the Portuguese *Vinho Verde* wine-quality datasets to predict wine quality scores from physicochemical measurements, then exports the trained models as C/C++ headers for Arduino-friendly inference.

Two workflows are documented in the repository:

- **Ensemble regression on the white wine dataset**
- **Neural-network regression on the red wine dataset**

The repository is therefore both:

- a set of notebook-based training experiments, and
- a collection of generated embedded inference artifacts.

## Repository purpose

The codebase shows an end-to-end path from dataset exploration to embedded deployment:

1. load wine-quality CSV datasets,
2. analyze feature/target relationships,
3. select the most relevant features,
4. scale the selected features,
5. train regression models,
6. evaluate them in notebooks, and
7. export the trained models to header files and Arduino sketches.

## Top-level structure

```text
.
├── Ensemble Regression/
│   ├── WhiteWineQuality_Regression.ipynb
│   ├── DecisionTreeRegressor.h
│   ├── RandomForestRegressor.h
│   ├── ExtraTreesRegressor.h
│   ├── emlearn_Arduino_dt_Regression/
│   ├── emlearn_Arduino_rf_Regression/
│   └── emlearn_Arduino_et_Regression/
├── Neural Network Regression/
│   ├── RedWineQuality_Regression.ipynb
│   ├── MLPRegressor.h
│   ├── NeuralNetworkRegressor.h
│   ├── emlearn_Arduino_mlp_Regression/
│   └── emlearn_Arduino_dnn_Regression/
├── winequality-red.csv
└── winequality-white.csv
```

## Data assets

### `winequality-white.csv`

- White wine dataset used by `Ensemble Regression/WhiteWineQuality_Regression.ipynb`
- 11 physicochemical input variables plus the `quality` target
- Notebook background text states the dataset contains **4,898 samples**

### `winequality-red.csv`

- Red wine dataset used by `Neural Network Regression/RedWineQuality_Regression.ipynb`
- 11 physicochemical input variables plus the `quality` target
- Notebook background text states the dataset contains **1,599 samples**

Both datasets use the same feature schema:

- fixed acidity
- volatile acidity
- citric acid
- residual sugar
- chlorides
- free sulfur dioxide
- total sulfur dioxide
- density
- pH
- sulphates
- alcohol
- quality (target)

## Main repository workflows

### 1. Ensemble regression workflow

Primary notebook:

- `/home/runner/work/emlearn-Regression/emlearn-Regression/Ensemble Regression/WhiteWineQuality_Regression.ipynb`

This notebook performs:

- dataset loading and exploratory analysis,
- feature/target correlation inspection,
- feature selection with `SelectKBest(f_regression, k=7)`,
- feature scaling with `MinMaxScaler`,
- training of:
  - `DecisionTreeRegressor`
  - `RandomForestRegressor`
  - `ExtraTreesRegressor`
- evaluation with MAE, MSE, RMSE, and R²,
- export to C/C++ with `emlearn.convert(..., method='inline', return_type='regressor')`

#### Selected white-wine feature subset

The embedded ensemble examples operate on 7 selected features. Based on the scaler headers, the serial input order used by the Arduino sketches is:

1. fixed acidity
2. volatile acidity
3. chlorides
4. total sulfur dioxide
5. density
6. pH
7. alcohol

#### Generated ensemble headers

- `DecisionTreeRegressor.h`
  - generated inline C model
  - exposes `DecisionTreeRegressor_predict(const float *features, int32_t features_length)`
- `RandomForestRegressor.h`
  - generated inline C model
  - exposes `RandomForestRegressor_predict(...)`
- `ExtraTreesRegressor.h`
  - generated inline C model
  - exposes `ExtraTreesRegressor_predict(...)`

These files are large because emlearn expands the learned trees directly into nested branch logic.

#### Embedded preprocessing for ensemble models

Each Arduino ensemble example ships with `MinMaxScaler.h`, which contains the fitted min/max constants for the 7 selected features and provides:

- `transform(float* X, float* X_scaled)`
- `inverse_transform(float* X_scaled, float* X)`

#### Arduino ensemble sketches

- `emlearn_Arduino_dt_Regression/emlearn_Arduino_dt_Regression.ino`
- `emlearn_Arduino_rf_Regression/emlearn_Arduino_rf_Regression.ino`
- `emlearn_Arduino_et_Regression/emlearn_Arduino_et_Regression.ino`

Each sketch:

- reads one comma-separated line from `Serial`,
- parses **7 float values**,
- scales them with `MinMaxScaler`,
- calls the exported regressor,
- prints `Prediction: <value>`

### 2. Neural-network regression workflow

Primary notebook:

- `/home/runner/work/emlearn-Regression/emlearn-Regression/Neural Network Regression/RedWineQuality_Regression.ipynb`

This notebook performs:

- dataset loading and exploratory analysis,
- feature selection with `SelectKBest(f_regression, k=7)`,
- scaling with `RobustScaler`,
- train/test/validation splitting,
- training of:
  - scikit-learn `MLPRegressor`
  - Keras dense neural network
- evaluation with MAE, MSE, RMSE, and R²,
- export to emlearn-compatible C/C++ headers

#### Selected red-wine feature subset

The neural-network deployment examples also use 7 selected features. The order embedded in the scaler headers is:

1. volatile acidity
2. citric acid
3. chlorides
4. total sulfur dioxide
5. density
6. sulphates
7. alcohol

#### Generated neural-network headers

- `MLPRegressor.h`
  - loadable emlearn network generated from scikit-learn `MLPRegressor`
  - exposes `MLPRegressor_regress1(const float *features, int32_t n_features)`
- `NeuralNetworkRegressor.h`
  - loadable emlearn network generated from the Keras dense model
  - exposes `NeuralNetworkRegressor_regress1(...)`

Unlike the inline tree models, these headers store layer weights and biases and rely on the emlearn neural-network runtime via:

- `#include <eml_net.h>`

#### Embedded preprocessing for neural models

Each neural-network Arduino example ships with `RobustScaler.h`, which stores:

- per-feature medians
- per-feature interquartile ranges (IQR)

It provides:

- `transform(float* X, float* X_scaled)`
- `inverse_transform(float* X_scaled, float* X)`

#### Arduino neural-network sketches

- `emlearn_Arduino_mlp_Regression/emlearn_Arduino_mlp_Regression.ino`
- `emlearn_Arduino_dnn_Regression/emlearn_Arduino_dnn_Regression.ino`

Each sketch:

- reads one comma-separated line from `Serial`,
- parses **7 float values**,
- scales them with `RobustScaler`,
- runs the exported regressor,
- prints `Prediction: <value>`

## File-by-file evaluation

### Notebooks

#### `Ensemble Regression/WhiteWineQuality_Regression.ipynb`

- Acts as the training and export pipeline for white-wine ensemble models
- Includes exploratory analysis, feature selection, preprocessing, model training, and emlearn conversion
- Evaluation output in the notebook shows `ExtraTreesRegressor` as the strongest of the three embedded ensemble candidates

#### `Neural Network Regression/RedWineQuality_Regression.ipynb`

- Acts as the training and export pipeline for red-wine neural regressors
- Compares scikit-learn MLP and a Keras dense network
- Produces embedded headers for both variants

### Generated model headers

These files should be treated as generated artifacts rather than hand-maintained source:

- `Ensemble Regression/DecisionTreeRegressor.h`
- `Ensemble Regression/RandomForestRegressor.h`
- `Ensemble Regression/ExtraTreesRegressor.h`
- `Neural Network Regression/MLPRegressor.h`
- `Neural Network Regression/NeuralNetworkRegressor.h`

### Scaler headers

These files capture fitted preprocessing parameters required to reproduce notebook inference behavior on-device:

- `Ensemble Regression/emlearn_Arduino_dt_Regression/MinMaxScaler.h`
- `Ensemble Regression/emlearn_Arduino_rf_Regression/MinMaxScaler.h`
- `Ensemble Regression/emlearn_Arduino_et_Regression/MinMaxScaler.h`
- `Neural Network Regression/emlearn_Arduino_mlp_Regression/RobustScaler.h`
- `Neural Network Regression/emlearn_Arduino_dnn_Regression/RobustScaler.h`

### Arduino sketches

The `.ino` files are thin integration layers around the generated model headers. They do not train models; they only:

- receive serial input,
- preprocess it,
- invoke a predictor,
- print the regression output

## How to use this repository

### Reproduce the training workflow

Open the notebooks and rerun the training/export cells with a Python environment that includes:

- pandas
- seaborn
- matplotlib
- scikit-learn
- keras / tensorflow
- emlearn

Note: the notebooks contain hard-coded local Windows dataset paths in their load cells. When rerunning them in a new environment, update those paths to point at the CSV files in this repository root.

### Run the embedded examples

1. Open the relevant Arduino sketch directory.
2. Build the sketch in the Arduino IDE or with Arduino CLI.
3. Provide a serial line containing **7 comma-separated feature values** in the exact order required by the matching scaler.
4. Read the predicted wine-quality score from the serial output.

## Architectural summary

The repository separates responsibilities cleanly:

- **datasets** live at the root,
- **training and evaluation** live in notebooks,
- **embedded inference logic** lives in generated headers,
- **device integration** lives in Arduino sketches,
- **preprocessing parity** is preserved through generated scaler headers

This makes the repository a compact reference project for taking regression models from notebook experimentation to microcontroller deployment with emlearn.

## Tooling status

No repository-level automation for linting, testing, packaging, or builds is currently present in the checked-in files. The project is organized primarily as notebooks plus generated embedded artifacts.
