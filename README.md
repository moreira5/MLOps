## Model Comparison

The initial model (`model_v1`) was trained using a single predictor (`x3`) to perform linear regression. While it provided useful insights, an improved model (`model_v2`) was developed using two predictors (`x2` and `x3`). The second model (`model_v2`) showed better predictive performance, achieving a higher R² score.

## How to Run the Model

### 1. Clone the Repository
If you haven’t already, clone the repository:
```
git clone https://github.com/your-username/mlops.git
cd mlops
```

### 2. Install Dependencies
Ensure you have Python installed, then run:
```sh
pip install -r requirements.txt
```

### 3. Run the Model for Predictions
To use the trained model for generating predictions, run:
```sh
python scripts/run_model.py
```
This will save the predictions in `data/predictions.csv`.

## File Structure
```
mlops/
│── data/        # Dataset files
│── models/      # Saved trained models
│── scripts/     # Python scripts for training and predictions
│── README.md    # Documentation
│── requirements.txt # Dependencies
│── .gitignore   # Ignored files
```
