import pandas as pd
import joblib

model = joblib.load('models/linear_model_v1.pkl')
df = pd.read_csv('data/04 sampregdata.csv')
X_new = df[['x3']]

predictions = model.predict(X_new)
df['predictions'] = predictions
df = df.drop(df.columns[0], axis=1) # Drop first column that is not needed
df.to_csv('predictions/predictions_model_v1.csv', index=False)