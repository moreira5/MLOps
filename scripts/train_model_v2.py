import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score

# Load data
df = pd.read_csv('data/04 sampregdata.csv')

# Choose the best two X's
X = df[['x2','x3']]
y = df['y']

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train model
model = LinearRegression()
model.fit(X_train, y_train)

# Evaluate
y_pred = model.predict(X_test)
print(f'R^2 Score: {r2_score(y_test, y_pred)}')

# Save the model
import joblib
joblib.dump(model, 'models/linear_model_v2.pkl')
