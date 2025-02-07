import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix
import joblib

# Load the data
gbm_train = pd.read_csv("balanced_train.csv")
gbm_test = pd.read_csv("balanced_test.csv")

# Define the features and target variable
X_train = gbm_train.drop(columns=['IsBadBuy'])
y_train = gbm_train['IsBadBuy']
X_test = gbm_test.drop(columns=['IsBadBuy'])
y_test = gbm_test['IsBadBuy']

# Define the parameter grid for tuning
param_grid = {
    'n_estimators': [600, 900],
    'learning_rate': [0.01],
    'max_depth': [5, 6, 7],
    'min_samples_split': [10]
}

# Define the model and the grid search
gbm = GradientBoostingClassifier()
grid_search = GridSearchCV(estimator=gbm, param_grid=param_grid, cv=3, scoring='accuracy')

# Train the model
grid_search.fit(X_train, y_train)

# Print the best parameters and the best score
print("Best parameters found: ", grid_search.best_params_)
print("Best accuracy found: ", grid_search.best_score_)

# Predict and evaluate
y_pred = grid_search.predict(X_test)
print("Accuracy: ", accuracy_score(y_test, y_pred))
print("Confusion Matrix: \n", confusion_matrix(y_test, y_pred, labels=[1, 0]))

# Optional: Save the trained model for future use
joblib.dump(grid_search, 'gbm_tuned_model.pkl')
