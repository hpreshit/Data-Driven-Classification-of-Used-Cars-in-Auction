import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE
from sklearn.decomposition import PCA
from scipy.stats import chi2_contingency

# Load dataset
file_path = "cleaned_training.csv"
car_data = pd.read_csv(file_path)

# Reducing number of categories (Grouping categories with count < 500)
categorical_cols = car_data.select_dtypes(include=['object']).columns
for col in categorical_cols:
    value_counts = car_data[col].value_counts()
    rare_categories = value_counts[value_counts < 500].index
    car_data[col] = car_data[col].replace(rare_categories, "Other")

# Encoding categorical variables
label_encoders = {}
for col in categorical_cols:
    le = LabelEncoder()
    car_data[col] = le.fit_transform(car_data[col])
    label_encoders[col] = le

# Feature selection: Chi-Square Test
chi_scores = {}
target_col = "IsBadBuy"
for col in categorical_cols:
    if col != target_col:
        contingency_table = pd.crosstab(car_data[col], car_data[target_col])
        chi2, p, _, _ = chi2_contingency(contingency_table)
        chi_scores[col] = (chi2, p)

# Dropping redundant columns based on Chi-Square and manual selection
drop_columns = ["Color", "TopThreeAmericanName", "SubModel", "Model"]
car_data = car_data.drop(columns=drop_columns)

# One-Hot Encoding
car_data = pd.get_dummies(car_data, drop_first=True)

# Splitting into train-test sets
train_data, test_data = train_test_split(car_data, test_size=0.15, random_state=9, stratify=car_data[target_col])

# Saving test data before scaling
test_data.to_csv("Test_Without_Scaling.csv", index=False)

# SMOTE: Balancing the dataset
X_train = train_data.drop(columns=[target_col])
y_train = train_data[target_col]
smote = SMOTE(sampling_strategy=1, random_state=9)
X_train_balanced, y_train_balanced = smote.fit_resample(X_train, y_train)

# Standard Scaling
scaler = StandardScaler()
X_train_balanced = pd.DataFrame(scaler.fit_transform(X_train_balanced), columns=X_train.columns)
test_data[X_train.columns] = scaler.transform(test_data[X_train.columns])

# Saving the processed train and test datasets
train_balanced = pd.concat([X_train_balanced, y_train_balanced.reset_index(drop=True)], axis=1)
train_balanced.to_csv("Balanced_Train.csv", index=False)
test_data.to_csv("Balanced_Test.csv", index=False)

# Principal Component Analysis
pca = PCA(n_components=0.95)  # Adjusted to 95% variance explained
X_pca = pca.fit_transform(X_train_balanced)
pca_df = pd.DataFrame(X_pca, columns=[f'PC{i+1}' for i in range(X_pca.shape[1])])
pca_df[target_col] = y_train_balanced.reset_index(drop=True)

# Normalizing PCA components
pca_df.iloc[:, :-1] = (pca_df.iloc[:, :-1] - pca_df.iloc[:, :-1].min()) / (pca_df.iloc[:, :-1].max() - pca_df.iloc[:, :-1].min())

# Saving PCA dataset
pca_df.to_csv("car_pca_df.csv", index=False)

# Binning numerical variables for Association Rule Mining (ARM)
numerical_cols = X_train.columns
for col in numerical_cols:
    # Check the number of unique values in the column
    unique_values = car_data[col].nunique()
    
    if unique_values >= 4:
        # If the column has at least 4 unique values, perform binning with q=4
        try:
            car_data[col] = pd.qcut(car_data[col], q=4, labels=[f"{col}-Q1", f"{col}-Q2", f"{col}-Q3", f"{col}-Q4"], duplicates='drop')
        except ValueError:
            print(f"Skipping binning for {col} (too few unique values for binning)")
    else:
        # If there are fewer than 4 unique values, skip binning and handle differently
        print(f"Skipping binning for {col} (only {unique_values} unique values)")
        # You can skip binning or convert the column to a categorical string type
        car_data[col] = car_data[col].astype(str)  # Convert to string or handle in another way

# Saving ARM dataset
car_data.to_csv("rules_df.csv", index=False)


print("Data processing complete!")
