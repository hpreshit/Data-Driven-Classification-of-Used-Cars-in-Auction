import pandas as pd
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
import seaborn as sns

url = 'https://raw.githubusercontent.com/Harshada13/Data-Driven-Classification-of-Used-Cars-in-Auction/refs/heads/main/training.csv'

# 1. Reading the training data CSV and storing it into a dataframe
df = pd.read_csv(url)

# 2. Display structure and summary
print(df.info())
print(df.describe())

# 3. Replacing string 'NULL' with NaN
df.replace('NULL', np.nan, inplace=True)

# 4. Identifying the number of NAs
print(df.isna().sum())

# 5. Replacing missing values with mode at the make/model level
# Function to get mode
def get_mode(series):
    return series.mode()[0]

# Replace missing 'Trim' with mode for each group of Make and Model
df['Trim'] = df.groupby(['Make', 'Model'])['Trim'].transform(
    lambda x: x.fillna(x.mode()[0] if not x.mode().empty else x.iloc[0])
)
# 6. Replace missing values in other columns with their mode
columns_to_replace_with_mode = ['SubModel', 'Color', 'Transmission', 'WheelType', 'Nationality', 'Size', 'TopThreeAmericanName']
for column in columns_to_replace_with_mode:
    df[column] = df[column].fillna(df[column].mode()[0])

# 7. Handle columns with too many missing values
df.drop(columns=['PRIMEUNIT', 'AUCGUART'], inplace=True)

# 8. Remove unimportant columns
columns_to_drop = ['WheelTypeID', 'VehYear', 'PurchDate', 'PRIMEUNIT', 'AUCGUART', 'BYRNO', 'RefId', 'VNZIP1']
df.drop(columns=columns_to_drop, inplace=True, errors='ignore')

# 9. Convert data types of categorical variables to 'category'
cat_vars = ['IsBadBuy', 'Auction', 'Make', 'Model', 'Trim', 'SubModel', 'Color', 'Transmission', 'WheelType', 'Nationality', 'Size', 'TopThreeAmericanName', 'VNST', 'IsOnlineSale']
for col in cat_vars:
    df[col] = df[col].astype('category')

# 10. Convert numeric columns to 'float'
num_vars = ['VehicleAge', 'VehOdo', 'MMRAcquisitionAuctionAveragePrice', 'MMRAcquisitionAuctionCleanPrice', 'MMRAcquisitionRetailAveragePrice', 
            'MMRAcquisitonRetailCleanPrice', 'MMRCurrentAuctionAveragePrice', 'MMRCurrentAuctionCleanPrice', 'MMRCurrentRetailAveragePrice', 
            'MMRCurrentRetailCleanPrice', 'VehBCost', 'WarrantyCost']
df[num_vars] = df[num_vars].apply(pd.to_numeric, errors='coerce')

# 11. Handling missing numeric data by imputing with the mean
df['MMRAcquisitionAuctionAveragePrice'].fillna(df['MMRAcquisitionAuctionAveragePrice'].mean(), inplace=True)
df['MMRAcquisitionAuctionCleanPrice'].fillna(df['MMRAcquisitionAuctionCleanPrice'].mean(), inplace=True)
df['MMRAcquisitionRetailAveragePrice'].fillna(df['MMRAcquisitionRetailAveragePrice'].mean(), inplace=True)
df['MMRAcquisitonRetailCleanPrice'].fillna(df['MMRAcquisitonRetailCleanPrice'].mean(), inplace=True)
df['MMRCurrentAuctionAveragePrice'].fillna(df['MMRCurrentAuctionAveragePrice'].mean(), inplace=True)
df['MMRCurrentAuctionCleanPrice'].fillna(df['MMRCurrentAuctionCleanPrice'].mean(), inplace=True)
df['MMRCurrentRetailAveragePrice'].fillna(df['MMRCurrentRetailAveragePrice'].mean(), inplace=True)
df['MMRCurrentRetailCleanPrice'].fillna(df['MMRCurrentRetailCleanPrice'].mean(), inplace=True)

# 12. Winsorizing outliers for numeric variables (using z-scores)
def winsorize_outliers(df, column):
    z_scores = np.abs(stats.zscore(df[column].dropna()))
    df[column] = df[column].where(z_scores < 3, df[column].median())  # Replace outliers with median
    return df

# Apply Winsorizing on specific columns
outlier_columns = ['VehicleAge', 'VehOdo', 'VehBCost', 'WarrantyCost']
for col in outlier_columns:
    df = winsorize_outliers(df, col)

# 13. Display histograms after handling outliers
for col in num_vars:
    plt.figure(figsize=(8, 6))
    sns.histplot(df[col], kde=True, bins=30)
    plt.title(f"Distribution of {col}")
    plt.show()

# 14. Checking final data types and missing values
print(df.info())
print(df.isna().sum())

# Save the cleaned dataframe to a new CSV file
df.to_csv("cleaned_training.csv", index=False)

print("Cleaned data saved to 'cleaned_training.csv'.")