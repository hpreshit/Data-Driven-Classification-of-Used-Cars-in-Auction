import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

# Load cleaned data (assuming it's saved as a pickle file)
car_buy_clean_df = pd.read_csv("cleaned_training.csv")

# Set seaborn style
sns.set(style="whitegrid")

# Bar graph: Target variable vs Nationality
plt.figure(figsize=(12, 6))
sns.histplot(data=car_buy_clean_df, x="IsBadBuy", hue="Nationality", stat="percent", multiple="dodge")
plt.title("Percentage of Bad Buys by Nationality")
plt.ylabel("Percent")
plt.show()

# Bar graph: Target variable vs Transmission
plt.figure(figsize=(12, 6))
sns.histplot(data=car_buy_clean_df, x="IsBadBuy", hue="Transmission", stat="percent", multiple="dodge", color="#FF6666")
plt.title("Transmission vs % of Bad Buys")
plt.ylabel("Percent")
plt.show()

# Count of cars sold by Transmission
transmission_count = car_buy_clean_df.groupby(["Transmission", "IsBadBuy"]).size().reset_index(name="count")
plt.figure(figsize=(12, 6))
sns.barplot(data=transmission_count, x="Transmission", y="count", hue="IsBadBuy")
plt.title("Transmission vs Number of Cars Sold by Bad Buy Categories")
plt.xticks(rotation=90)
plt.show()

# Similar bar plots for Auction, TopThreeAmericanName, Size, WheelType, VNST, VehicleAge
features = ["Auction", "TopThreeAmericanName", "Size", "WheelType", "VNST", "VehicleAge"]
for feature in features:
    plt.figure(figsize=(12, 6))
    sns.histplot(data=car_buy_clean_df, x="IsBadBuy", hue=feature, stat="percent", multiple="dodge")
    plt.title(f"{feature} vs % of Bad Buys")
    plt.ylabel("Percent")
    plt.show()

# Vehicle Odometer vs Bad Buy
car_buy_clean_df["OdometerGroup"] = pd.qcut(car_buy_clean_df["VehOdo"], q=4)
plt.figure(figsize=(12, 6))
sns.histplot(data=car_buy_clean_df, x="IsBadBuy", hue="OdometerGroup", stat="percent", multiple="dodge", color="#FF6666")
plt.title("% of Bad Buy vs Vehicle Odometer Reading Groups")
plt.ylabel("Percent")
plt.show()

# Acquisition Cost vs Bad Buy
car_buy_clean_df["VehBCostGroup"] = pd.qcut(car_buy_clean_df["VehBCost"], q=4)
plt.figure(figsize=(12, 6))
sns.histplot(data=car_buy_clean_df, x="IsBadBuy", hue="VehBCostGroup", stat="percent", multiple="dodge", color="#FF6666")
plt.title("% of Bad Buy vs Vehicle Acquisition Cost Groups")
plt.ylabel("Percent")
plt.show()

# Warranty Cost vs Bad Buy
car_buy_clean_df["WarrantyCostGroup"] = pd.qcut(car_buy_clean_df["WarrantyCost"], q=4)
plt.figure(figsize=(12, 6))
sns.histplot(data=car_buy_clean_df, x="IsBadBuy", hue="WarrantyCostGroup", stat="percent", multiple="dodge", color="#FF6666")
plt.title("% of Bad Buy vs Warranty Cost Groups")
plt.ylabel("Percent")
plt.show()

# Online vs Offline Sales
plt.figure(figsize=(12, 6))
sns.histplot(data=car_buy_clean_df, x="IsBadBuy", hue="IsOnlineSale", stat="percent", multiple="dodge")
plt.title("% of Bad Buy vs Online/Offline Sales")
plt.ylabel("Percent")
plt.show()

# Convert 'IsBadBuy' to categorical if it's not already
car_buy_clean_df['IsBadBuy'] = car_buy_clean_df['IsBadBuy'].astype('category')

# Create bins for acquisition prices
bins = np.arange(0, car_buy_clean_df['VehOdo'].max() + 5000, 5000)
car_buy_clean_df['OdometerGroup'] = pd.cut(df['VehOdo'], bins=bins)

# Compute the proportion of IsBadBuy within each bin
prop_bad_buy = car_buy_clean_df.groupby('OdometerGroup')['IsBadBuy'].mean()
count_bad_buy = car_buy_clean_df.groupby('OdometerGroup')['IsBadBuy'].count()

# Combine into a new DataFrame for plotting
plot_df = pd.DataFrame({'OdometerGroup': prop_bad_buy.index, 'ProportionBadBuy': prop_bad_buy.values, 'Count': count_bad_buy.values})

# Plot using seaborn
plt.figure(figsize=(12, 6))
sns.barplot(data=plot_df, x='OdometerGroup', y='ProportionBadBuy', palette='coolwarm')
plt.xticks(rotation=45)
plt.xlabel("Odometer Range")
plt.ylabel("Proportion of Bad Buys")
plt.title("Proportion of Bad Buys by Odometer Reading")
plt.show()