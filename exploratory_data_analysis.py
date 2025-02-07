import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

# Reading cleaned data into DataFrame
car_buy_clean_df = pd.read_pickle("cleaned_training.pkl")

# Bar graph for Plotting Target variable vs Nationality
# The % of Bad Buy's are equal amongst all the nationalities and hence Nationality may not be an important feature
plt.figure(figsize=(10, 6))
sns.catplot(x="IsBadBuy", hue="IsBadBuy", col="Nationality", data=car_buy_clean_df, kind="count", height=4, aspect=0.7)
plt.show()

# Bar graph for Target variable vs Transmission
# Both AUTO & MANUAL Transmission have approximately equal proportion of Bad Buys
plt.figure(figsize=(10, 6))
sns.catplot(x="IsBadBuy", hue="IsBadBuy", col="Transmission", data=car_buy_clean_df, kind="count", height=4, aspect=0.7)
plt.show()

# Bar graph for Target variable vs Auction
# Auction ADESA has slightly higher % of bad buys
plt.figure(figsize=(10, 6))
sns.catplot(x="IsBadBuy", hue="IsBadBuy", col="Auction", data=car_buy_clean_df, kind="count", height=4, aspect=0.7)
plt.show()

# Bar graph for Target variable vs TopThreeAmericanNames
# The company Ford higher % of bad buys in comparison to other car companies
plt.figure(figsize=(10, 6))
sns.catplot(x="IsBadBuy", hue="IsBadBuy", col="TopThreeAmericanName", data=car_buy_clean_df, kind="count", height=4, aspect=0.7)
plt.show()

# SIZE VS Proportion of Bad Buys
# Size does not play a significant role in determining whether car is a good buy or bad buy
car_count_by_size = car_buy_clean_df.groupby(['Size', 'IsBadBuy']).size().reset_index(name='count_cars')
plt.figure(figsize=(10, 6))
sns.barplot(x="Size", y="count_cars", hue="IsBadBuy", data=car_count_by_size)
plt.xticks(rotation=90)
plt.title("Size VS proportion of Bad Buys")
plt.show()

# The Wheel Type OTHER has a high proportion of Bad Buys
car_count_by_WheelType = car_buy_clean_df.groupby(['WheelType', 'IsBadBuy']).size().reset_index(name='count_cars')
plt.figure(figsize=(10, 6))
sns.barplot(x="WheelType", y="count_cars", hue="IsBadBuy", data=car_count_by_WheelType)
plt.xticks(rotation=90)
plt.title("Wheel Type VS Number of Cars sold by Bad Buy categories")
plt.show()

# There is a high % of Bad Buys amongst "OTHER" wheel type
plt.figure(figsize=(10, 6))
sns.catplot(x="IsBadBuy", hue="IsBadBuy", col="WheelType", data=car_buy_clean_df, kind="count", height=4, aspect=0.7)
plt.show()

# There is not much significant difference in the % of bad buys amongst different makes
car_count_by_make = car_buy_clean_df.groupby(['Make', 'IsBadBuy']).size().reset_index(name='count_cars')
plt.figure(figsize=(10, 6))
sns.barplot(x="Make", y="count_cars", hue="IsBadBuy", data=car_count_by_make)
plt.xticks(rotation=90)
plt.title("Make VS proportion of Bad Buys")
plt.show()

# We can observe that Florida and Texas states have the highest number of car sales and hence highest proportion of Bad Buys
car_count_by_VNST = car_buy_clean_df.groupby(['VNST', 'IsBadBuy']).size().reset_index(name='count_cars')
plt.figure(figsize=(10, 6))
sns.barplot(x="VNST", y="count_cars", hue="IsBadBuy", data=car_count_by_VNST)
plt.xticks(rotation=90)
plt.title("VNST VS Number of Cars sold by Bad Buy categories")
plt.show()

# Plotting Vehicle Age with Target Variable
# The % of bad buy increases with vehicle age
car_count_by_VehicleAge = car_buy_clean_df[car_buy_clean_df['VehicleAge'] != 0]
car_count_by_VehicleAge['VehicleAge'] = car_count_by_VehicleAge['VehicleAge'].astype(str)
plt.figure(figsize=(10, 6))
sns.catplot(x="IsBadBuy", hue="IsBadBuy", col="VehicleAge", data=car_count_by_VehicleAge, kind="count", height=4, aspect=0.7)
plt.show()

# Plotting Vehicle Odometer with Target Variable
# We can observe that % of Bad Buys increases with higher vehicle odometer reading groups
car_count_by_VehicleAge['VehOdoGroups'] = pd.qcut(car_count_by_VehicleAge['VehOdo'], q=4, labels=False)
plt.figure(figsize=(10, 6))
sns.catplot(x="IsBadBuy", hue="IsBadBuy", col="VehOdoGroups", data=car_count_by_VehicleAge, kind="count", height=4, aspect=0.7)
plt.show()

# Plotting VehBCost - Acquisition cost paid for the vehicle with Target Variable
# We can observe that % of Bad Buys is higher for lower Acquistion cost groups
car_count_by_VehicleAge['VehBCostGroups'] = pd.qcut(car_count_by_VehicleAge['VehBCost'], q=4, labels=False)
plt.figure(figsize=(10, 6))
sns.catplot(x="IsBadBuy", hue="IsBadBuy", col="VehBCostGroups", data=car_count_by_VehicleAge, kind="count", height=4, aspect=0.7)
plt.show()

# Plotting WarrantyCost with Target Variable
# We can observe that %(proportion) of Bad Buys increases for higher Warranty Costs
car_count_by_VehicleAge['WarrantyCostGroups'] = pd.qcut(car_count_by_VehicleAge['WarrantyCost'], q=4, labels=False)
plt.figure(figsize=(10, 6))
sns.catplot(x="IsBadBuy", hue="IsBadBuy", col="WarrantyCostGroups", data=car_count_by_VehicleAge, kind="count", height=4, aspect=0.7)
plt.show()

# Plotting Online/Offline Sales with Target Variable
plt.figure(figsize=(10, 6))
sns.catplot(x="IsBadBuy", hue="IsBadBuy", col="IsOnlineSale", data=car_count_by_VehicleAge, kind="count", height=4, aspect=0.7)
plt.show()

# Plotting Acquisition Auction Average Price with Target Variable
car_count_by_VehicleAge['MMRAcquisitionAuctionAveragePriceGroups'] = pd.qcut(car_count_by_VehicleAge['MMRAcquisitionAuctionAveragePrice'], q=4, labels=False)
plt.figure(figsize=(10, 6))
sns.catplot(x="IsBadBuy", hue="IsBadBuy", col="MMRAcquisitionAuctionAveragePriceGroups", data=car_count_by_VehicleAge, kind="count", height=4, aspect=0.7)
plt.show()

# Plotting Acquisition Auction Clean Price with Target Variable
car_count_by_VehicleAge['MMRAcquisitionAuctionCleanPriceGroups'] = pd.qcut(car_count_by_VehicleAge['MMRAcquisitionAuctionCleanPrice'], q=4, labels=False)
plt.figure(figsize=(10, 6))
sns.catplot(x="IsBadBuy", hue="IsBadBuy", col="MMRAcquisitionAuctionCleanPriceGroups", data=car_count_by_VehicleAge, kind="count", height=4, aspect=0.7)
plt.show()

# Plotting Acquisition Retail Average Price with Target Variable
car_count_by_VehicleAge['MMRAcquisitionRetailAveragePriceGroups'] = pd.qcut(car_count_by_VehicleAge['MMRAcquisitionRetailAveragePrice'], q=4, labels=False)
plt.figure(figsize=(10, 6))
sns.catplot(x="IsBadBuy", hue="IsBadBuy", col="MMRAcquisitionRetailAveragePriceGroups", data=car_count_by_VehicleAge, kind="count", height=4, aspect=0.7)
plt.show()

# Plotting Acquisition Retail Clean Price with Target Variable
car_count_by_VehicleAge['MMRAcquisitionRetailCleanPriceGroups'] = pd.qcut(car_count_by_VehicleAge['MMRAcquisitonRetailCleanPrice'], q=4, labels=False)
plt.figure(figsize=(10, 6))
sns.catplot(x="IsBadBuy", hue="IsBadBuy", col="MMRAcquisitionRetailCleanPriceGroups", data=car_count_by_VehicleAge, kind="count", height=4, aspect=0.7)
plt.show()

# Plotting Current Auction Average Price with Target Variable
car_count_by_VehicleAge['MMRCurrentAuctionAveragePriceGroups'] = pd.qcut(car_count_by_VehicleAge['MMRCurrentAuctionAveragePrice'], q=4, labels=False)
plt.figure(figsize=(10, 6))
sns.catplot(x="IsBadBuy", hue="IsBadBuy", col="MMRCurrentAuctionAveragePriceGroups", data=car_count_by_VehicleAge, kind="count", height=4, aspect=0.7)
plt.show()

# Plotting Current Auction Clean Price with Target Variable
car_count_by_VehicleAge['MMRCurrentAuctionCleanPriceGroups'] = pd.qcut(car_count_by_VehicleAge['MMRCurrentAuctionCleanPrice'], q=4, labels=False)
plt.figure(figsize=(10, 6))
sns.catplot(x="IsBadBuy", hue="IsBadBuy", col="MMRCurrentAuctionCleanPriceGroups", data=car_count_by_VehicleAge, kind="count", height=4, aspect=0.7)
plt.show()

# Plotting Current Retail Average Price with Target Variable
car_count_by_VehicleAge['MMRCurrentRetailAveragePriceGroups'] = pd.qcut(car_count_by_VehicleAge['MMRCurrentRetailAveragePrice'], q=4, labels=False)
plt.figure(figsize=(10, 6))
sns.catplot(x="IsBadBuy", hue="IsBadBuy", col="MMRCurrentRetailAveragePriceGroups", data=car_count_by_VehicleAge, kind="count", height=4, aspect=0.7)
plt.show()

# Plotting Current Retail Clean Price with Target Variable
car_count_by_VehicleAge['MMRCurrentRetailCleanPriceGroups'] = pd.qcut(car_count_by_VehicleAge['MMRCurrentRetailCleanPrice'], q=4, labels=False)
plt.figure(figsize=(10, 6))
sns.catplot(x="IsBadBuy", hue="IsBadBuy", col="MMRCurrentRetailCleanPriceGroups", data=car_count_by_VehicleAge, kind="count", height=4, aspect=0.7)
plt.show()

# Distribution of Vehicle Cost
Ixos = car_buy_clean_df[car_buy_clean_df['IsBadBuy'] == 0]['VehBCost']
Primadur = car_buy_clean_df[car_buy_clean_df['IsBadBuy'] == 1]['VehBCost']

plt.hist(Ixos, bins=30, color='red', alpha=0.5, label='Not_Bad_buy')
plt.hist(Primadur, bins=30, color='blue', alpha=0.5, label='Bad_buy')
plt.xlabel('Vehicle Cost')
plt.ylabel('Number of vehicles')
plt.title('Distribution of Vehicle Cost')
plt.legend(loc='upper right')
plt.show()

# Correlation Matrix
import seaborn as sns
import matplotlib.pyplot as plt

num_df = car_buy_clean_df.select_dtypes(include=[np.number])
corMatrix = num_df.corr()

plt.figure(figsize=(12, 10))
sns.heatmap(corMatrix, annot=True, cmap='coolwarm', fmt='.2f')
plt.title('Correlation Matrix')
plt.show()
