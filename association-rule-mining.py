import pandas as pd
from mlxtend.frequent_patterns import apriori, association_rules
import matplotlib.pyplot as plt

# Importing the csv data and matrix data
rules_mat = pd.read_pickle("rules_df.pkl")

# Sample the dataset (e.g., 10% of the data)
rules_mat_sampled = rules_mat.sample(frac=0.1, random_state=1)

# Convert the DataFrame to boolean type
rules_mat_sampled = rules_mat_sampled.astype(bool)

# Rules for Bad Buys
frequent_itemsets = apriori(rules_mat_sampled, min_support=0.01, use_colnames=True)
rules_Bad_Buy = association_rules(frequent_itemsets, metric="confidence", min_threshold=0.85)
rules_Bad_Buy = rules_Bad_Buy[rules_Bad_Buy['consequents'] == frozenset(['IsBadBuy=1'])]

# Plotting the rules
plt.scatter(rules_Bad_Buy['support'], rules_Bad_Buy['confidence'], alpha=0.5)
plt.xlabel('Support')
plt.ylabel('Confidence')
plt.title('Rules for Bad Buys')
plt.show()

# Display best lift - five rules
rules_BB_sorted = rules_Bad_Buy.sort_values(by='lift', ascending=False).head(5)
print(rules_BB_sorted)

# Rules for Good Buys
frequent_itemsets = apriori(rules_mat_sampled, min_support=0.3, use_colnames=True)
rules_Good_Buy = association_rules(frequent_itemsets, metric="confidence", min_threshold=0.85)
rules_Good_Buy = rules_Good_Buy[rules_Good_Buy['consequents'] == frozenset(['IsBadBuy=0'])]

# Plotting the rules
plt.scatter(rules_Good_Buy['support'], rules_Good_Buy['confidence'], alpha=0.5)
plt.xlabel('Support')
plt.ylabel('Confidence')
plt.title('Rules for Good Buys')
plt.show()

# Display best lift - five rules
rules_GB_sorted = rules_Good_Buy.sort_values(by='lift', ascending=False).head(5)
print(rules_GB_sorted)
