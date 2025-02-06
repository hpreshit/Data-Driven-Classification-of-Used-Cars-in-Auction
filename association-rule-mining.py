import pandas as pd
from mlxtend.frequent_patterns import apriori, association_rules
import matplotlib.pyplot as plt
import seaborn as sns

# Load the CSV data into pandas DataFrame
df = pd.read_pickle('rules_df.pkl')

# Convert the data into a one-hot encoded matrix
# Assuming the dataset contains transaction-like data
# and 'IsBadBuy' is a column indicating whether it's a bad buy (1) or good buy (0)
one_hot_encoded_df = pd.get_dummies(df)

# Ensure all values are binary (0 or 1)
one_hot_encoded_df = one_hot_encoded_df.apply(lambda x: (x > 0).astype(int)).astype(bool)

# Define function to get rules for Bad Buys
def get_bad_buy_rules(df):
    # Apply Apriori algorithm to find frequent itemsets with min support of 1%
    frequent_itemsets = apriori(df, min_support=0.01, use_colnames=True)
    # Generate association rules with a minimum confidence of 85%
    rules = association_rules(frequent_itemsets, metric="confidence", min_threshold=0.85)
    # Filter for Bad Buy rules (rhs="IsBadBuy=1")
    bad_buy_rules = rules[rules['consequents'].apply(lambda x: 'IsBadBuy=1' in str(x))]
    
    return bad_buy_rules

# Define function to get rules for Good Buys
def get_good_buy_rules(df):
    # Apply Apriori algorithm to find frequent itemsets with min support of 30%
    frequent_itemsets = apriori(df, min_support=0.3, use_colnames=True)
    # Generate association rules with a minimum confidence of 85%
    rules = association_rules(frequent_itemsets, metric="confidence", min_threshold=0.85)
    # Filter for Good Buy rules (rhs="IsBadBuy=0")
    good_buy_rules = rules[rules['consequents'].apply(lambda x: 'IsBadBuy=0' in str(x))]
    
    return good_buy_rules

# Get rules for Bad Buys
bad_buy_rules = get_bad_buy_rules(one_hot_encoded_df)

# Sort Bad Buy rules by lift
bad_buy_rules_sorted = bad_buy_rules.sort_values(by='lift', ascending=False)

# Display top 5 Bad Buy rules
print("Top 5 Bad Buy Rules:")
print(bad_buy_rules_sorted.head())

# Get rules for Good Buys
good_buy_rules = get_good_buy_rules(one_hot_encoded_df)

# Sort Good Buy rules by lift
good_buy_rules_sorted = good_buy_rules.sort_values(by='lift', ascending=False)

# Display top 5 Good Buy rules
print("\nTop 5 Good Buy Rules:")
print(good_buy_rules_sorted.head())

# Plotting Bad Buy rules
plt.figure(figsize=(10, 6))
sns.barplot(x='lift', y='antecedents', data=bad_buy_rules_sorted.head(5), color='blue')
plt.title("Top 5 Bad Buy Rules by Lift")
plt.xlabel("Lift")
plt.ylabel("Antecedents")
plt.show()

# Plotting Good Buy rules
plt.figure(figsize=(10, 6))
sns.barplot(x='lift', y='antecedents', data=good_buy_rules_sorted.head(5), color='green')
plt.title("Top 5 Good Buy Rules by Lift")
plt.xlabel("Lift")
plt.ylabel("Antecedents")
plt.show()
