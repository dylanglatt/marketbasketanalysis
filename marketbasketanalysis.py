import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import seaborn as sns
from csv import reader
from mlxtend.preprocessing import TransactionEncoder
from mlxtend.frequent_patterns import apriori, association_rules

# Load the data
groceries = []
with open('/Users/dylanglatt/Desktop/market basket analysis/groceries.csv', 'r') as read_obj:
    csv_reader = reader(read_obj)
    for row in csv_reader:
        groceries.append(row)

# Convert transactions to a one-hot encoded DataFrame
encoder = TransactionEncoder()
transactions = encoder.fit(groceries).transform(groceries)
transactions = transactions.astype('int')
df = pd.DataFrame(transactions, columns=encoder.columns_)

# Input support and length thresholds
min_support = float(input("Enter the minimum support threshold (e.g., 0.02 for 2%): "))
min_length = int(input("Enter the minimum length of itemsets: "))

# Find frequent itemsets
frequent_itemsets = apriori(df, min_support=min_support, use_colnames=True)
frequent_itemsets['length'] = frequent_itemsets['itemsets'].apply(lambda x: len(x))

# Find itemsets with the specified length and support
length_filtered_itemsets = frequent_itemsets[
    (frequent_itemsets['length'] == min_length) &
    (frequent_itemsets['support'] >= min_support)
]
print(f"\nItemsets of length {min_length} with support >= {min_support}:\n")
print(length_filtered_itemsets)

# Find top 10 association rules with the specified support, sorted by confidence
rules = association_rules(frequent_itemsets, metric='support', min_threshold=min_support)
top_rules = rules.sort_values(by='confidence', ascending=False).head(10)
print(f"\nTop 10 association rules with support >= {min_support}, sorted by confidence:\n")
print(top_rules)

# Find association rules with the specified support and lift > 1
filtered_rules = rules[(rules['support'] >= min_support) & (rules['lift'] > 1.0)]
print(f"\nAssociation rules with support >= {min_support} and lift > 1:\n")
print(filtered_rules)