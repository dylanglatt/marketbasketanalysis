import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import seaborn as sns
from csv import reader
from mlxtend.preprocessing import TransactionEncoder
from mlxtend.frequent_patterns import apriori, association_rules
import networkx as nx

# Load data
def load_grocery_data(filepath):
    groceries = []
    with open(filepath, 'r') as file:
        csv_reader = reader(file)
        for row in csv_reader:
            groceries.append(row)
    return groceries

# Encode transactions into a one-hot dataframe
def encode_transactions(groceries):
    encoder = TransactionEncoder()
    transactions = encoder.fit(groceries).transform(groceries)
    # Convert to boolean type for better performance in Apriori
    return pd.DataFrame(transactions, columns=encoder.columns_).astype(bool)

# Get frequent itemsets
def get_frequent_itemsets(df, min_support, min_length):
    frequent_itemsets = apriori(df, min_support=min_support, use_colnames=True)
    frequent_itemsets['length'] = frequent_itemsets['itemsets'].apply(lambda x: len(x))
    filtered_itemsets = frequent_itemsets[
        (frequent_itemsets['length'] >= min_length) &
        (frequent_itemsets['support'] >= min_support)
    ]
    return filtered_itemsets

# Find association rules 
def get_association_rules(frequent_itemsets, min_support):
    if frequent_itemsets.empty:
        print("\nNo frequent itemsets found with the given parameters. Try lowering the support or itemset length.\n")
        return pd.DataFrame(), pd.DataFrame()  # Return empty DataFrames if no frequent itemsets
    
    rules = association_rules(frequent_itemsets, metric='support', min_threshold=min_support, support_only=True)
    top_rules = rules.sort_values(by='support', ascending=False).head(10)
    filtered_rules = rules[(rules['support'] >= min_support)]
    return top_rules, filtered_rules

# Plot bar chart of top association rules by support
def plot_bar_chart(top_rules):
    if top_rules.empty:
        print("No rules to plot.")
        return
    plt.figure(figsize=(10, 6))
    plt.barh(top_rules.index, top_rules['support'], color='skyblue')
    plt.xlabel('Support')
    plt.ylabel('Rule Index')
    plt.title('Top 10 Association Rules by Support')
    plt.gca().invert_yaxis()  # Invert y-axis to show highest support at the top
    plt.show()

# Plot network graph of association rules
def plot_network_graph(rules):
    if rules.empty:
        print("No rules to visualize.")
        return
    G = nx.DiGraph()

    # Add edges
    for _, rule in rules.iterrows():
        lhs = ', '.join(list(rule['antecedents']))
        rhs = ', '.join(list(rule['consequents']))
        G.add_edge(lhs, rhs, weight=rule['lift'])

    plt.figure(figsize=(10, 8))
    pos = nx.spring_layout(G)
    nx.draw(G, pos, with_labels=True, node_color='lightblue', node_size=3000,
            font_size=10, edge_color='gray', width=[d['weight'] for _, _, d in G.edges(data=True)])
    plt.title('Association Rule Network Graph')
    plt.show()

# Main function
def market_basket_analysis():
    # Load and encode data
    groceries = load_grocery_data('/Users/dylanglatt/Desktop/marketbasketanalysis/groceries.csv')
    df = encode_transactions(groceries)

    # Get user inputs
    min_support = float(input("Enter the minimum support threshold (e.g., 0.02 for 2%): "))
    min_length = int(input("Enter the minimum length of itemsets: "))

    # Get frequent itemsets
    itemsets = get_frequent_itemsets(df, min_support, min_length)
    if itemsets.empty:
        print("No frequent itemsets found. Please try different support or length values.")
        return
    
    print(f"\nFrequent Itemsets (length >= {min_length}):\n", itemsets)

    # Get association rules
    top_rules, filtered_rules = get_association_rules(itemsets, min_support)
    if top_rules.empty or filtered_rules.empty:
        print("No association rules found. Please try different parameters.")
        return
    
    print(f"\nTop 10 Association Rules (by support):\n", top_rules)
    print(f"\nFiltered Rules (Lift > 1):\n", filtered_rules)

    # Visualizations
    plot_bar_chart(top_rules)
    plot_network_graph(filtered_rules)

# Run the analysis
if __name__ == "__main__":
    market_basket_analysis()
