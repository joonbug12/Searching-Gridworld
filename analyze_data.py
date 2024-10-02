import pandas as pd
import matplotlib.pyplot as plt

# Load the CSV results
df = pd.read_csv('a_star_comparison_results.csv')

# Display basic statistics
print(df.describe())

# Filter out any rows with errors
df_clean = df[df['Runtime_Seconds'] != 'Error']

# Convert 'Runtime_Seconds' and 'Expanded_Nodes' to numeric
df_clean['Runtime_Seconds'] = pd.to_numeric(df_clean['Runtime_Seconds'])
df_clean['Expanded_Nodes'] = pd.to_numeric(df_clean['Expanded_Nodes'])

# Group by Algorithm and calculate mean runtime and expanded nodes
grouped = df_clean.groupby('Algorithm').agg({
    'Runtime_Seconds': ['mean', 'std'],
    'Expanded_Nodes': ['mean', 'std']
})

print(grouped)

# Plot average runtime per algorithm
plt.figure(figsize=(10, 6))
df_clean.groupby('Algorithm')['Runtime_Seconds'].mean().plot(kind='bar', yerr=df_clean.groupby('Algorithm')['Runtime_Seconds'].std())
plt.ylabel('Average Runtime (Seconds)')
plt.title('Average Runtime per A* Algorithm')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# Plot average expanded nodes per algorithm
plt.figure(figsize=(10, 6))
df_clean.groupby('Algorithm')['Expanded_Nodes'].mean().plot(kind='bar', yerr=df_clean.groupby('Algorithm')['Expanded_Nodes'].std(), color='orange')
plt.ylabel('Average Number of Expanded Nodes')
plt.title('Average Expanded Nodes per A* Algorithm')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()
