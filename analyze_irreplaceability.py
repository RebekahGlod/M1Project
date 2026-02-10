import pandas as pd
import matplotlib.pyplot as plt

# Load the dataset
file_path = 'Alternative CPA Pathways Survey_December 31, 2025_09.45.csv'
try:
    df = pd.read_csv(file_path)
except FileNotFoundError:
    print(f"Error: File not found at {file_path}")
    exit(1)

# Drop the second row (index 0) which contains the question text
df = df.drop(0)

# Define the columns of interest and their mapping
columns_mapping = {
    'Q24_1': 'CPA Exam Prep',
    'Q24_2': 'Networking (Peers/Alumni)',
    'Q24_3': 'Faculty Interaction',
    'Q24_4': 'Technical Skills',
    'Q24_5': 'Soft Skills',
    'Q24_6': 'Recruiting/Internships'
}

# Filter for the relevant columns
relevant_columns = list(columns_mapping.keys())
df_subset = df[relevant_columns].copy()

# Convert columns to numeric, coercing errors to NaN
for col in relevant_columns:
    df_subset[col] = pd.to_numeric(df_subset[col], errors='coerce')

# Calculate the mean rank for each feature
mean_ranks = df_subset.mean()

# Rename the index using the mapping
mean_ranks = mean_ranks.rename(index=columns_mapping)

# Sort the features by mean rank in ascending order (lowest mean = most essential)
sorted_ranks = mean_ranks.sort_values(ascending=True)

# Print the calculated means
print("Average Rankings (Lower is More Essential/Harder to Replace):")
print(sorted_ranks)

# Insights
most_essential = sorted_ranks.index[0]
technical_score = sorted_ranks['Technical Skills']
networking_score = sorted_ranks['Networking (Peers/Alumni)']
recruiting_score = sorted_ranks['Recruiting/Internships']

print("\n--- Insights ---")
print(f"1. Most Essential Feature: {most_essential}")

print(f"2. Comparison: Technical Skills ({technical_score:.2f}) vs Networking/Recruiting")
print(f"   - Networking (Peers/Alumni): {networking_score:.2f}")
print(f"   - Recruiting/Internships: {recruiting_score:.2f}")

comparison_insight = ""
if technical_score < min(networking_score, recruiting_score):
     comparison_insight = "Students value Technical Skills more than Networking and Recruiting."
elif technical_score > max(networking_score, recruiting_score):
     comparison_insight = "Students value Networking and Recruiting more than Technical Skills."
else:
     comparison_insight = "Students value Technical Skills somewhere in between Networking and Recruiting."

print(f"   -> {comparison_insight}")

# Save insights to file
with open('insights.txt', 'w') as f:
    f.write("--- Insights ---\n")
    f.write(f"1. Most Essential Feature: {most_essential}\n")
    f.write(f"2. Comparison: Technical Skills ({technical_score:.2f}) vs Networking/Recruiting\n")
    f.write(f"   - Networking (Peers/Alumni): {networking_score:.2f}\n")
    f.write(f"   - Recruiting/Internships: {recruiting_score:.2f}\n")
    f.write(f"   -> {comparison_insight}\n")

# Visualization
plt.figure(figsize=(10, 6))
# Invert y-axis to have the top ranked item at the top visually if using barh,
# but since we sorted ascending (lowest value first), the first item will be at the bottom by default in barh.
# So we should invert the order for plotting to have the "best" (lowest score) at the top.
sorted_ranks_plot = sorted_ranks.iloc[::-1]

bars = plt.barh(sorted_ranks_plot.index, sorted_ranks_plot.values, color='skyblue')
plt.xlabel('Average Rank (Lower is More Essential)')
plt.title('Program Irreplaceability: What is Hardest to Replace via Work Experience?')
plt.grid(axis='x', linestyle='--', alpha=0.7)

# Add value labels to the bars
for bar in bars:
    width = bar.get_width()
    plt.text(width + 0.05, bar.get_y() + bar.get_height()/2, f'{width:.2f}',
             va='center', color='black')

plt.tight_layout()
plt.savefig('program_irreplaceability.png')
print("\nChart saved as 'program_irreplaceability.png'")
