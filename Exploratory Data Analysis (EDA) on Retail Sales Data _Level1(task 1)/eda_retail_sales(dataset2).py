import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Load the dataset
file_path = "D:\My Data\Mugdha jadhav\Desktop\oasis internship\Exploratory Data Analysis (EDA) on Retail Sales Data _Level1(task 1)\dataset 2\menu.csv"  # Update this with the correct path

df = pd.read_csv(file_path)

# Display basic info
df.info()
print(df.head())

# Clean column names
df.columns = df.columns.str.strip().str.replace(' ', '_').str.replace('(', '').str.replace(')', '')

# Check for missing values
print("Missing Values:\n", df.isnull().sum())

# Descriptive Statistics
print("Descriptive Stats:\n", df.describe())

# Identify top categories by avg calories
category_avg_calories = df.groupby('Category')['Calories'].mean().sort_values(ascending=False)
print("Top Categories by Avg Calories:\n", category_avg_calories)

# Find highest and lowest calorie items
max_cal_item = df.loc[df['Calories'].idxmax(), ['Item', 'Calories']]
min_cal_item = df.loc[df['Calories'].idxmin(), ['Item', 'Calories']]
print("Highest Calorie Item:", max_cal_item)
print("Lowest Calorie Item:", min_cal_item)

# Select only numeric columns for correlation
numeric_df = df.select_dtypes(include=[np.number])

# Correlation Analysis
plt.figure(figsize=(10,6))
sns.heatmap(numeric_df.corr(), annot=True, cmap='coolwarm', fmt='.2f')
plt.title('Nutrient Correlation Heatmap')
plt.show()


# Visualization - Category Wise Calories
plt.figure(figsize=(12,6))
sns.barplot(x=category_avg_calories.index, y=category_avg_calories.values, palette='viridis')
plt.xticks(rotation=90)
plt.title('Average Calories per Category')
plt.ylabel('Calories')
plt.show()

# Scatter plot: Calories vs Fat
plt.figure(figsize=(8,6))
sns.scatterplot(data=df, x='Total_Fat', y='Calories', hue='Category', palette='deep')
plt.title('Calories vs Total Fat')
plt.xlabel('Total Fat (g)')
plt.ylabel('Calories')
plt.show()

# Recommendations
print("\nBusiness Recommendations:")
print("1. Promote lower-calorie, high-protein items for health-conscious customers.")
print("2. Reformulate high-sodium and high-fat items to improve nutrition scores.")
print("3. Highlight best-selling healthy options to attract a wider audience.")
