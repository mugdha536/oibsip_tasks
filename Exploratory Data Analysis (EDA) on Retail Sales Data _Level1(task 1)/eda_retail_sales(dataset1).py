import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
 
#loading the dataset
file_path = "D:\My Data\Mugdha jadhav\Desktop\oasis internship\Exploratory Data Analysis (EDA) on Retail Sales Data _Level1(task 1)\data set 1\customer_shopping_data.csv"
df = pd.read_csv(file_path)

#converting invoice date to datetime format
df['invoice_date']= pd.to_datetime(df['invoice_date'],errors = 'coerce',dayfirst = True)

#checking missing values
print("missing values:")
print(df.isnull().sum())

#summary statistics
print("\n Summary Stats:")
print(df.describe())

#sales analysis
category_sales = df.groupby("category")["price"].sum().sort_values(ascending = False)
plt.figure(figsize=(10,5))
category_sales.plot(kind="bar",color="skyblue")
plt.title("Total sales per category")
plt.xlabel("Category")
plt.ylabel("Total Sales")
plt.xticks(rotation=45)
plt.show()

#analysis of above- clothing category has the highest sales 

#gender-wise spending
gender_spending = df.groupby("gender")["price"].sum()
plt.figure(figsize=(6,4))
gender_spending.plot(kind="bar",color=["blue","pink"])
plt.title("total spending by gender")
plt.ylabel("Total sales")
plt.xticks(rotation=45)
plt.show()  # female spend more 


#payment metgod preference 
plt.figure(figsize=(6,4))
df["payment_method"].value_counts().plot(kind="pie",autopct="%1.1f%%",colors=["lightblue","lightgreen","lightcoral"])
plt.title("Payment method distribution")
plt.ylabel("")
plt.show()  #42% in chash payment 

#monthly sales trend
df['month']=df['invoice_date'].dt.month
monthly_sales = df.groupby("month")["price"].sum()
plt.figure(figsize=(8,5))
monthly_sales.plot(kind="line",marker='o',color="red")
plt.title("monthly sales trend")
plt.xlabel("month")
plt.ylabel("totalsales")
plt.xticks(range(1,13))
plt.show()



print(df.dtypes)

# Selecting only numerical columns for correlation

numeric_df = df.select_dtypes(include=['number'])

# Print selected numeric columns to verify
print("Numeric Columns:\n", numeric_df.head())

# Heatmap for Correlation Analysis
plt.figure(figsize=(8, 5))
sns.heatmap(numeric_df.corr(), annot=True, cmap="coolwarm", fmt=".2f")
plt.title("Correlation Heatmap")
plt.show()

# Shopping Mall Analysis
mall_sales = df.groupby("shopping_mall")["price"].sum().sort_values(ascending=False).head(10)
plt.figure(figsize=(10, 5))
mall_sales.plot(kind="bar", color="purple")
plt.title("Top 10 Shopping Malls by Sales")
plt.xticks(rotation=45)
plt.show()


# Recommendations
print("\n### Business Recommendations ###")
print("- Stock more of best-selling product categories.")
print("- Adjust marketing strategies based on gender spending patterns.")
print("- Promote digital payment methods based on their popularity.")
print("- Optimize inventory and promotions based on peak shopping months.")



