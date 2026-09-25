#!/usr/bin/env python
# coding: utf-8

# # NextHikes IT Solutions

# ### Exploratory Data Analysis of Large-Scale Retail Transaction Data Using Univariate, Bivariate, and Multivariate Statistical Techniques

# In[26]:


# Import Libraries

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

print("Libraries imported successfully!")


# In[ ]:





# In[27]:


# Loaded The Dataset

EDA = pd.read_csv("C:\\Users\\adity\\OneDrive\\Desktop\\NextHikes Project\\EDA Project 3\\retail_large_dataset.csv")
print("Dataset Loaded Successfully")


# In[4]:


EDA.head()


#  Shows the first 5 rows.


# In[5]:


# Understand The Dataset

print("Shape:", EDA.shape)

print("\nColumns:", EDA.columns)


#   EDA.shape → Shows the number of rows and columns.
#   EDA.columns → Shows all column names.


# In[6]:


print("\nData Types:")
print(EDA.dtypes)


#  EDA.dtypes → Shows the data type of each column


# In[7]:


#Check The Missing Value

print("Missing Values:", EDA.isnull().sum())


#  EDA.isnull().sum()

#Checks how many missing/empty values are present in each column.


# In[8]:


#Check The Duplicates Value

print("\nDuplicates:", EDA.duplicated().sum())

#  EDA.duplicated().sum()

#  Counts duplicate rows in the dataset.


# In[9]:


#Convert Date

EDA["order_date"] = pd.to_datetime(EDA["order_date"])

#  pd.to_datetime()

# Converts the order_date column into a proper date format.


# In[10]:


# Basic Statistics

EDA.describe()

# EDA.describe()

# Provides statistical information such as count, mean, standard deviation, minimum, maximum, and quartiles. 


# In[30]:


# Check invalid dates

print("\nInvalid Dates:")
print(EDA["order_date"].isnull().sum())

# Checks how many missing/empty values are present in each column.


# In[31]:


# Check numeric columns

numeric_columns = EDA.select_dtypes(include=np.number).columns

print("\nNumerical Columns:")
print(numeric_columns.tolist())

# Selects only numerical columns.


# In[32]:


#Check categorical columns

categorical_columns = EDA.select_dtypes(include="object").columns

print("\nCategorical Columns:")
print(categorical_columns.tolist())

# Selects categorical/text columns.


# ### Univariate Analysis

# In[11]:


# Numerical Variables 

num = EDA.select_dtypes(include="number")

print("Mean:")
print(num.mean())

print("\nMedian:")
print(num.median())

print("\nStandard Deviation:")
print(num.std())

print("\nMinimum:")
print(num.min())

print("\nMaximum:")
print(num.max())

print("\nSkewness:")
print(num.skew())

print("\nKurtosis:")
print(num.kurtosis())


# Mean → Average value.
# Median → Middle value.
# Std Dev → Shows how spread out the values are.
# Minimum → Smallest value.
# Maximum → Largest value.
# Skewness → Shows whether the distribution is tilted left or right.
# Kurtosis → Shows the presence of extreme values/heavy tails


# In[113]:


# Histograms

num.hist(
    figsize=(12, 10),
    bins=20,
    color="lightblue",
    edgecolor="white"
)

plt.tight_layout()
plt.show()

# Creates a histogram to show how numerical values are distributed.


# In[13]:


# Boxplots / Outliers

plt.figure(figsize=(12, 6))

sns.boxplot(data=num)

plt.xticks(rotation=45)
plt.title("Boxplot of Numerical Variables")

plt.show()

# Shows the distribution, median, spread, and possible outliers of a numerical variable.


# In[14]:


# IQR

Q1 = num.quantile(0.25)
Q3 = num.quantile(0.75)

IQR = Q3 - Q1

outliers = ((num < Q1 - 1.5 * IQR) |
            (num > Q3 + 1.5 * IQR))

print(outliers.sum())

#  These boundaries are used to identify potential outliers


# #### Categorical Variables
# 
# We analyze:
# 
# Product category
# 
# Customer segment
# 
# Payment method
# 
# Return status
# 
# value_counts() counts how many records belong to each category.
# 
# 
# kind="bar" displays those counts as a bar chart
# 

# In[112]:


# Product Category

print(EDA["product_category"].value_counts())

EDA["product_category"].value_counts().plot(
    kind="bar",
    color=["lightblue", "lightgreen", "lightpink", "lavender", "lightyellow"]
)

plt.title("Product Category")
plt.xlabel("Product Category")
plt.ylabel("Count")

plt.xticks(rotation=45)
plt.tight_layout()
plt.show()


# In[111]:


# Customer Segment

print(EDA["customer_segment"].value_counts())

EDA["customer_segment"].value_counts().plot(
    kind="bar",
    color=["lightblue", "lightgreen", "lightpink", "plum", "lightyellow"]
)

plt.title("Customer Segment")
plt.xlabel("Customer Segment")
plt.ylabel("Count")

plt.xticks(rotation=45)
plt.tight_layout()
plt.show()


# In[110]:


# Payment Method

print(EDA["payment_method"].value_counts())

EDA["payment_method"].value_counts().plot(
    kind="bar",
    color=["lightblue", "lightgreen", "lightpink", "plum", "lightyellow"]
)

plt.title("Payment Method")
plt.xlabel("Payment Method")
plt.ylabel("Count")

plt.xticks(rotation=45)
plt.tight_layout()
plt.show()


# In[19]:


# Return Status

print(EDA["return_status"].value_counts())

EDA["return_status"].value_counts().plot(kind="bar")

plt.title("Return Status")
plt.show()


# ### Bivariate Analysis
# 
# Analysis of two variables together.
# Shows the relationship between two numerical variables.
# It helps us see whether one variable changes when another variable changes.
# 
# We have used it for:
# 
# Product Price vs Final Price
# 
# Quantity vs Final Price
# 
# Age vs Final Price
# 
# Delivery Days vs Final Price

# In[20]:


#  Product Price vs Final Price

sns.scatterplot(
    data=EDA,
    x="product_price",
    y="final_price"
)

plt.title("Product Price vs Final Price")
plt.show()


# In[21]:


#  Quantity vs Final Price

sns.scatterplot(
    data=EDA,
    x="quantity",
    y="final_price"
)

plt.title("Quantity vs Final Price")
plt.show()


# In[22]:


# Age vs Final Price

sns.scatterplot(
    data=EDA,
    x="age",
    y="final_price"
)

plt.title("Age vs Final Price")
plt.show()


# In[34]:


# Delivery Days vs Final Price

sns.scatterplot(
    data=EDA,
    x="delivery_days",
    y="final_price"
)

plt.title("delivery_days vs Final Price")
plt.show()


# #### Correlation Matrix Analysis

# In[23]:


corr = num.corr()

print(corr)


# In[109]:


# Correlation Heatmap

plt.figure(figsize=(10, 7))

sns.heatmap(
    corr,
    annot=True,
    cmap="Blues"
)

plt.title("Correlation Heatmap")
plt.show()


# Displays the correlation matrix visually.
# Displays the correlation values inside the heatmap.
# Applies a color scheme to make strong and weak correlations easier to identify.


# #### Numerical vs Categorical Analysis

# In[36]:


# Customer Segment vs Average Final Price

segment_spending = EDA.groupby(
    "customer_segment"
)["final_price"].mean().sort_values(ascending=False)

print("\nAverage Final Price by Customer Segment:")
print(segment_spending)


# Groups customers by segment and calculates the average final price for each segment.


# In[108]:


# Graph

plt.figure(figsize=(8, 5))

segment_spending.plot(
    kind="bar",
    color=["lightblue", "lightgreen", "lightpink", "plum", "lightyellow"]
)

plt.title("Average Final Price by Customer Segment")
plt.xlabel("Customer Segment")
plt.ylabel("Average Final Price")

plt.xticks(rotation=45)
plt.tight_layout()
plt.show()


# In[38]:


# Product Category vs Average Final Price

category_spending = EDA.groupby(
    "product_category"
)["final_price"].mean().sort_values(ascending=False)

print("\nAverage Final Price by Product Category:")
print(category_spending)

# Same idea, but groups the data by product_category.
# It tells us the average final price for each product category.


# In[107]:


# Graph

plt.figure(figsize=(9, 5))

category_spending.plot(
    kind="bar",
    color=["lightblue", "lightgreen", "lightpink", "plum", "lightyellow"]
)

plt.title("Average Final Price by Product Category")
plt.xlabel("Product Category")
plt.ylabel("Average Final Price")

plt.xticks(rotation=45)
plt.tight_layout()
plt.show()


# In[45]:


# Shipping Type vs Delivery Days

shipping_delivery = EDA.groupby(
    "shipping_type"
)["delivery_days"].mean().sort_values()

print("\nAverage Delivery Days by Shipping Type:")
print(shipping_delivery)

# Calculates the average delivery time for each shipping type.
# This helps us compare delivery performance.


# In[46]:


# Graph

shipping_delivery.plot(
    kind="bar",
    figsize=(8, 5)
)

plt.title("Average Delivery Days by Shipping Type")
plt.xlabel("Shipping Type")
plt.ylabel("Average Delivery Days")

plt.xticks(rotation=45)
plt.tight_layout()
plt.show()


# #### Multivariate Analysis

# In[51]:


# CATEGORY × DISCOUNT × FINAL PRICE


EDA["age_group"] = pd.cut(
    EDA["age"],
    bins=[0, 18, 30, 45, 60, 100],
    labels=[
        "Below 18",
        "19-30",
        "31-45",
        "46-60",
        "60+"
    ]
)


# Groups data by product category and calculates:

# Average discount
# Average final price

# This allows us to study multiple factors together.


# In[52]:


age_segment_spending = EDA.groupby(
    ["customer_segment", "age_group"],
    observed=True
)["final_price"].mean().reset_index()

print("\nCustomer Segment × Age Group × Spending:")
print(age_segment_spending)


# Studies average spending based on both:

# Customer segment
# Age group

# This is a multivariate analysis.


# In[53]:


# Graph

plt.figure(figsize=(12, 6))

sns.barplot(
    data=age_segment_spending,
    x="age_group",
    y="final_price",
    hue="customer_segment"
)

plt.title("Customer Segment × Age Group × Average Spending")
plt.xlabel("Age Group")
plt.ylabel("Average Final Price")

plt.tight_layout()
plt.show()


# In[54]:


# Delivery Days × Shipping Type × Return Status

delivery_return = EDA.groupby(
    ["shipping_type", "return_status"]
)["delivery_days"].mean().reset_index()

print("\nShipping Type × Return Status × Delivery Days:")
print(delivery_return)


# Examines delivery time based on:

# Shipping type
# Return status

#This helps identify whether delivery patterns differ between returned and non-returned orders.


# In[55]:


# Graph

plt.figure(figsize=(10, 6))

sns.barplot(
    data=delivery_return,
    x="shipping_type",
    y="delivery_days",
    hue="return_status"
)

plt.title("Shipping Type × Delivery Days × Return Status")
plt.xlabel("Shipping Type")
plt.ylabel("Average Delivery Days")

plt.tight_layout()
plt.show()


# #### Skewness Analysis

# In[56]:


skewness_results = EDA[numeric_columns].skew()

print("\nSkewness:")
print(skewness_results)


# Calculates the skewness of numerical variables.

# Around 0 → Approximately symmetrical
# Positive → Right-skewed
# Negative → Left-skewed


# #### Kurtosis Analysis

# In[57]:


kurtosis_results = EDA[numeric_columns].kurtosis()

print("\nKurtosis:")
print(kurtosis_results)


# Measures the shape of the distribution and the presence of extreme observations.

# High positive kurtosis → More extreme values/heavier tails.
# Around 0 → Similar to a normal distribution.
# Negative kurtosis → Lighter tails.


# #### Revenue Analysis

# In[59]:


# Total Revenue

total_revenue = EDA["final_price"].sum()

print("\nTotal Revenue:")
print(total_revenue)


# Adds all final prices together.
# Purpose: Finds total revenue represented by the transactions.


# In[60]:


# Average Transaction Value

average_transaction = EDA["final_price"].mean()

print("\nAverage Transaction Value:")
print(average_transaction)

# Calculates the average transaction value.


# In[62]:


# Median transaction

median_transaction = EDA["final_price"].median()

print("\nMedian Transaction Value:")
print(median_transaction)

# Finds the middle transaction value when all transactions are arranged in order.


# #### Revenue by Product Category

# In[63]:



revenue_by_category = EDA.groupby(
    "product_category"
)["final_price"].sum().sort_values(ascending=False)

print("\nRevenue by Product Category:")
print(revenue_by_category)

# Calculates the total revenue generated by each product category
# Arranges categories from highest to lowest revenue.


# In[106]:


# Ghaph

plt.figure(figsize=(10, 6))

revenue_by_category.plot(
    kind="bar",
    color=["lightblue", "lightgreen", "lightpink", "plum", "lightyellow"]
)

plt.title("Revenue by Product Category")
plt.xlabel("Product Category")
plt.ylabel("Total Revenue")

plt.xticks(rotation=45)
plt.tight_layout()
plt.show()


# #### Revenue by Customer Segment

# In[66]:


revenue_by_segment = EDA.groupby(
    "customer_segment"
)["final_price"].sum().sort_values(ascending=False)

print("\nRevenue by Customer Segment:")
print(revenue_by_segment)


# Calculates total revenue generated by each customer segment.
# This helps understand which customer groups contribute to revenue.


# In[104]:


plt.figure(figsize=(8, 5))

revenue_by_segment.plot(
    kind="bar",
    color=["skyblue", "orange", "green", "purple", "red"]
)

plt.title("Revenue by Customer Segment")
plt.xlabel("Customer Segment")
plt.ylabel("Total Revenue")

plt.xticks(rotation=45)
plt.tight_layout()
plt.show()


# #### Monthly Revenue Analysis

# In[75]:



EDA["order_date"] = pd.to_datetime(EDA["order_date"])

EDA["month"] = EDA["order_date"].dt.month

monthly_revenue = EDA.groupby(
    "month"
)["final_price"].sum()

print("\nMonthly Revenue:")
print(monthly_revenue)

# Extracts the month from the order date.
# calculates total revenue for each month.


# In[76]:


# Graph

plt.figure(figsize=(10, 5))

monthly_revenue.plot(marker="o")

plt.title("Monthly Revenue")
plt.xlabel("Month")
plt.ylabel("Revenue")

plt.xticks(range(1, 13))

plt.tight_layout()
plt.show()


# #### Yearly Revenue

# In[77]:




EDA["year"] = EDA["order_date"].dt.year

yearly_revenue = EDA.groupby(
    "year"
)["final_price"].sum()

print("\nYearly Revenue:")
print(yearly_revenue)

# Extracts the year from the order date.
# Then groupby() calculates total revenue for each year.


# In[102]:


# Graph

plt.figure(figsize=(8, 5))

yearly_revenue.plot(
    kind="bar",
    color=plt.cm.Set3(range(len(yearly_revenue)))
)

plt.title("Yearly Revenue")
plt.xlabel("Year")
plt.ylabel("Revenue")

plt.tight_layout()
plt.show()


# #### Discount Analysis

# In[79]:


discount_analysis = EDA.groupby(
    "discount_percentage"
)["final_price"].mean()

print("\nAverage Final Price by Discount:")
print(discount_analysis)

# Calculates the average final price for each discount percentage.
# The line graph helps visualize how final price changes across different discount levels.


# In[101]:


# Graph

plt.figure(figsize=(10, 5))

plt.plot(
    discount_analysis.index,
    discount_analysis.values,
    marker="o",
    color="skyblue"
)

plt.title("Discount Percentage vs Average Final Price")
plt.xlabel("Discount Percentage")
plt.ylabel("Average Final Price")

plt.tight_layout()
plt.show()


# #### Quantity Analysis

# In[81]:


quantity_analysis = EDA.groupby(
    "quantity"
)["final_price"].mean()

print("\nAverage Final Price by Quantity:")
print(quantity_analysis)

# Calculates the average final price for each quantity purchased.
# This helps examine the relationship between quantity and transaction value


# In[100]:


# Graph

plt.figure(figsize=(8, 5))

plt.plot(
    quantity_analysis.index,
    quantity_analysis.values,
    marker="o",
    color="black"
)

plt.title("Quantity vs Average Final Price")
plt.xlabel("Quantity")
plt.ylabel("Average Final Price")

plt.tight_layout()
plt.show()


# #### Payment Method Analysis

# In[83]:


payment_analysis = EDA.groupby(
    "payment_method"
)["final_price"].agg(
    ["count", "sum", "mean"]
)

print("\nPayment Method Analysis:")
print(payment_analysis)

# Calculates three things for each payment method:

# count → Number of transactions
# sum → Total revenue
# mean → Average transaction value

# This gives a complete overview of payment methods.


# In[97]:


# Graph

payment_analysis["sum"].plot(
    kind="bar",
    figsize=(9, 5),
    color=plt.cm.Set3(range(len(payment_analysis)))
)

plt.title("Revenue by Payment Method")
plt.xlabel("Payment Method")
plt.ylabel("Total Revenue")

plt.xticks(rotation=45)
plt.tight_layout()
plt.show()


# #### Customer Age Analysis

# In[96]:


plt.figure(figsize=(9, 5))

plt.hist(
    EDA["age"],
    bins=20,
    color="skyblue",
    edgecolor="black"
)

plt.title("Customer Age Distribution")
plt.xlabel("Age")
plt.ylabel("Number of Customers")

plt.tight_layout()
plt.show()

# Creates a histogram showing the distribution of customer ages.
# It helps identify which age groups are most common.


# #### Product Subcategory Analysis

# In[86]:


subcategory_counts = EDA[
    "product_subcategory"
].value_counts().head(10)

print("\nTop 10 Product Subcategories:")
print(subcategory_counts)

# Counts transactions for each product subcategory and shows the top 10.
#This helps identify the most frequently purchased subcategories.


# In[94]:


# Graph

subcategory_counts.plot(
    kind="bar",
    figsize=(10, 5),
    color=plt.cm.Set3(range(len(subcategory_counts)))
)

plt.title("Top 10 Product Subcategories")
plt.xlabel("Product Subcategory")
plt.ylabel("Transactions")

plt.xticks(rotation=45)
plt.tight_layout()
plt.show()


# In[ ]:





# In[ ]:





# In[ ]:




