import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Task 1: Load and Explore the Dataset
try:
    # Load dataset 
    df = sns.load_dataset("iris")
    
    # Display first few rows
    print("First few rows of the dataset:")
    print(df.head())
    
    # Check data types and missing values
    print("\nDataset Info:")
    print(df.info())
    print("\nMissing values per column:")
    print(df.isnull().sum())
    
    # Clean dataset (if necessary)
    df.dropna(inplace=True) 
    print("\nAfter cleaning, missing values per column:")
    print(df.isnull().sum())
    
except FileNotFoundError:
    print("Error: File not found. Please check the file path.")
except Exception as e:
    print(f"An error occurred: {e}")

# Task 2: Basic Data Analysis
print("\nBasic Statistics:")
print(df.describe())


species_group = df.groupby("species").mean()
print("\nMean of numerical columns grouped by species:")
print(species_group)

# Task 3: Data Visualization
sns.set_style("whitegrid")

df["index"] = range(len(df))  
plt.figure(figsize=(8,5))
sns.lineplot(x="index", y="sepal_length", data=df)
plt.title("Trend of Sepal Length")
plt.xlabel("Index")
plt.ylabel("Sepal Length")
plt.show()

# 2. Bar chart 
plt.figure(figsize=(8,5))
sns.barplot(x="species", y="sepal_length", data=df)
plt.title("Average Sepal Length per Species")
plt.xlabel("Species")
plt.ylabel("Average Sepal Length")
plt.show()

# 3. Histogram 
plt.figure(figsize=(8,5))
sns.histplot(df["petal_length"], bins=20, kde=True)
plt.title("Distribution of Petal Length")
plt.xlabel("Petal Length")
plt.ylabel("Count")
plt.show()

# 4. Scatter plot 
plt.figure(figsize=(8,5))
sns.scatterplot(x="sepal_length", y="petal_length", hue="species", data=df)
plt.title("Sepal Length vs. Petal Length")
plt.xlabel("Sepal Length")
plt.ylabel("Petal Length")
plt.legend(title="Species")
plt.show()
