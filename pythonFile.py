# %%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# %%
df = pd.read_csv(
    '/Users/surya/Documents/LPU/Semester-4/INT375 - Python/Data Sets/appleAppData.csv')
pd.set_option('display.max_columns', None)
df.head()

# %%
# to get information about the data
df.info()

# %%

# Exploring the data

# %%
# data type of App_Id
print('The data type of App_Id:', df['App_Id'].dtype)

# checking for missing values in App_Id
print('Missing values in App_Id:', df['App_Id'].isnull().sum())

# checking duplicates in App_Id
print('Number of duplicates in App_Id:', df['App_Id'].duplicated().sum())

# %%
sns.set_style("whitegrid")
palette = sns.color_palette(
    "tab10", n_colors=len(df['Primary_Genre'].unique()))

plt.figure(figsize=(10, 6))
plot = sns.countplot(
    data=df,
    y='Primary_Genre',
    hue='Primary_Genre',  # Category for color coding
    order=df['Primary_Genre'].value_counts().index,
    palette=palette,
    legend=False  # Disabling redudant legend
)

# Adding value annotations to the bars
for p in plot.patches:
    plot.annotate(f'{int(p.get_width())}',
                  (p.get_width(), p.get_y() + p.get_height() / 2),
                  ha='left', va='center', fontsize=12, color='black')

# Customising the plot
plt.title('Primary Genre', fontsize=20, color='darkblue', fontfamily='Arial')
plt.xlabel('Count', fontsize=16)
plt.ylabel('Primary Genre', fontsize=16)
plt.xticks(fontsize=14)
plt.yticks(fontsize=14)
plt.grid(axis='x', linestyle='--', alpha=0.7)

plt.show()

# %%
labels = ['4+', '17+', '12+', '9+', 'Not yet rated']
values = [980971, 124127, 89570, 35698, 10]

explode = [0.2, 0, 0, 0, 0]

plt.figure(figsize=(8, 6))
plt.pie(values, labels=labels, autopct='%1.1f%%', explode=explode, startangle=140,
        colors=plt.cm.Paired.colors, wedgeprops={'edgecolor': 'black'})

plt.title('Content Rating Distribution', fontsize=25, color='black')

plt.show()

# %%
# Price of Apps

# Data type of Price
print("The Data type of Price: ", df['Price'].dtype, "\n")

# Check for missing values
print('Number of missing values in Price:', df['Price'].isnull().sum(), "\n")

# Duplicate values in Price
print('Number of duplicates in Price:', df['Price'].duplicated().sum(), "\n")

# Mean (average) of Price
print('Mean of Price:', df['Price'].mean(), "\n")

# Median of Price
print('Median of Price:', df['Price'].median(), "\n")

# Mode of Price
print('Mode of Price:', df['Price'].mode(), "\n")

# Minimum value of Price
print('Minimum value of Price:', df['Price'].min(), "\n")

# Maximum value of Price
print('Maximum value of Price:', df['Price'].max())

# %%
# Rows where duplicates exist
df[df['Price'].duplicated()].head()

# %%
top_10_developers = df['Developer'].value_counts().head(10)


colors = plt.cm.tab10.colors

plt.figure(figsize=(9, 7))
plt.pie(
    top_10_developers,
    labels=top_10_developers.index,
    autopct='%1.1f%%',
    startangle=140,
    colors=colors,
    wedgeprops={'edgecolor': 'black', 'linewidth': 1.2},
    textprops={'fontsize': 12, 'color': 'black'}
)

plt.title('Top 10 Developers', fontsize=25, color='black')

plt.show()

# %%
# Average user rating
df['Average_User_Rating'].head()

# %%
plt.figure(figsize=(8, 6))
sns.displot(df['Average_User_Rating'], bins=50, kde=True, color='royalblue')

plt.title('Distribution of Average User Ratings', fontsize=20, color='black')
plt.xlabel('Average User Rating', fontsize=14)
plt.ylabel('Frequency', fontsize=14)

plt.show()

# %%
# Visualising missing values from the dataset
sns.heatmap(df.isnull())

# %%
# Correlation of numeric columns
plt.figure(figsize=(16, 10))

numeric_cols = ['Size_Bytes',
                'Price',
                'DeveloperId',
                'Average_User_Rating',
                'Reviews',
                'Current_Version_Score',
                'Current_Version_Reviews']

sns.heatmap(df[numeric_cols].corr(), annot=True)

# %%
# Checking unique values

print("Content_Rating: ", df['Content_Rating'].unique(), "\n")
print()

print("Size_Bytes: ", df['Size_Bytes'].unique(), "\n")
print()

print("Required_IOS_Version: ", df['Required_IOS_Version'].unique(), "\n")
print()

print("Released: ", df['Released'].unique(), "\n")
print()

print("Currency: ", df['Currency'].unique())


# %%
# Category of user rating apps

# Creating a separate column for type
df['Type'] = np.where(df['Free'] == True, "Free", "Paid")

plt.figure(figsize=(16, 6))
sns.barplot(x='Content_Rating', y='Average_User_Rating', data=df, hue='Type')

# %%
appsData = df.groupby("Primary_Genre")[
    'Average_User_Rating'].mean().sort_values(ascending=False)
appsData = appsData.head(10)

plt.figure(figsize=(10, 6))
plt.bar(appsData.index, appsData.values, color='royalblue')

plt.title('Top 10 Categories Based on Rating')
plt.xlabel('App Category')
plt.xticks(rotation=90)
plt.ylabel('Rating')

plt.show()

# %%
# Price vs Size
plt.figure(figsize=(10, 6))
sns.scatterplot(x=df['Price'], y=df['Size_Bytes'], hue=df['Type'])
plt.title('Price vs Size in Bytes')
plt.xlabel('Price')
plt.ylabel('Size in Bytes')
plt.show()

# %%
# Year vs Apps Released
df['Released'] = pd.to_datetime(df['Released'])  # convert to datetime format
df['Year_Release'] = df['Released'].dt.strftime('%Y')

yearRelease = df.groupby(['Year_Release']).size().reset_index(name='Count')
yearRelease.plot(x='Year_Release', y='Count', kind='line', marker='o')

plt.title('Count Trend over a Year')
plt.show()

# %%
# Boxplot for Average_User_Rating across Content_Rating
plt.figure(figsize=(10, 6))
sns.boxplot(x='Content_Rating', y='Average_User_Rating',
            data=df, palette='Set2')
plt.title('Average User Rating by Content Rating')
plt.xlabel('Content Rating')
plt.ylabel('Average User Rating')
plt.xticks(rotation=45)
plt.grid(True, axis='y', linestyle='--', alpha=0.7)
plt.show()


# %%
# Creating price bins
bins = [0, 0.99, 2.99, 5.99, 9.99, 19.99, df['Price'].max()]
labels = ['Free to $0.99', '$1 to $2.99', '$3 to $5.99',
          '$6 to $9.99', '$10 to $19.99', '$20+']
df['Price_Range'] = pd.cut(df['Price'], bins=bins, labels=labels)

plt.figure(figsize=(10, 6))
sns.countplot(data=df[df['Type'] == 'Paid'], x='Price_Range', palette='pastel')
plt.title('Distribution of Paid Apps by Price Range')
plt.xlabel('Price Range')
plt.ylabel('Number of Apps')
plt.xticks(rotation=45)
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.show()


# %%
# iOS version mode
ios_version_counts = df['Required_IOS_Version'].value_counts().head(10)

plt.figure(figsize=(10, 5))
sns.barplot(x=ios_version_counts.index,
            y=ios_version_counts.values, palette='muted')
plt.title('Top 10 Required iOS Versions')
plt.xlabel('iOS Version')
plt.ylabel('Number of Apps')
plt.show()


# %%
df['Month_Release'] = df['Released'].dt.month_name()
monthly_release = df['Month_Release'].value_counts().reindex([
    'January', 'February', 'March', 'April', 'May', 'June',
    'July', 'August', 'September', 'October', 'November', 'December'
])

plt.figure(figsize=(12, 6))
monthly_release.plot(kind='bar', color='lightseagreen')
plt.title('Number of Apps Released by Month')
plt.ylabel('Number of Apps')
plt.xticks(rotation=45)
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.show()
