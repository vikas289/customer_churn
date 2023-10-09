#!/usr/bin/env python
# coding: utf-8

# In[2]:


pip install --upgrade tensorflow


# In[3]:


# Uploading the Packages
get_ipython().system('pip install xgboost')
get_ipython().system('pip install pandas')
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler, MinMaxScaler, Normalizer, OneHotEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.metrics import classification_report
from sklearn.ensemble import RandomForestClassifier
from sklearn.cluster import KMeans
from sklearn.metrics import accuracy_score
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.decomposition import PCA
from xgboost import XGBClassifier
import tensorflow as tf
from tensorflow.keras.layers import Dense, BatchNormalization, Dropout
from tensorflow.keras.models import Sequential
from scipy.stats import chi2_contingency


# In[4]:


#Uploading the Dataset
df_churn = pd.read_csv('telecom_customer_churn.csv')
df_churn


# In[5]:


# repositioning Customer Status (target variable) at the end
df_churn["Customer Status"] = df_churn.pop("Customer Status")
df_churn.head()


# In[6]:


df_churn.info()


# In[7]:


df_churn.rename({"Customer ID": "CustomerID"}, axis=1, inplace=True)


# In[8]:


# Dealing with missing values 
# Replacing missing values
df_churn[['Internet Type', 'Online Security', 'Online Backup', 'Device Protection Plan', 'Premium Tech Support', 'Streaming TV', 'Streaming Movies','Streaming Music','Unlimited Data']] = df_churn[['Internet Type', 'Online Security', 'Online Backup', 'Device Protection Plan', 
           'Premium Tech Support', 'Streaming TV', 'Streaming Movies','Streaming Music','Unlimited Data']].replace(np.nan,'No Internet')
df_churn['Avg Monthly GB Download'] = df_churn['Avg Monthly GB Download'].replace(np.nan,0.0)
df_churn['Multiple Lines'] = df_churn['Multiple Lines'].replace(np.nan,'NO phone Service')
df_churn['Avg Monthly Long Distance Charges'] = df_churn['Avg Monthly Long Distance Charges'].replace(np.nan,0.0)


# In[9]:


# Removing irregularities with Monthly Charge 
df_churn = df_churn[df_churn['Monthly Charge'] >= 0]


# In[10]:


df_churn.info()


# In[11]:


# Explaratory Data Analysis
df_churn.describe()


# In[12]:


# Key observations:

# Average Age: 46.5
# Average Number of Referrals: 1.9
# Median Number of Referrals: 0 (this indicates that at least 50% of the customers do not give referrals)
# Average Tenure in Months: 32 (2.66 years)


# In[13]:


# Analysing descriptive statistics for customers who have churned
df_churn[df_churn["Customer Status"] == "Churned"].describe()


# In[14]:


# Key observations:

# Average Age: 49.7
# Average Number of Referrals: 0.5
# Median Number of Referrals: 0
# Average Tenure in Months: 17 (1.4 years)
    
# Conclusions:
# Average Age seems to be slightly higher for churned customers
# Average Number of Referrals is significantly lower for churned customers. This seems logical as the reason for churn is most often that the product was not satisfactory, hence the customer would not recommend the product to others
# Average Tenure in Months is significantly lower. Again, expected.


# In[15]:


df_churn['Customer Status'].value_counts()


# In[16]:


df_churn['Offer'].value_counts()


# In[17]:


df_churn['Offer'].groupby(df_churn['Customer Status']).value_counts()


# In[18]:


#Most reasons are come from Attitude of support person and there are some competitors
df_churn['Churn Reason'].groupby(df_churn['Churn Category']).value_counts()


# In[19]:


# Group the data and calculate the counts
grouped_counts = df_churn.groupby(['Churn Category', 'Churn Reason']).size().reset_index(name='Count')

# Create a bar plot using Seaborn
plt.figure(figsize=(10, 6))
sns.barplot(x='Churn Reason', y='Count', hue='Churn Category', data=grouped_counts)
plt.xticks(rotation=45, ha='right')
plt.xlabel('Churn Reason')
plt.ylabel('Count')
plt.title('Churn Reasons by Category')
plt.tight_layout()

# Show the plot
plt.show()


# In[20]:


#use pearson method to find correlation
df_correlation = df_churn.corr(method ='pearson')
df_correlation

# Create a heatmap
fig, ax = plt.subplots(figsize=(15, 15))
heatmap = sns.heatmap(df_correlation, annot=True, fmt=".2f", linewidths=.3, ax=ax, cmap="YlGnBu")

# Set the title
plt.title("Correlation Heatmap")

# Show the plot
plt.show()


# In[21]:


df_churn.drop(columns=["CustomerID", "Churn Category", "Churn Reason"], inplace=True)


# In[22]:


df_churn["Customer Status"].value_counts()


# In[23]:


# Select the columns you want to keep
selected_columns = [
    'Gender', 'Married', 'Offer', 'Phone Service', 'Multiple Lines',
    'Internet Service', 'Internet Type', 'Online Security', 'Online Backup',
    'Device Protection Plan', 'Premium Tech Support', 'Streaming TV',
    'Streaming Movies', 'Streaming Music', 'Unlimited Data', 'Contract',
    'Paperless Billing', 'Payment Method', 'Customer Status'
]

# Create a new DataFrame with the selected columns
df = df_churn[selected_columns]


# In[24]:


df


# In[25]:


print("Unique values of categorical columns\n")
for col in df.columns:
  if not df[col].dtype in ["int64", "float64"]: 
    print(f"{col}:")
    print(f"\tUniques: {df[col].unique()}")
    print(f"\tNumber of uniques: {df[col].nunique()}\n")


# In[26]:


#Logistic Regression 
# Select the columns you want to use as features and target
selected_features = [
    'Gender', 'Married', 'Offer', 'Phone Service', 'Multiple Lines',
    'Internet Service', 'Internet Type', 'Online Security', 'Online Backup',
    'Device Protection Plan', 'Premium Tech Support', 'Streaming TV',
    'Streaming Movies', 'Streaming Music', 'Unlimited Data', 'Contract',
    'Paperless Billing', 'Payment Method'
]
target_column = 'Customer Status'

model = LogisticRegression(max_iter=10000)

# Create a subset of the DataFrame with selected features and target
subset_data = df[selected_features + [target_column]]

# Split the data into features (X) and target (y)
X = subset_data.drop(target_column, axis=1)
y = subset_data[target_column]

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)

# Define preprocessing steps for categorical and numeric features
categorical_features = ['Gender', 'Married', 'Offer', 'Phone Service', 'Multiple Lines',
                        'Internet Service', 'Internet Type', 'Online Security', 'Online Backup',
                        'Device Protection Plan', 'Premium Tech Support', 'Streaming TV',
                        'Streaming Movies', 'Streaming Music', 'Unlimited Data', 'Contract',
                        'Paperless Billing', 'Payment Method']

categorical_transformer = Pipeline(steps=[
    ('onehot', OneHotEncoder(handle_unknown='ignore'))
])

preprocessor = ColumnTransformer(
    transformers=[
        ('cat', categorical_transformer, categorical_features)
    ])

# Combine preprocessing and logistic regression into a single pipeline
model = Pipeline(steps=[('preprocessor', preprocessor),
                      ('classifier', LogisticRegression())])

# Train the model
model.fit(X_train, y_train)

# Make predictions on the test set
y_pred = model.predict(X_test)

# Calculate accuracy
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy:.2f}")

# Print classification report
print(classification_report(y_test, y_pred))


# In[27]:


from sklearn.metrics import confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

# Generate the confusion matrix
conf_matrix = confusion_matrix(y_test, y_pred)

# Define class labels
class_labels = ["Churned", "Joined", "Stayed"]

# Display the confusion matrix as a heatmap with class labels
plt.figure(figsize=(8, 6))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', cbar=False,
            xticklabels=class_labels, yticklabels=class_labels)
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix')
plt.show()


# # Random forest classifier 

# In[28]:


# Select features and target
selected_features = [
    'Gender', 'Married', 'Offer', 'Phone Service', 'Multiple Lines',
    'Internet Service', 'Internet Type', 'Online Security', 'Online Backup',
    'Device Protection Plan', 'Premium Tech Support', 'Streaming TV',
    'Streaming Movies', 'Streaming Music', 'Unlimited Data', 'Contract',
    'Paperless Billing', 'Payment Method'
]
target_column = 'Customer Status'

X = df_churn[selected_features]
y = df_churn[target_column]

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)

# Encode categorical features
encoder = OneHotEncoder(handle_unknown='ignore', sparse=False)
X_train_encoded = encoder.fit_transform(X_train)
X_test_encoded = encoder.transform(X_test)

# Initialize Random Forest classifier
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)

# Train the model
rf_model.fit(X_train_encoded, y_train)

# Make predictions on the test set
y_pred = rf_model.predict(X_test_encoded)

# Calculate accuracy
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy:.2f}")

# Print classification report
print(classification_report(y_test, y_pred))

# Generate the confusion matrix
conf_matrix = confusion_matrix(y_test, y_pred)

# Display the confusion matrix as a heatmap with class labels
class_labels = ["Churned", "Joined", "Stayed"]
plt.figure(figsize=(8, 6))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', cbar=False,
            xticklabels=class_labels, yticklabels=class_labels)
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix')
plt.show()


# In[ ]:




