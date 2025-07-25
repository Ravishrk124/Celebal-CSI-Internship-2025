# -*- coding: utf-8 -*-
"""Untitled63.ipynb

Automatically generated by Colab.

Original file is located at
    https://colab.research.google.com/drive/1ISZD3UKdE4NktiIJpisGhzOV35ao1tUv
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.impute import SimpleImputer

sns.set(style="whitegrid")
plt.rcParams["figure.figsize"] = (12, 6)

df = pd.read_csv('/content/train.csv')
print("Shape of dataset:", df.shape)
df.head()

# Visualize missing values
missing = df.isnull().sum()
missing = missing[missing > 0].sort_values(ascending=False)
sns.barplot(x=missing.values, y=missing.index, palette='viridis')
plt.title("Missing Values per Column")
plt.show()

# Drop columns with too many missing values
df = df.drop(['Alley', 'PoolQC', 'Fence', 'MiscFeature'], axis=1)

# Numeric columns – fill with median
num_cols = df.select_dtypes(include=[np.number]).columns
df[num_cols] = df[num_cols].fillna(df[num_cols].median())

# Categorical columns – fill with mode
cat_cols = df.select_dtypes(include=['object']).columns
df[cat_cols] = df[cat_cols].fillna(df[cat_cols].mode().iloc[0])

# Label encode ordinal columns
ordinal_cols = ['ExterQual', 'ExterCond', 'BsmtQual', 'BsmtCond',
                'HeatingQC', 'KitchenQual', 'FireplaceQu', 'GarageQual', 'GarageCond']

le = LabelEncoder()
for col in ordinal_cols:
    if col in df.columns:
        df[col] = le.fit_transform(df[col])

# One-hot encode remaining categorical variables
df = pd.get_dummies(df, drop_first=True)
print("Dataset shape after encoding:", df.shape)

# New features
df['TotalBathrooms'] = (df['FullBath'] + 0.5 * df['HalfBath'] +
                         df['BsmtFullBath'] + 0.5 * df['BsmtHalfBath'])

df['HouseAge'] = df['YrSold'] - df['YearBuilt']
df['Remodeled'] = (df['YearBuilt'] != df['YearRemodAdd']).astype(int)

scaler = StandardScaler()
scale_cols = ['GrLivArea', 'GarageArea', 'TotalBathrooms', 'HouseAge']
df[scale_cols] = scaler.fit_transform(df[scale_cols])

# Log-transform SalePrice
df['SalePrice'] = np.log1p(df['SalePrice'])
sns.histplot(df['SalePrice'], kde=True, color='blue')
plt.title("Log-Transformed SalePrice")
plt.show()

print("Final dataset shape:", df.shape)
df.head()