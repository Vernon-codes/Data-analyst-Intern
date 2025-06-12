import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler # Added for numerical feature scaling

# Load dataset
data_set = pd.read_csv("ecommerce_furniture_dataset_2024.csv")

# Initial info
print(data_set.head())
print("\nMissing values before processing:")
print(data_set.isnull().sum())
total_rows = len(data_set)
print(f"Total rows: {total_rows}")

# Check missing percentage in 'originalPrice'
per_missing_originalPrice = (data_set['originalPrice'].isnull().sum() / total_rows) * 100
print(f"Missing % in 'originalPrice': {per_missing_originalPrice:.2f}%")

# Convert 'price' to numeric after removing '$'
# Do this early so originalPrice can be used for discount calculation if it exists
data_set['price'] = pd.to_numeric(
    data_set['price'].astype(str).str.replace('$', '', regex=False), # Convert to string first to handle potential non-string types
    errors='coerce'
)

# Handle 'originalPrice' - make it numeric first for proper evaluation
# Convert 'originalPrice' to numeric, coerce errors will turn non-convertible values to NaN
data_set['originalPrice'] = pd.to_numeric(
    data_set['originalPrice'].astype(str).str.replace('$', '', regex=False),
    errors='coerce'
)

# Decision: If originalPrice has too many missing values, drop it. Otherwise, impute.
if per_missing_originalPrice > 50:
    print("Dropping 'originalPrice' due to high missing values")
    data_set.drop('originalPrice', axis=1, inplace=True)
    data_set['discount_percentage'] = 0 # If originalPrice is dropped, no discount calculation
else:
    # If keeping originalPrice, impute its missing values with the median
    data_set['originalPrice'].fillna(data_set['originalPrice'].median(), inplace=True)
    # Create discount percentage after originalPrice is numerical and imputed
    data_set['discount_percentage'] = np.where(
        (data_set['originalPrice'] > 0) & (data_set['price'] < data_set['originalPrice']), # Ensure valid calculation
        ((data_set['originalPrice'] - data_set['price']) / data_set['originalPrice']) * 100,
        0
    ).round(2)


# Interpolate numeric columns to fill any remaining NaNs (like 'price' if any)
# This handles 'price' and 'sold' if they had any NaNs after initial conversion
data_set.interpolate(inplace=True)

# Fill remaining missing values in other columns (categorical/text) with 'notag'
# This should be done AFTER numerical imputation and calculations, and for columns that are still objects
for col in data_set.columns:
    if data_set[col].dtype == 'object':
        data_set[col].fillna('notag', inplace=True)

print("\nMissing values after initial processing and imputation:")
print(data_set.isnull().sum())
data_set.info()

print("\nDescriptive statistics for 'price' and 'sold':")
print(data_set[['price', 'sold']].describe())

# Exploratory Data Analysis (EDA)
print("\nEDA")

upper_bound_sold = data_set['sold'].quantile(0.99)
sns.histplot(data_set[data_set['sold'] < upper_bound_sold]['sold'], kde=True)
plt.title('Distribution of Furniture Items Sold (excluding top 1%)')
plt.show()

# Apply log1p transformation to the target variable 'sold'
# This is crucial for models when the target is highly skewed
data_set['sold_log'] = np.log1p(data_set['sold'])
sns.histplot(data_set['sold_log'], kde=True)
plt.title('Distribution of Log-Transformed Furniture Items Sold')
plt.show()
print()

print("Feature Engineering")

# TF-IDF Vectorizer on 'productTitle'
# Ensure 'productTitle' is string type before TF-IDF
data_set['productTitle'] = data_set['productTitle'].astype(str)
tfidf = TfidfVectorizer(max_features=100)
productTitle_tfidf = tfidf.fit_transform(data_set['productTitle'])

productTitle_tfidf_df = pd.DataFrame(
    productTitle_tfidf.toarray(),
    columns=['tfidf_' + col for col in tfidf.get_feature_names_out()]
)

# Combine TF-IDF features with original dataset
df = pd.concat([data_set.reset_index(drop=True), productTitle_tfidf_df.reset_index(drop=True)], axis=1)

# Drop original 'productTitle' column as it's been vectorized
df.drop('productTitle', axis=1, inplace=True)

print("\nDataFrame head after feature engineering:")
print(df.head())

# Prepare features and target variable
# We will use 'sold_log' as the target for training
X = df.drop(['sold', 'sold_log'], axis=1, errors='ignore')
y = df['sold_log'] # Train on the log-transformed target

# Convert categorical features to numeric with one-hot encoding
X = pd.get_dummies(X, drop_first=True)

# Scaling numerical features
# Identify numerical columns for scaling, excluding those created by one-hot encoding
numerical_cols = X.select_dtypes(include=np.number).columns.tolist()
# Exclude binary (0/1) columns that might have been created by get_dummies if they are not truly continuous
numerical_cols_to_scale = [col for col in numerical_cols if not (X[col].isin([0, 1]).all() and X[col].nunique() == 2)]

scaler = StandardScaler()
X[numerical_cols_to_scale] = scaler.fit_transform(X[numerical_cols_to_scale])


# Split into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize models
lr_model = LinearRegression()
rf_model = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1) # Use all cores for RF

# Train models
print("\nTraining models...")
lr_model.fit(X_train, y_train)
rf_model.fit(X_train, y_train)
print("Models trained.")

# Predict on log-transformed target
y_pred_lr_log = lr_model.predict(X_test)
y_pred_rf_log = rf_model.predict(X_test)

# Inverse transform predictions to original scale
y_pred_lr = np.expm1(y_pred_lr_log)
y_pred_rf = np.expm1(y_pred_rf_log)

# Ensure predictions are non-negative
y_pred_lr[y_pred_lr < 0] = 0
y_pred_rf[y_pred_rf < 0] = 0

# Evaluate on original scale
print("\nLinear Regression Performance (on original scale):")
print("MSE:", mean_squared_error(np.expm1(y_test), y_pred_lr)) # Compare actual original values
print("R² Score:", r2_score(np.expm1(y_test), y_pred_lr))

print("\nRandom Forest Performance (on original scale):")
print("MSE:", mean_squared_error(np.expm1(y_test), y_pred_rf)) # Compare actual original values
print("R² Score:", r2_score(np.expm1(y_test), y_pred_rf))