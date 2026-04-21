import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_absolute_error, mean_squared_error

# --- 1. HELPER FUNCTIONS ---
def calculate_haversine(lat1, lon1, lat2, lon2):
    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
    dlat = lat2 - lat1 
    dlon = lon2 - lon1 
    a = np.sin(dlat/2)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon/2)**2
    c = 2 * np.arcsin(np.sqrt(a)) 
    r = 6371 # Radius of earth in km
    return c * r

def preprocess(df):
    # Feature Engineering: CBD Distance
    cbd_lat, cbd_lon = 1.2830, 103.8513
    df['dist_to_cbd'] = calculate_haversine(df['Latitude'], df['Longitude'], cbd_lat, cbd_lon)
    
    # Feature Engineering: Log Distances
    df['mrt_nearest_distance_log'] = np.log1p(df['mrt_nearest_distance'])
    df['Mall_Nearest_Distance'] = df['Mall_Nearest_Distance'].fillna(df['Mall_Nearest_Distance'].median())
    df['mall_nearest_distance_log'] = np.log1p(df['Mall_Nearest_Distance'])

    # Extract Postal Sector (First 2 digits)
    df['postal_sector'] = df['postal'].astype(str).str.zfill(6).str[:2]

    # Dropping logic-based redundant columns (keeping school names for now)
    redundant_cols = ['Tranc_YearMonth', 'storey_range', 'mid', 'full_flat_type', 'address', 'postal']
    df = df.drop(columns=[c for c in redundant_cols if c in df.columns])

    # Amenity Imputation
    amenity_cols = ['Mall_Within_500m', 'Mall_Within_1km', 'Mall_Within_2km',
                    'Hawker_Within_500m', 'Hawker_Within_1km', 'Hawker_Within_2km']
    df[amenity_cols] = df[amenity_cols].fillna(0)

    # Convert Binary Y/N to 1/0
    binary_cols = ['residential', 'commercial', 'market_hawker', 'multistorey_carpark', 'precinct_pavilion']
    for col in binary_cols:
        if col in df.columns:
            df[col] = df[col].map({'Y': 1, 'N': 0})
            
    return df

# --- 2. DATA LOADING ---
print("Loading and cleaning data...")
# Loading with dtype specified to avoid mixed type warnings
train = pd.read_csv('train.csv', dtype={'postal': str})
test = pd.read_csv('test.csv', dtype={'postal': str})

train_df = preprocess(train)
test_df = preprocess(test)

# --- 3. TARGET ENCODING ---
print("Target encoding postal sectors...")
sector_means = train_df.groupby('postal_sector')['resale_price'].mean()
train_df['postal_sector_price'] = train_df['postal_sector'].map(sector_means)
test_df['postal_sector_price'] = test_df['postal_sector'].map(sector_means)

global_mean = train_df['resale_price'].mean()
test_df['postal_sector_price'] = test_df['postal_sector_price'].fillna(global_mean)

# Drop original string sector
train_df = train_df.drop(columns=['postal_sector'])
test_df = test_df.drop(columns=['postal_sector'])

# --- 4. CATEGORICAL ENCODING (For selected text features) ---
cat_cols = ['town', 'flat_type', 'flat_model', 'planning_area', 'mrt_name']
for col in cat_cols:
    le = LabelEncoder()
    combined = pd.concat([train_df[col], test_df[col]], axis=0).astype(str)
    le.fit(combined)
    train_df[col] = le.transform(train_df[col].astype(str))
    test_df[col] = le.transform(test_df[col].astype(str))

# --- 5. DATA FILTERING & TRAINING ---
# Separate target and features
X = train_df.drop(columns=['id', 'resale_price'])
y = np.log1p(train_df['resale_price']) # Log transformation
X_test_all = test_df.drop(columns=['id'])

# FILTER: Keep only numerical/encoded columns for the model
numerical_features = X.select_dtypes(exclude=['object']).columns.tolist()
X_filtered = X[numerical_features]
X_test_filtered = X_test_all[numerical_features]

X_train, X_val, y_train, y_val = train_test_split(X_filtered, y, test_size=0.15, random_state=42)

print(f"Training on {len(numerical_features)} features. (Strings like 'block' ignored)")
model = xgb.XGBRegressor(
    n_estimators=3000,
    learning_rate=0.02,
    max_depth=8,
    subsample=0.8,
    colsample_bytree=0.8,
    n_jobs=-1,
    random_state=42,
    tree_method='hist',
    early_stopping_rounds=50
)

model.fit(
    X_train, y_train,
    eval_set=[(X_val, y_val)],
    verbose=100
)

# --- 6. EVALUATION ---
val_preds_log = model.predict(X_val)
val_preds = np.expm1(val_preds_log)
y_val_actual = np.expm1(y_val)

rmse = np.sqrt(mean_squared_error(y_val_actual, val_preds))
print(f"\n--- Validation Result ---")
print(f"RMSE: ${rmse:,.2f}")

# --- 7. FINAL SUBMISSION ---
print("\nGenerating submission...")
test_preds_log = model.predict(X_test_filtered)
test_preds = np.expm1(test_preds_log)

submission = pd.DataFrame({'id': test['id'], 'resale_price': test_preds})
submission.to_csv('my_final_optimized_submission.csv', index=False)
print("✅ Done! File 'my_final_optimized_submission.csv' is ready.")