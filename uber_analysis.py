import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from geopy.distance import geodesic

# Load dataset (adjust the path as needed)
df = pd.read_csv('./data/uber.csv')

# 1. Dataset structure and dimensions
print("Dataset shape (rows, columns):", df.shape)
print("\nFirst 5 rows:")
print(df.head())

# 2. Data types and variable descriptions
print("\nData types of each column:")
print(df.dtypes)

print("\nSummary statistics for numeric variables:")
print(df.describe())

print("\nSummary statistics for categorical/object variables:")
print(df.describe(include=['object']))

# 3. Initial data quality assessment
print("\nMissing values per column:")
print(df.isnull().sum())

print("\nNumber of duplicate rows:")
print(df.duplicated().sum())

# Optionally: check unique values for categorical variables
for col in df.select_dtypes(include=['object']).columns:
    print(f"\nUnique values in column '{col}':")
    print(df[col].value_counts().head(10))  # show top 10 frequent values


# ---------------------------
# 4. Handle missing values and clean the data
# ---------------------------

# Drop rows with missing critical fields
df = df.dropna(subset=[
    'fare_amount', 
    'pickup_longitude', 'pickup_latitude',
    'dropoff_longitude', 'dropoff_latitude',
    'pickup_datetime'
])

# Convert pickup_datetime to datetime format
df['pickup_datetime'] = pd.to_datetime(df['pickup_datetime'], errors='coerce')
df = df.dropna(subset=['pickup_datetime'])  # drop if conversion failed

# Remove negative or zero fares
df = df[df['fare_amount'] > 0] 

# Remove rides with 0 or unusually high passenger count (if exists)
if 'passenger_count' in df.columns:
    df = df[(df['passenger_count'] > 0) & (df['passenger_count'] <= 6)]

# Remove duplicates
df = df.drop_duplicates()

# ---------------------------
# 5. Save cleaned dataset as CSV
# ---------------------------
df.to_csv('./data/uber_cleaned.csv', index=False)
print("\n✅ Cleaned dataset exported to './data/uber_cleaned.csv'")

# ---------------------------
# 6. Final dataset overview
# ---------------------------
print("\nAfter cleaning:")
print("Shape:", df.shape)
print("Remaining missing values:")
print(df.isnull().sum())



# ---------------------------
# 8. Exploratory Data Analysis (EDA)
# ---------------------------

print("\n🔍 Exploratory Data Analysis")

# ----------------------------
# 1. Descriptive Statistics
# ----------------------------
print("\n📊 Descriptive Statistics:")
print("Mean fare: $", round(df['fare_amount'].mean(), 2))
print("Median fare: $", round(df['fare_amount'].median(), 2))
print("Mode fare: $", df['fare_amount'].mode()[0])
print("Standard deviation: $", round(df['fare_amount'].std(), 2))

print("\n📐 Quartiles:")
print(df['fare_amount'].quantile([0.25, 0.5, 0.75]))

print("\n🎯 Fare range:")
print("Min fare: $", df['fare_amount'].min())
print("Max fare: $", df['fare_amount'].max())

# ----------------------------
# 2. Outlier Detection (IQR)
# ----------------------------
Q1 = df['fare_amount'].quantile(0.25)
Q3 = df['fare_amount'].quantile(0.75)
IQR = Q3 - Q1
lower_bound = Q1 - 1.5 * IQR
upper_bound = Q3 + 1.5 * IQR

outliers = df[(df['fare_amount'] < lower_bound) | (df['fare_amount'] > upper_bound)]
print("\n🚨 Number of outliers in fare_amount:", len(outliers))

# ----------------------------
# 3. Histogram of Fare Amount
# ----------------------------
plt.figure(figsize=(10, 5))
sns.histplot(df['fare_amount'], bins=100, kde=True, color='skyblue')
plt.title('Fare Amount Distribution')
plt.xlabel('Fare Amount ($)')
plt.ylabel('Frequency')
plt.xlim(0, 100)
plt.grid(True)
plt.tight_layout()
plt.show()

# ----------------------------
# 4. Compute Distance (km)
# ----------------------------

# Remove rows with invalid lat/lon
df = df[
    (df['pickup_latitude'].between(-90, 90)) &
    (df['dropoff_latitude'].between(-90, 90)) &
    (df['pickup_longitude'].between(-180, 180)) &
    (df['dropoff_longitude'].between(-180, 180))
]

from geopy.distance import geodesic

def haversine_distance(row):
    pickup = (row['pickup_latitude'], row['pickup_longitude'])
    dropoff = (row['dropoff_latitude'], row['dropoff_longitude'])
    return geodesic(pickup, dropoff).km

df['distance_km'] = df.apply(haversine_distance, axis=1)
df = df[df['distance_km'] > 0]  # remove zero-distance trips

# ----------------------------
# 5. Extract Hour from Datetime
# ----------------------------
df['hour'] = df['pickup_datetime'].dt.hour

# ----------------------------
# 6. Fare vs Distance (Scatter)
# ----------------------------
plt.figure(figsize=(10, 6))
sns.scatterplot(x='distance_km', y='fare_amount', data=df, alpha=0.4, color='purple')
plt.title('Fare Amount vs. Distance Traveled')
plt.xlabel('Distance (km)')
plt.ylabel('Fare Amount ($)')
plt.xlim(0, 30)
plt.ylim(0, 100)
plt.grid(True)
plt.tight_layout()
plt.show()

# ----------------------------
# 7. Fare by Hour (Boxplot)
# ----------------------------
plt.figure(figsize=(10, 6))
sns.boxplot(x='hour', y='fare_amount', data=df, palette='viridis')
plt.title('Fare Amount by Hour of Day')
plt.xlabel('Hour of Day')
plt.ylabel('Fare Amount ($)')
plt.grid(True)
plt.tight_layout()
plt.show()

# ----------------------------
# 8. Correlation Heatmap
# ----------------------------
corr = df[['fare_amount', 'distance_km', 'hour']].corr()

plt.figure(figsize=(8, 5))
sns.heatmap(corr, annot=True, cmap='coolwarm', fmt=".2f")
plt.title("Correlation Between Key Variables")
plt.tight_layout()
plt.show()

# ----------------------------
# 9. Extended Correlation Heatmap including passenger_count
# ----------------------------
if 'passenger_count' in df.columns:
    corr_extended = df[['fare_amount', 'distance_km', 'hour', 'passenger_count']].corr()
    
    plt.figure(figsize=(8, 6))
    sns.heatmap(corr_extended, annot=True, cmap='coolwarm', fmt=".2f")
    plt.title("Correlation Matrix Including Passenger Count")
    plt.tight_layout()
    plt.show()
    
    corr_fare_passenger = corr_extended.loc['fare_amount', 'passenger_count']
    print(f"Correlation between fare amount and passenger count: {corr_fare_passenger:.2f}")
else:
    print("Column 'passenger_count' not found in dataset.")




# 3. Feature Engineering

# a. Create new features from 'pickup_datetime'
df['hour'] = df['pickup_datetime'].dt.hour
df['day'] = df['pickup_datetime'].dt.day
df['month'] = df['pickup_datetime'].dt.month
df['day_of_week'] = df['pickup_datetime'].dt.dayofweek  # Monday=0, Sunday=6
df['year'] = df['pickup_datetime'].dt.year

# Create a readable day name column
df['day_name'] = df['pickup_datetime'].dt.day_name()

# Map day names to numbers for sorting
day_order_map = {
    'Monday': 1,
    'Tuesday': 2,
    'Wednesday': 3,
    'Thursday': 4,
    'Friday': 5,
    'Saturday': 6,
    'Sunday': 7
}
df['day_order'] = df['day_name'].map(day_order_map)

# Add Season column based on month
def assign_season(month):
    if month in [12, 1, 2]:
        return 'Winter'
    elif month in [3, 4, 5]:
        return 'Spring'
    elif month in [6, 7, 8]:
        return 'Summer'
    else:
        return 'Fall'
df['season'] = df['month'].apply(assign_season)

# Add Weekend vs Weekday column
df['is_weekend'] = df['day_of_week'].apply(lambda x: 'Weekend' if x >= 5 else 'Weekday')

# Make these categorical
df['season'] = df['season'].astype('category')
df['is_weekend'] = df['is_weekend'].astype('category')

# Ensure datetime types
df['pickup_datetime'] = pd.to_datetime(df['pickup_datetime'])

# Simulate ride duration using distance_km and average speed of 30 km/h
df['ride_duration'] = (df['distance_km'] / 30) * 60

# Remove invalid or zero durations
df = df[df['ride_duration'] > 0]
df['ride_duration'] = df['ride_duration'].round(2)

# Define peak/off-peak based on hour
def peak_offpeak(hour):
    if 7 <= hour <= 9 or 16 <= hour <= 19:
        return 'Peak'
    else:
        return 'Off-Peak'
df['peak_offpeak'] = df['hour'].apply(peak_offpeak)

# Encode categorical variables
categorical_cols = ['peak_offpeak', 'day_name']
for col in categorical_cols:
    df[col] = df[col].astype('category')

# Save the enhanced dataset for Power BI import
df.to_csv('./data/uber_enhanced.csv', index=False)
print("\n✅ Enhanced dataset with new features exported to 'uber_enhanced.csv'")
