import pandas as pd
import numpy as np
from datetime import datetime, timedelta

# Load the PaySim CSV
df = pd.read_csv("fraud_detection.csv")

# Step → transaction_datetime (assuming step = 1 is 1 day)
start_date = datetime(2020, 1, 1)
df['transaction_datetime'] = df['step'].apply(lambda x: start_date + timedelta(days=int(x)))

# Add transaction_id
df['transaction_id'] = df.index + 1

# Rename and map columns
df['client_id'] = df['nameOrig']
df['account_id'] = df['nameOrig']
df['account_type'] = "checking"
df['office_id'] = np.random.randint(100, 999, df.shape[0])
df['transaction_type'] = df['type']
df['amount'] = df['amount']
df['running_balance'] = df['newbalanceOrig']
df['currency_code'] = "USD"
df['day_of_week'] = df['transaction_datetime'].dt.day_name()
df['hour_of_day'] = df['transaction_datetime'].dt.hour
df['fraud'] = df['isFraud']

# Sort by account and datetime for rolling calculations
df = df.sort_values(by=['account_id', 'transaction_datetime'])

# 1. Time since last transaction (minutes)
df['time_since_last_txn_mins'] = df.groupby('account_id')['transaction_datetime']\
    .diff().dt.total_seconds().div(60)

# 2. Transaction frequency in the last 24 hours
df['txn_frequency_last_24h'] = df.groupby('account_id').rolling('1D', on='transaction_datetime')['transaction_id']\
    .count().reset_index(level=0, drop=True) - 1  # subtract current txn

# 3. Average amount in last 30 days
df['avg_amount_last_30d'] = df.groupby('account_id').rolling('30D', on='transaction_datetime')['amount']\
    .mean().reset_index(level=0, drop=True)

# 4. Amount deviation score (Z-score)
rolling_std = df.groupby('account_id').rolling('30D', on='transaction_datetime')['amount']\
    .std().reset_index(level=0, drop=True)
df['amount_dev_score'] = (df['amount'] - df['avg_amount_last_30d']) / rolling_std

# Replace infinite values with NaN
df.replace([np.inf, -np.inf], np.nan, inplace=True)

# Final column order
final_columns = [
    'transaction_id','client_id','account_id','account_type','office_id','transaction_type','amount',
    'running_balance','currency_code','transaction_datetime','day_of_week','hour_of_day',
    'time_since_last_txn_mins','txn_frequency_last_24h','avg_amount_last_30d','amount_dev_score','fraud'
]

# Export final CSV
df_final = df[final_columns]
df_final.to_csv("paysim_transformed.csv", index=False)

print("✅ paysim_transformed.csv created successfully with engineered features filled.")
