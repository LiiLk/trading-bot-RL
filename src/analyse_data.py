import pandas as pd
import matplotlib.pyplot as plt

# Load the data
df = pd.read_csv('eurusd_data.csv')
df['timestamp'] = pd.to_datetime(df['timestamp'])
df.set_index('timestamp', inplace=True)

# Plot the price
plt.figure(figsize=(12, 6))
plt.plot(df['close'])
plt.title('EURUSD Price')
plt.xlabel('Date')
plt.ylabel('Price')
plt.savefig('eurusd_price.png')
plt.close()

# Calculate daily returns
df['returns'] = df['close'].pct_change()

# Plot the distribution of returns
plt.figure(figsize=(12, 6))
df['returns'].hist(bins=50)
plt.title('Distribution of Returns')
plt.xlabel('Returns')
plt.ylabel('Frequency')
plt.savefig('returns_distribution.png')
plt.close()

# Print some statistics
print(f"Data range: {df.index.min()} to {df.index.max()}")
print(f"Number of candles: {len(df)}")
print(f"Mean return: {df['returns'].mean():.6f}")
print(f"Standard deviation of returns: {df['returns'].std():.6f}")
print(f"Skewness of returns: {df['returns'].skew():.6f}")
print(f"Kurtosis of returns: {df['returns'].kurtosis():.6f}")