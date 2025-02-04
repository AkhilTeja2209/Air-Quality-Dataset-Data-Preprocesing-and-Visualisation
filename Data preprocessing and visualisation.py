import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from ucimlrepo import fetch_ucirepo

air_quality = fetch_ucirepo(id=360)
df = air_quality.data.features

# Data cleaning
df.replace(-200, pd.NA, inplace=True)
df['datetime'] = pd.to_datetime(df['Date'] + ' ' + df['Time'])
df = df.drop(['Date', 'Time'], axis=1)

# Set style
sns.set(style='whitegrid', palette='muted')

# 1. Daily Pollution Patterns
plt.figure(figsize=(14, 6))
sns.lineplot(data=df, x='datetime', y='CO(GT)', 
             hue=df['datetime'].dt.month, palette='viridis')
plt.title('CO Concentration Trends with Monthly Variation')
plt.xlabel('Date')
plt.ylabel('CO (mg/m³)')
plt.legend(title='Month')
plt.show()

# 2. Sensor vs Reference Comparison
fig, ax = plt.subplots(1, 2, figsize=(16, 6))
sns.scatterplot(data=df, x='PT08.S1(CO)', y='CO(GT)', ax=ax[0])
ax[0].set_title('Metal Oxide Sensor vs Reference CO')
sns.regplot(data=df, x='NOx(GT)', y='NO2(GT)', ax=ax[1])
ax[1].set_title('NOx vs NO2 Correlation')
plt.tight_layout()
plt.show()

# 3. Multi-Pollutant Distribution
plt.figure(figsize=(12, 6))
sns.boxplot(data=df.melt(id_vars='datetime', 
                        value_vars=['CO(GT)', 'NOx(GT)', 'C6H6(GT)']), 
            x='variable', y='value')
plt.title('Pollutant Concentration Distributions')
plt.xlabel('Pollutant')
plt.ylabel('Concentration (mg/m³)')
plt.xticks(rotation=45)
plt.show()

# 4. Advanced Missing Data Visualization
missing = df.isna().mean().sort_values(ascending=False)*100
plt.figure(figsize=(10, 6))
sns.heatmap(df.isna().T, cmap='YlGnBu', 
            cbar_kws={'label': 'Missing Data'})
plt.title('Missing Data Pattern Analysis')
plt.xlabel('Observation Index')
plt.ylabel('Features')
plt.show()
