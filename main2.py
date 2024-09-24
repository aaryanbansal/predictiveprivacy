from sklearn.metrics import mean_squared_error

# Calculate MSE for age before and after applying DP
mse_age = mean_squared_error(df['age'], df['age_dp'])

print(f'Mean Square Error for Age: {mse_age}')
