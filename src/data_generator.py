import pandas as pd
import numpy as np
import datetime

def generate_data(num_days=365):
    """
    Generates synthetic sales data with trend, seasonality, and noise.
    """
    start_date = datetime.date(2023, 1, 1)
    dates = [start_date + datetime.timedelta(days=i) for i in range(num_days)]
    
    # Base trend (linear growth)
    trend = np.linspace(100, 200, num_days)
    
    # Seasonality (weekly)
    seasonality = 20 * np.sin(np.linspace(0, 2 * np.pi * num_days / 7, num_days))
    
    # Noise (random)
    noise = np.random.normal(0, 5, num_days)
    
    # Combine components
    sales = trend + seasonality + noise
    sales = np.maximum(sales, 0) # Ensure non-negative sales
    
    # Create DataFrame
    df = pd.DataFrame({'date': dates, 'sales_units': sales})
    
    # Add other features (simulated)
    df['price'] = np.random.uniform(50, 150, num_days)
    df['economic_index'] = np.random.uniform(0, 1, num_days)
    df['weather_impact'] = np.random.choice([0, 1], num_days)
    df['competitor_price_index'] = np.random.uniform(0.8, 1.2, num_days)
    df['promotion_applied'] = np.random.choice([0, 1], num_days)
    df['holiday_season'] = np.random.choice([0, 1], num_days)
    
    return df

if __name__ == "__main__":
    df = generate_data(2000) # Ensure sufficient data for convergence
    df.to_csv('demand_forecasting_dataset.csv', index=False)
    print("Synthetic data generated and saved to demand_forecasting_dataset.csv")
