import pandas as pd
from sklearn.model_selection import train_test_split

from src.features.transformations import (
    driver_distance_to_pickup,
    driver_historical_completed_bookings,
    hour_of_day,
)
from src.utils.store import AssignmentStore


def main():
    store = AssignmentStore()

    dataset = store.get_processed("dataset.csv")

    # Print the columns of the dataset
        
    dataset = apply_feature_engineering(dataset)

    final_dataset = aggregate_order_data(dataset)
    store.put_processed("transformed_dataset.csv", final_dataset)

def aggregate_order_data(df: pd.DataFrame) -> pd.DataFrame:
    # Group by order_id and aggregate necessary columns
    df_agg = df.groupby('order_id').agg({
        'driver_id': 'first',  # Assuming the same driver for each order
        'trip_distance': 'mean',  # Average trip distance if there are multiple rows
        'pickup_latitude': 'first',  # Assuming same pickup location per order
        'pickup_longitude': 'first',
        'driver_gps_accuracy': 'first',
        'completed_booking_count':'mean',
        'event_hour': 'mean',  # Average event hour
        'driver_distance': 'mean',  # Average driver distance to pickup
        'is_completed': 'last'  # Assuming the last status indicates completion
    }).reset_index()

    return df_agg

def apply_feature_engineering(df: pd.DataFrame) -> pd.DataFrame:
    return (
        df.pipe(driver_distance_to_pickup)
        .pipe(hour_of_day)
        .pipe(driver_historical_completed_bookings)
    )


if __name__ == "__main__":
    main()
