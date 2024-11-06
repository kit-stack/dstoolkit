import pandas as pd
from haversine import haversine

from src.utils.time import robust_hour_of_iso_date

from src.utils.store import AssignmentStore
store = AssignmentStore()
dataset = store.get_processed("dataset.csv")

def driver_distance_to_pickup(df: pd.DataFrame) -> pd.DataFrame:
    df["driver_distance"] = df.apply(
        lambda r: haversine(
            (r["driver_latitude"], r["driver_longitude"]),
            (r["pickup_latitude"], r["pickup_longitude"]),
        ),
        axis=1,
    )
    return df


def hour_of_day(df: pd.DataFrame) -> pd.DataFrame:
    df["event_hour"] = df["event_timestamp"].apply(robust_hour_of_iso_date)
    return df

def driver_historical_completed_bookings(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate the cumulative number of completed bookings for each driver up to each event's timestamp.
    This avoids future data leakage and performs the merge directly in one step.
    """
 
    # Ensure the event_timestamp is parsed in datetime format, allowing mixed formats
    df['event_timestamp'] = pd.to_datetime(df['event_timestamp'], errors='coerce')

    # Check for any parsing errors
    if df['event_timestamp'].isnull().any():
        print("Warning: Some timestamps could not be parsed and were set to NaT (Not a Time).")

    # Sort by driver_id and event_timestamp to process data in time order
    df = df.sort_values(by=['driver_id', 'event_timestamp'])

    # Initialize an empty list to store cumulative counts for merging later
    completed_booking_counts = []

    # Group the data by driver
    for driver_id, driver_data in df.groupby('driver_id'):
        # Track the cumulative count of completed bookings for the current driver
        cumulative_count = 0

        # Iterate over the driver-specific data in chronological order
        for idx, row in driver_data.iterrows():
            # Count the number of completed bookings up to this point
            completed_booking_counts.append({
                'driver_id': driver_id,
                'event_timestamp': row['event_timestamp'],
                'completed_booking_count': cumulative_count  # Add count before updating
            })

            # If the current booking status is 'COMPLETED', increment the cumulative count (case-insensitive)
            if 'booking_status' in df.columns and str(row['booking_status']).lower() == 'completed':
                cumulative_count += 1

    # Convert the completed_booking_counts list to a DataFrame for merging
    df_completed_counts = pd.DataFrame(completed_booking_counts)

    # Drop the original 'completed_booking_count' column before merging to avoid conflicts
    if 'completed_booking_count' in df.columns:
        df = df.drop(columns=['completed_booking_count'])

    # Merge the cumulative completed booking counts directly back into the main dataset
    df = pd.merge(df, df_completed_counts[['driver_id', 'event_timestamp', 'completed_booking_count']],
                  on=['driver_id', 'event_timestamp'], how='left')

    # Debug print: Check if completed_booking_count column exists after merge
    print("Columns after merge:", df.columns)
    print("Sample rows after merge:", df[['driver_id', 'event_timestamp', 'completed_booking_count']].head())

    # Return the updated DataFrame with the completed_booking_count feature
    return df