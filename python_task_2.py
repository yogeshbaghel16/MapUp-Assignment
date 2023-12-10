import pandas as pd


def calculate_distance_matrix(df)->pd.DataFrame():
    """
    Calculate a distance matrix based on the dataframe, df.

    Args:
        df (pandas.DataFrame)

    Returns:
        pandas.DataFrame: Distance matrix
    """
    # Write your logic here
    def calculate_distance_matrix(df):
    # Read the CSV file into a DataFrame
    df = pd.read_csv("dataset-3.csv")

    # Create an empty distance matrix with IDs as both index and columns
    ids = sorted(set(df['id_start'].tolist() + df['id_end'].tolist()))
    distance_matrix = pd.DataFrame(0, index=ids, columns=ids)

    # Populate the distance matrix with cumulative distances
    for index, row in df.iterrows():
        distance_matrix.at[row['id_start'], row['id_end']] += row['distance']
        distance_matrix.at[row['id_end'], row['id_start']] += row['distance']

    # Set diagonal values to 0
    distance_matrix.values[[range(len(ids))]*2] = 0

    return distance_matrix

# Example usage
file_path = 'path_to_your_dataset/dataset-3.csv'
result_matrix = calculate_distance_matrix(file_path)

# Print or use the resulting distance matrix
print(result_matrix)

    return df


def unroll_distance_matrix(df)->pd.DataFrame():
    """
    Unroll a distance matrix to a DataFrame in the style of the initial dataset.

    Args:
        df (pandas.DataFrame)

    Returns:
        pandas.DataFrame: Unrolled DataFrame containing columns 'id_start', 'id_end', and 'distance'.
    """
    # Write your logic here(before solving this question ..import warnings = import warningswarnings.filterwarnings("ignore"))
    def unroll_distance_matrix(distance_matrix):
    # Create an empty DataFrame to store unrolled distances
    unrolled_df = pd.DataFrame(columns=['id_start', 'id_end', 'distance'])

    # Iterate over the rows of the distance_matrix
    for index, row in distance_matrix.iterrows():
        id_start = row.name
        # Iterate over the columns of the distance_matrix
        for id_end, distance in row.items():
            # Add the combination to the unrolled DataFrame
            unrolled_df = unrolled_df.append({'id_start': id_start, 'id_end': id_end, 'distance': distance}, ignore_index=True)

    return unrolled_df

# Example usage with the result_matrix from the previous question
result_matrix = calculate_distance_matrix('dataset-3.csv')
unrolled_result = unroll_distance_matrix(result_matrix)

# Print or use the resulting unrolled DataFrame
print(unrolled_result)

    return df


def find_ids_within_ten_percentage_threshold(df, reference_id)->pd.DataFrame():
    """
    Find all IDs whose average distance lies within 10% of the average distance of the reference ID.

    Args:
        df (pandas.DataFrame)
        reference_id (int)

    Returns:
        pandas.DataFrame: DataFrame with IDs whose average distance is within the specified percentage threshold
                          of the reference ID's average distance.
    """
    # Write your logic here
    def find_ids_within_ten_percentage_threshold(df, reference_value):
    # Filter DataFrame based on the reference_value
    reference_df = df[df['id_start'] == reference_value]

    # Check if the reference value exists in the DataFrame
    if reference_df.empty:
        raise ValueError(f"Reference value {reference_value} not found in the DataFrame.")

    # Calculate the average distance for the reference value
    average_distance = reference_df['distance'].mean()

    # Calculate the threshold range (10% of the average distance)
    threshold_range = 0.1 * average_distance

    # Find IDs within the threshold range
    result_ids = df[(df['distance'] >= (average_distance - threshold_range)) & (df['distance'] <= (average_distance + threshold_range))]['id_start'].unique()

    # Sort the result IDs
    result_ids = sorted(result_ids)

    return result_ids

# Example usage with the provided DataFrame
# Assuming the DataFrame is named df
reference_value = 1001436  # Replace with your desired reference value
result_within_threshold = find_ids_within_ten_percentage_threshold(df, reference_value)

# Print or use the resulting list of IDs
print(result_within_threshold)

    return df


def calculate_toll_rate(df)->pd.DataFrame():
    """
    Calculate toll rates for each vehicle type based on the unrolled DataFrame.

    Args:
        df (pandas.DataFrame)

    Returns:
        pandas.DataFrame
    """
    # Wrie your logic here
    def calculate_toll_rate(df):
    # Define rate coefficients for each vehicle type
    rate_coefficients = {'moto': 0.8, 'car': 1.2, 'rv': 1.5, 'bus': 2.2, 'truck': 3.6}

    # Add columns for each vehicle type with their toll rates
    for vehicle_type, rate_coefficient in rate_coefficients.items():
        df[vehicle_type] = df['distance'] * rate_coefficient

    return df

# Example usage with the provided DataFrame
# Assuming the DataFrame is named unrolled_result
result_with_toll_rates = calculate_toll_rate(unrolled_result)

# Print or use the resulting DataFrame with toll rates
print(result_with_toll_rates)

    return df


def calculate_time_based_toll_rates(df)->pd.DataFrame():
    """
    Calculate time-based toll rates for different time intervals within a day.

    Args:
        df (pandas.DataFrame)

    Returns:
        pandas.DataFrame
    """
    # Write your logic here
    from datetime import time

def calculate_time_based_toll_rates(df):
    # Define time ranges and discount factors for weekdays and weekends
    weekday_time_ranges = [(time(0, 0, 0), time(10, 0, 0)),
                           (time(10, 0, 0), time(18, 0, 0)),
                           (time(18, 0, 0), time(23, 59, 59))]

    weekend_time_range = (time(0, 0, 0), time(23, 59, 59))
    discount_factors_weekday = [0.8, 1.2, 0.8]
    discount_factor_weekend = 0.7

    # Add columns for time-based toll rates
    for start_day in ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]:
        for start_time, end_time in weekday_time_ranges:
            df.loc[(df['start_day'] == start_day) & (df['start_time'] >= start_time) & (df['start_time'] <= end_time), (start_day, start_time, start_day, end_time)] = df['distance'] * discount_factors_weekday[weekday_time_ranges.index((start_time, end_time))]

        df.loc[df['start_day'] == start_day, (start_day, weekend_time_range[0], start_day, weekend_time_range[1])] = df['distance'] * discount_factor_weekend

    # Concatenate time-based toll rates columns
    time_based_toll_rates_columns = [f'{start_day}_{start_time}_{end_day}_{end_time}' for start_day in ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
                                     for start_time, _ in weekday_time_ranges + [weekend_time_range]
                                     for end_day in ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
                                     for _, end_time in weekday_time_ranges + [weekend_time_range]]

    # Convert columns to proper types
    df[time_based_toll_rates_columns] = df[time_based_toll_rates_columns].fillna(0)
    df[['start_day', 'end_day']] = df[['start_day', 'end_day']].astype(str)
    df[['start_time', 'end_time']] = df[['start_time', 'end_time']].apply(pd.to_datetime, errors='coerce').dt.time

    return df

# Example usage with the provided DataFrame
# Assuming the DataFrame is named result_with_toll_rates
result_with_time_based_toll_rates = calculate_time_based_toll_rates(result_with_toll_rates)

# Print or use the resulting DataFrame with time-based toll rates
print(result_with_time_based_toll_rates)

    return df
