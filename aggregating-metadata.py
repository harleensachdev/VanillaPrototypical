import pandas as pd
import numpy as np
from datetime import datetime
from typing import Dict
import os

# Define all labels
LABEL_MAP = {
    "alarm": 0,
    "non_alarm": 1,
    "background":2,
    "highfreq_noise" : 3,
    "insect_call" : 4,
    "weather_rain" :5,
    "lowfreq_noise": 6
}
def extract_datetime_from_metadata(row) -> str:
    """
    Extract datetime key from metadata row for grouping.
    Combines date and time into a single datetime string.
    
    Args:
        row: Pandas Series containing 'date' and 'time' columns
        
    Returns:
        String in format 'YYYYMMDD_HHMMSS' for grouping
    """
    date_str = str(row['date'])
    time_str = str(row['time']).zfill(6)  # Pad with zeros if needed
    return f"{date_str}_{time_str}"

def get_label_columns(df: pd.DataFrame) -> list:
    """
    Identify all label count columns in the DataFrame.
    
    Args:
        df: Input DataFrame
        
    Returns:
        List of column names that contain label counts
    """
    label_columns = []
    for label in LABEL_MAP.keys():
        col_name = f"{label}_count"
        if col_name in df.columns:
            label_columns.append(col_name)
    return label_columns

def aggregate_metadata_by_time(metadata_df: pd.DataFrame) -> pd.DataFrame:
    """
    Aggregate metadata by date+time, calculating mean counts for files recorded
    at the same time for all available labels.
    
    Args:
        metadata_df: DataFrame with columns including 'date', 'time', 
                    and various '*_count' columns for each label
    
    Returns:
        Aggregated DataFrame with mean counts per datetime for all labels
    """
    # Create datetime grouping key
    metadata_df['datetime_key'] = metadata_df.apply(extract_datetime_from_metadata, axis=1)
    
    # Identify available label columns
    label_columns = get_label_columns(metadata_df)
    print(f"Found label columns: {label_columns}")
    
    # Group by datetime and calculate statistics
    aggregated_data = []
    
    for datetime_key, group in metadata_df.groupby('datetime_key'):
        # Extract date and time components
        date_part = datetime_key.split('_')[0]
        time_part = datetime_key.split('_')[1]
        
        # Additional statistics
        num_files = len(group)
        sites = group['site'].unique().tolist() if 'site' in group.columns else ['unknown']
        
        # Create aggregated row with basic info
        agg_row = {
            'date': date_part,
            'time': time_part,
            'datetime_key': datetime_key,
            'num_soundscapes': num_files,
            'sites': ','.join(sites)
        }
        
        # Calculate statistics for each available label
        for label_col in label_columns:
            label_name = label_col.replace('_count', '')
            
            # Mean counts
            agg_row[f'mean_{label_name}_count'] = round(group[label_col].mean(), 2)
            
            # Total counts
            agg_row[f'total_{label_name}_count'] = group[label_col].sum()
            
            # Standard deviation
            agg_row[f'std_{label_name}_count'] = round(group[label_col].std(), 2) if num_files > 1 else 0
        
        aggregated_data.append(agg_row)
    
    # Create final DataFrame
    result_df = pd.DataFrame(aggregated_data)
    
    # Sort by datetime for better readability
    result_df = result_df.sort_values(['date', 'time']).reset_index(drop=True)
    
    return result_df

def add_time_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add additional time-based features for analysis.
    
    Args:
        df: Aggregated DataFrame
        
    Returns:
        DataFrame with additional time features
    """
    # Extract hour for hourly analysis
    df['hour'] = df['time'].astype(str).str[:2].astype(int)
    
    # Convert to datetime for additional features
    df['datetime'] = pd.to_datetime(df['date'] + df['time'], format='%Y%m%d%H%M%S')
    
    # Add day of week, month, etc.
    df['day_of_week'] = df['datetime'].dt.day_name()
    df['month'] = df['datetime'].dt.month
    df['day_of_year'] = df['datetime'].dt.dayofyear
    
    return df

def process_metadata_file(input_path: str, output_path: str = None) -> pd.DataFrame:
    """
    Main function to process and aggregate metadata file.
    
    Args:
        input_path: Path to input metadata CSV file
        output_path: Path to save aggregated results (optional)
        
    Returns:
        Aggregated DataFrame
    """
    # Read metadata file
    print(f"Reading metadata from: {input_path}")
    metadata_df = pd.read_csv(input_path)
    print(f"Loaded {len(metadata_df)} rows of metadata")
    
    # Show original data structure
    print("\nOriginal data sample:")
    print(metadata_df.head())
    print(f"\nOriginal columns: {list(metadata_df.columns)}")
    
    # Aggregate by time
    print("\nAggregating by date+time...")
    aggregated_df = aggregate_metadata_by_time(metadata_df)
    
    # Add time features
    aggregated_df = add_time_features(aggregated_df)
    
    print(f"\nAggregated to {len(aggregated_df)} unique datetime periods")
    
    # Show sample of aggregated data with available columns
    sample_columns = ['date', 'time', 'num_soundscapes']
    label_columns = get_label_columns(pd.read_csv(input_path))
    for label_col in label_columns:
        label_name = label_col.replace('_count', '')
        sample_columns.append(f'mean_{label_name}_count')
    
    print("\nAggregated data sample:")
    available_sample_columns = [col for col in sample_columns if col in aggregated_df.columns]
    print(aggregated_df[available_sample_columns].head())
    
    # Save if output path provided
    if output_path:
        aggregated_df.to_csv(output_path, index=False)
        print(f"\nSaved aggregated metadata to: {output_path}")
    
    return aggregated_df

def generate_hourly_summary(aggregated_df: pd.DataFrame) -> pd.DataFrame:
    """
    Generate hourly summary statistics across all dates for all available labels.
    
    Args:
        aggregated_df: Aggregated DataFrame from process_metadata_file
        
    Returns:
        Hourly summary DataFrame
    """
    # Find all mean count columns
    mean_columns = [col for col in aggregated_df.columns if col.startswith('mean_') and col.endswith('_count')]
    
    if not mean_columns:
        print("No mean count columns found for hourly summary")
        return pd.DataFrame()
    
    # Create aggregation dictionary
    agg_dict = {}
    for col in mean_columns:
        agg_dict[col] = ['mean', 'std', 'count']
    
    # Add num_soundscapes if available
    if 'num_soundscapes' in aggregated_df.columns:
        agg_dict['num_soundscapes'] = 'sum'
    
    hourly_summary = aggregated_df.groupby('hour').agg(agg_dict).round(2)
    
    # Flatten column names
    hourly_summary.columns = ['_'.join(col).strip() for col in hourly_summary.columns]
    hourly_summary = hourly_summary.reset_index()
    
    return hourly_summary

def generate_label_summary(aggregated_df: pd.DataFrame) -> pd.DataFrame:
    """
    Generate summary statistics for each label across all time periods.
    
    Args:
        aggregated_df: Aggregated DataFrame from process_metadata_file
        
    Returns:
        Label summary DataFrame
    """
    # Find all mean and total count columns
    mean_columns = [col for col in aggregated_df.columns if col.startswith('mean_') and col.endswith('_count')]
    total_columns = [col for col in aggregated_df.columns if col.startswith('total_') and col.endswith('_count')]
    
    summary_data = []
    
    for mean_col in mean_columns:
        label_name = mean_col.replace('mean_', '').replace('_count', '')
        total_col = f'total_{label_name}_count'
        
        summary_row = {
            'label': label_name,
            'mean_per_period': round(aggregated_df[mean_col].mean(), 2),
            'std_per_period': round(aggregated_df[mean_col].std(), 2),
            'min_per_period': round(aggregated_df[mean_col].min(), 2),
            'max_per_period': round(aggregated_df[mean_col].max(), 2),
        }
        
        if total_col in aggregated_df.columns:
            summary_row['total_across_all_periods'] = aggregated_df[total_col].sum()
        
        summary_data.append(summary_row)
    
    return pd.DataFrame(summary_data)

# Example usage
if __name__ == "__main__":
    # Example usage with your metadata file
    input_file ='/Users/caramelloveschicken/Desktop/data/Habitat Park/PH-Multiple-Distractor-Results/ph-1-fs3-results.csv'
    output_file = '/Users/caramelloveschicken/Desktop/data/Habitat Park/PH-Multiple-Distractor-Results/ph-1-fs3-results-aggregated.csv'
    # Process the metadata
    aggregated_data = process_metadata_file(input_file, output_file)
    
    # Generate additional summaries
    print("\n" + "="*50)
    print("HOURLY SUMMARY")
    print("="*50)
    hourly_summary = generate_hourly_summary(aggregated_data)
    print(hourly_summary.head(10))
    
    print("\n" + "="*50)
    print("LABEL SUMMARY")
    print("="*50)
    label_summary = generate_label_summary(aggregated_data)
    print(label_summary)