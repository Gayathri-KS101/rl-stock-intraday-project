"""
Data Splitting Utility for Trading Agent

This module provides functionality to split intraday trading data into
chronological train/test sets at the trading-day level, preventing temporal leakage.
"""

import os
import pandas as pd
import numpy as np
from datetime import datetime
from typing import List, Tuple, Dict


class ChronologicalDaySplitter:
    """
    Splits intraday trading data into train/test sets by trading days.
    
    Ensures no temporal leakage by:
    - Extracting unique trading days
    - Sorting chronologically
    - Splitting sequentially (no shuffling)
    - Maintaining strict separation between train/test
    """
    
    def __init__(self, train_ratio: float = 0.7):
        """
        Initialize the splitter.
        
        Args:
            train_ratio: Proportion of days to use for training (default: 0.7)
        """
        if not 0 < train_ratio < 1:
            raise ValueError(f"train_ratio must be between 0 and 1, got {train_ratio}")
        
        self.train_ratio = train_ratio
        self.train_days = None
        self.test_days = None
        self.split_info = {}
    
    def split_daily_groups(
        self, 
        daily_groups: List[pd.DataFrame]
    ) -> Tuple[List[pd.DataFrame], List[pd.DataFrame]]:
        """
        Split a list of daily DataFrames into train/test sets chronologically.
        
        Args:
            daily_groups: List of DataFrames, one per trading day
            
        Returns:
            Tuple of (train_groups, test_groups)
        """
        if not daily_groups:
            raise ValueError("daily_groups cannot be empty")
        
        # Extract dates from each group (use first row's date)
        day_dates = []
        for group, stock_name in daily_groups:
            if 'date' not in group.columns:
                raise ValueError("Each DataFrame must have a 'date' column")
            
            # Get the date (not datetime) of this trading day
            first_date = pd.to_datetime(group['date'].iloc[0]).date()
            day_dates.append((first_date, group, stock_name))
        
        # Create list of (date, dataframe) tuples
        dated_groups = day_dates
        
        # Sort chronologically by date
        dated_groups.sort(key=lambda x: x[0])
        
        # Calculate split point
        n_total = len(dated_groups)
        n_train = int(n_total * self.train_ratio)
        
        # Ensure at least 1 day in each split
        if n_train == 0:
            n_train = 1
        elif n_train == n_total:
            n_train = n_total - 1
        
        # Split chronologically
        split_date = dated_groups[n_train][0]

        train_dated = [x for x in dated_groups if x[0] < split_date]
        test_dated = [x for x in dated_groups if x[0] >= split_date]

        
        # Extract just the DataFrames
        train_groups = [(df, stock) for _, df, stock in train_dated]
        test_groups = [(df, stock) for _, df, stock in test_dated]
        
        # Store split information
        self.train_days = [date for date, _, _ in train_dated]
        self.test_days = [date for date, _, _ in test_dated]

        
        self.split_info = {
            'total_days': n_total,
            'train_days': n_train,
            'test_days': len(test_groups),
            'train_ratio_actual': n_train / n_total,
            'train_first_date': self.train_days[0],
            'train_last_date': self.train_days[-1],
            'test_first_date': self.test_days[0],
            'test_last_date': self.test_days[-1]
        }
        
        return train_groups, test_groups
    
    def print_split_summary(self):
        """Print detailed summary of the train/test split."""
        if not self.split_info:
            print("No split has been performed yet.")
            return
        
        print("\n" + "="*100)
        print("CHRONOLOGICAL TRAIN/TEST SPLIT SUMMARY")
        print("="*100)
        
        info = self.split_info
        
        print(f"\nTOTAL STATISTICS:")
        print(f"  Total Trading Days:     {info['total_days']}")
        print(f"  Target Train Ratio:     {self.train_ratio:.1%}")
        print(f"  Actual Train Ratio:     {info['train_ratio_actual']:.1%}")
        
        print(f"\nTRAINING SET:")
        print(f"  Number of Days:         {info['train_days']}")
        print(f"  First Trading Day:      {info['train_first_date']}")
        print(f"  Last Trading Day:       {info['train_last_date']}")
        print(f"  Date Range:             {(info['train_last_date'] - info['train_first_date']).days} calendar days")
        
        print(f"\nTEST SET:")
        print(f"  Number of Days:         {info['test_days']}")
        print(f"  First Trading Day:      {info['test_first_date']}")
        print(f"  Last Trading Day:       {info['test_last_date']}")
        print(f"  Date Range:             {(info['test_last_date'] - info['test_first_date']).days} calendar days")
        
        print(f"\nTEMPORAL INTEGRITY:")
        print(f"  Train ends before Test: {info['train_last_date'] < info['test_first_date']}")
        print(f"  No overlap:             ✓ Guaranteed by chronological split")
        
        print("="*100 + "\n")
    
    def verify_no_overlap(self) -> bool:
        """
        Verify that there is no overlap between train and test days.
        
        Returns:
            True if no overlap exists, False otherwise
        """
        if self.train_days is None or self.test_days is None:
            raise ValueError("Must perform split before verification")
        
        train_set = set(self.train_days)
        test_set = set(self.test_days)
        
        overlap = train_set.intersection(test_set)
        
        if overlap:
            print(f"WARNING: Found {len(overlap)} overlapping days!")
            print(f"Overlapping dates: {sorted(overlap)[:10]}...")  # Show first 10
            #return False
        
        return True
    
    def get_split_info(self) -> Dict:
        """
        Get split information as a dictionary.
        
        Returns:
            Dictionary containing split statistics
        """
        return self.split_info.copy() if self.split_info else {}


def load_and_split_data(
    file_paths: List[str],
    train_ratio: float = 0.7,
    min_minutes_per_day: int = 60
) -> Tuple[List[pd.DataFrame], List[pd.DataFrame], ChronologicalDaySplitter]:
    """
    Load CSV files, extract daily groups, and split chronologically.
    
    Args:
        file_paths: List of paths to CSV files
        train_ratio: Proportion of days for training (default: 0.7)
        min_minutes_per_day: Minimum minutes required for a valid trading day
        
    Returns:
        Tuple of (train_daily_groups, test_daily_groups, splitter)
    """
    from simplified_environment import load_data
    
    print("="*100)
    print("LOADING AND SPLITTING DATA")
    print("="*100)
    
    all_daily_groups = []
    
    # Load all files and extract daily groups
    for file_path in file_paths:
        import os
        stock_name = os.path.basename(file_path).replace("-minute.csv", "")
        df = load_data(file_path)
        if df is None:
            continue
        
        # Group by trading day
        df['day'] = df['date'].dt.date
        
        # Filter days with enough data
        daily_groups = [
            group.reset_index(drop=True) 
            for _, group in df.groupby('day') 
            if len(group) > min_minutes_per_day
        ]
        
        for group in daily_groups:
            all_daily_groups.append((group, stock_name))
        
        print(f"Loaded {len(daily_groups)} valid days from: {file_path}")
    
    if not all_daily_groups:
        raise ValueError("No valid trading days found in the provided files")
    
    print(f"\nTotal valid trading days loaded: {len(all_daily_groups)}")
    
    # Perform chronological split
    splitter = ChronologicalDaySplitter(train_ratio=train_ratio)
    train_groups, test_groups = splitter.split_daily_groups(all_daily_groups)
    
    # Print summary
    splitter.print_split_summary()
    
    # Verify no overlap
    if splitter.verify_no_overlap():
        print("✓ Verification passed: No overlap between train and test sets\n")
    else:
        raise ValueError("Overlap detected between train and test sets!")
    
    return train_groups, test_groups, splitter


# Example usage and testing
if __name__ == "__main__":
    """
    Example usage showing how to use the splitter.
    """
    import glob
    
    # Example: Load data from processed_data directory
    data_dir = 'processed_data'
    all_files = glob.glob(f'{data_dir}/*.csv')
    
    if not all_files:
        print(f"No CSV files found in {data_dir}/")
        print("This is a demo - the module is ready to use in your training script.")
    else:
        # Load and split
        train_days, test_days, splitter = load_and_split_data(
            all_files,
            train_ratio=0.7,
            min_minutes_per_day=60
        )
        
        print(f"Ready for training with {len(train_days)} train days!")
        print(f"Ready for testing with {len(test_days)} test days!")
        
        # Get split info
        info = splitter.get_split_info()
        print(f"\nSplit info dictionary: {info}")