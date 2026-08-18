"""
Utility functions for the GB BESS Market Analysis project.
"""

import os
import yaml
from pathlib import Path
from typing import Dict, Any, Optional
from datetime import datetime, timedelta
import pandas as pd
from loguru import logger
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Project root directory
PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
REPORTS_DIR = PROJECT_ROOT / "reports"
LOGS_DIR = PROJECT_ROOT / "logs"

# Ensure directories exist
for directory in [RAW_DATA_DIR, PROCESSED_DATA_DIR, REPORTS_DIR, LOGS_DIR]:
    directory.mkdir(parents=True, exist_ok=True)


def load_config(config_path: Optional[str] = None) -> Dict[str, Any]:
    """
    Load configuration from YAML file.
    
    Args:
        config_path: Path to config file. If None, uses default config.yaml
        
    Returns:
        Dictionary containing configuration
    """
    if config_path is None:
        config_path = PROJECT_ROOT / "config.yaml"
    
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    return config


def setup_logging(config: Optional[Dict[str, Any]] = None):
    """
    Set up logging configuration.
    
    Args:
        config: Configuration dictionary. If None, loads from config.yaml
    """
    if config is None:
        config = load_config()
    
    log_config = config.get('logging', {})
    log_level = os.getenv('LOG_LEVEL', log_config.get('level', 'INFO'))
    log_file = LOGS_DIR / log_config.get('file', 'bess_analysis.log')
    log_format = log_config.get('format', '{time:YYYY-MM-DD HH:mm:ss} | {level} | {message}')
    
    # Remove default handler
    logger.remove()
    
    # Add console handler
    logger.add(
        lambda msg: print(msg, end=""),
        format=log_format,
        level=log_level,
        colorize=True
    )
    
    # Add file handler
    logger.add(
        log_file,
        format=log_format,
        level=log_level,
        rotation="10 MB",
        retention="30 days"
    )
    
    logger.info(f"Logging initialized at {log_level} level")


def get_api_key(service: str) -> str:
    """
    Retrieve API key from environment variables.
    
    Args:
        service: Name of the service (e.g., 'ELEXON')
        
    Returns:
        API key string
        
    Raises:
        ValueError: If API key not found
    """
    env_var = f"{service.upper()}_API_KEY"
    api_key = os.getenv(env_var)
    
    if not api_key:
        raise ValueError(
            f"API key for {service} not found. "
            f"Please set {env_var} in your .env file."
        )
    
    return api_key


def parse_date(date_str: str) -> datetime:
    """
    Parse date string to datetime object.
    
    Args:
        date_str: Date string in various formats
        
    Returns:
        datetime object
    """
    formats = [
        "%Y-%m-%d",
        "%Y/%m/%d",
        "%d-%m-%Y",
        "%d/%m/%Y",
        "%Y-%m-%d %H:%M:%S",
    ]
    
    for fmt in formats:
        try:
            return datetime.strptime(date_str, fmt)
        except ValueError:
            continue
    
    raise ValueError(f"Unable to parse date: {date_str}")


def generate_date_range(
    start_date: str,
    end_date: str,
    freq: str = 'D'
) -> pd.DatetimeIndex:
    """
    Generate a date range between start and end dates.
    
    Args:
        start_date: Start date string
        end_date: End date string
        freq: Frequency ('D' for daily, 'H' for hourly, etc.)
        
    Returns:
        DatetimeIndex with date range
    """
    start = parse_date(start_date)
    end = parse_date(end_date)
    
    return pd.date_range(start=start, end=end, freq=freq)


def save_dataframe(
    df: pd.DataFrame,
    filename: str,
    data_type: str = 'raw',
    format: str = 'csv'
) -> Path:
    """
    Save DataFrame to file.
    
    Args:
        df: DataFrame to save
        filename: Name of the file
        data_type: 'raw' or 'processed'
        format: File format ('csv', 'parquet', 'pickle')
        
    Returns:
        Path to saved file
    """
    if data_type == 'raw':
        directory = RAW_DATA_DIR
    elif data_type == 'processed':
        directory = PROCESSED_DATA_DIR
    else:
        raise ValueError(f"Invalid data_type: {data_type}")
    
    # Add extension if not present
    if not filename.endswith(f'.{format}'):
        filename = f"{filename}.{format}"
    
    filepath = directory / filename
    
    if format == 'csv':
        df.to_csv(filepath, index=False)
    elif format == 'parquet':
        df.to_parquet(filepath, index=False)
    elif format == 'pickle':
        df.to_pickle(filepath)
    else:
        raise ValueError(f"Unsupported format: {format}")
    
    logger.info(f"Saved data to {filepath}")
    return filepath


def load_dataframe(
    filename: str,
    data_type: str = 'raw',
    format: str = 'csv'
) -> pd.DataFrame:
    """
    Load DataFrame from file.
    
    Args:
        filename: Name of the file
        data_type: 'raw' or 'processed'
        format: File format ('csv', 'parquet', 'pickle')
        
    Returns:
        Loaded DataFrame
    """
    if data_type == 'raw':
        directory = RAW_DATA_DIR
    elif data_type == 'processed':
        directory = PROCESSED_DATA_DIR
    else:
        raise ValueError(f"Invalid data_type: {data_type}")
    
    # Add extension if not present
    if not filename.endswith(f'.{format}'):
        filename = f"{filename}.{format}"
    
    filepath = directory / filename
    
    if not filepath.exists():
        raise FileNotFoundError(f"File not found: {filepath}")
    
    if format == 'csv':
        df = pd.read_csv(filepath)
    elif format == 'parquet':
        df = pd.read_parquet(filepath)
    elif format == 'pickle':
        df = pd.read_pickle(filepath)
    else:
        raise ValueError(f"Unsupported format: {format}")
    
    logger.info(f"Loaded data from {filepath}")
    return df


def calculate_settlement_period(timestamp: datetime) -> int:
    """
    GB electricity settlement period (1-48) for a local-time timestamp.

    Settlement period 1 covers 00:00-00:30 local time, period 48 covers
    22:30-23:00 — matching the convention used by Elexon (which reports
    settlementPeriod 47 at 23:00 and 48 at 23:30 on the same settlementDate)
    and by EFA_PERIODS in src/analysis/revenue_stack.py.

    Note this is a clock-time mapping and assumes a 48-period day; on the two
    BST transition days a date has 46 or 50 periods. Use
    settlement_periods_in_day() for the count.

    Args:
        timestamp: naive datetime in GB local time

    Returns:
        Settlement period number (1-48)
    """
    return (timestamp.hour * 60 + timestamp.minute) // 30 + 1


def settlement_periods_in_day(date) -> int:
    """
    Number of settlement periods on a given GB calendar date.

    Normally 48. On the spring-forward day the 01:00-02:00 hour does not exist,
    giving 46; on the fall-back day it occurs twice, giving 50. Derived from the
    actual UTC offset change rather than hardcoded transition dates, so it stays
    correct as the transition dates move year to year.

    Args:
        date: date or datetime (naive; interpreted as a GB calendar date)

    Returns:
        46, 48 or 50
    """
    from datetime import date as _date, datetime as _datetime, timedelta
    from zoneinfo import ZoneInfo

    tz = ZoneInfo("Europe/London")
    day = date.date() if isinstance(date, _datetime) else _date(date.year, date.month, date.day)

    start = _datetime(day.year, day.month, day.day, tzinfo=tz)
    nxt = day + timedelta(days=1)
    end = _datetime(nxt.year, nxt.month, nxt.day, tzinfo=tz)

    hours = (end.astimezone(ZoneInfo("UTC")) - start.astimezone(ZoneInfo("UTC"))).total_seconds() / 3600
    return int(round(hours * 2))


if __name__ == "__main__":
    # Test utility functions
    setup_logging()
    config = load_config()
    logger.info(f"Configuration loaded: {config['project']['name']}")
    
    # Test date range generation
    dates = generate_date_range("2024-01-01", "2024-01-07")
    logger.info(f"Generated {len(dates)} dates")
    
    # Test settlement period calculation
    test_time = datetime(2024, 1, 1, 15, 30)
    period = calculate_settlement_period(test_time)
    logger.info(f"Settlement period for {test_time}: {period}")
