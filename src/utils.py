"""
Utility functions for the universal e-nose system
"""

import os
import json
import logging
from datetime import datetime
from typing import Dict, Any

logger = logging.getLogger(__name__)


def setup_logging(log_level: str = "INFO"):
    """Setup logging configuration"""
    log_format = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=getattr(logging, log_level.upper()), format=log_format)


def load_config(config_path: str) -> Dict[str, Any]:
    """Load configuration from JSON file"""
    try:
        with open(config_path, 'r') as f:
            return json.load(f)
    except FileNotFoundError:
        logger.error(f"Configuration file not found: {config_path}")
        raise
    except json.JSONDecodeError as e:
        logger.error(f"Invalid JSON in configuration file: {e}")
        raise


def ensure_directory_exists(directory: str):
    """Ensure directory exists, create if necessary"""
    if not os.path.exists(directory):
        os.makedirs(directory)
        logger.info(f"Created directory: {directory}")


def get_timestamp() -> str:
    """Get current timestamp in ISO format"""
    return datetime.now().isoformat()


def safe_save_json(data: Dict, filepath: str):
    """Safely save JSON data with backup"""
    try:
        if os.path.exists(filepath):
            backup_path = f"{filepath}.backup"
            os.rename(filepath, backup_path)

        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2)

        logger.info(f"Saved data to: {filepath}")

    except Exception as e:
        logger.error(f"Error saving JSON data: {e}")
        if os.path.exists(f"{filepath}.backup"):
            os.rename(f"{filepath}.backup", filepath)
        raise