"""
Comprehensive logging utility for AI Fight Coach application.
Handles debug, info, warning, and error logging with proper formatting.
"""

import logging
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Optional


class Logger:
    """Centralized logging utility for the AI Fight Coach application."""
    
    def __init__(self, name: str = "ai_fight_coach", log_dir: str = "debug_logs"):
        self.name = name
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(exist_ok=True)
        
        # Create logger
        self.logger = logging.getLogger(name)
        self.logger.setLevel(logging.DEBUG)
        
        # Prevent duplicate handlers
        if not self.logger.handlers:
            self._setup_handlers()
    
    def _setup_handlers(self):
        """Setup file and console handlers with proper formatting."""
        # Create formatters
        detailed_formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(funcName)s:%(lineno)d - %(message)s'
        )
        simple_formatter = logging.Formatter(
            '%(asctime)s - %(levelname)s - %(message)s'
        )
        
        # File handler for all logs
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        file_handler = logging.FileHandler(
            self.log_dir / f"app_{timestamp}.log",
            encoding='utf-8'
        )
        file_handler.setLevel(logging.DEBUG)
        file_handler.setFormatter(detailed_formatter)
        
        # Console handler for important logs
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(logging.INFO)
        console_handler.setFormatter(simple_formatter)
        
        # Add handlers
        self.logger.addHandler(file_handler)
        self.logger.addHandler(console_handler)
    
    def debug(self, message: str):
        """Log debug message."""
        self.logger.debug(message)
    
    def info(self, message: str):
        """Log info message."""
        self.logger.info(message)
    
    def warning(self, message: str):
        """Log warning message."""
        self.logger.warning(message)
    
    def error(self, message: str):
        """Log error message."""
        self.logger.error(message)
    
    def critical(self, message: str):
        """Log critical message."""
        self.logger.critical(message)
    
    def log_gemini_request(self, prompt: str, video_path: str, metadata: dict):
        """Log Gemini API request details."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_file = self.log_dir / f"gemini_request_{timestamp}.log"
        
        with open(log_file, 'w', encoding='utf-8') as f:
            f.write(f"GEMINI REQUEST LOG - {timestamp}\n")
            f.write("=" * 50 + "\n")
            f.write(f"Video Path: {video_path}\n")
            f.write(f"Metadata: {metadata}\n")
            f.write(f"Prompt:\n{prompt}\n")
            f.write("=" * 50 + "\n")
        
        self.info(f"Gemini request logged to {log_file}")
    
    def log_gemini_response(self, response: str, timestamp: str):
        """Log Gemini API response."""
        log_file = self.log_dir / f"gemini_response_{timestamp}.log"
        
        with open(log_file, 'w', encoding='utf-8') as f:
            f.write(f"GEMINI RESPONSE LOG - {timestamp}\n")
            f.write("=" * 50 + "\n")
            f.write(f"Raw Response:\n{response}\n")
            f.write("=" * 50 + "\n")
        
        self.info(f"Gemini response logged to {log_file}")


# Global logger instance
logger = Logger() 