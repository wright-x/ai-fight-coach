"""
AI Fight Coach Utilities Package
"""

from .logger import logger

# Don't import at module level to avoid cv2 import errors
# Components will be imported individually when needed

__all__ = ['logger'] 