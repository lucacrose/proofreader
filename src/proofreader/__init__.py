# This file tells Python what the 'proofreader' module contains
from .core.engine import get_trade_data

# This makes it accessible as proofreader.get_trade_data
__all__ = ["get_trade_data"]