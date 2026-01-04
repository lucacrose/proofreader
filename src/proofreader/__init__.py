from .core.engine import TradeEngine

_engine = None

def get_trade_data(image_path: str, conf_threshold: float = 0.25):
    global _engine
    if _engine is None:
        _engine = TradeEngine()
    
    return _engine.process_image(image_path, conf_threshold)

__all__ = ["get_trade_data"]
