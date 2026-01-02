from core.engine import TradeEngine

engine = TradeEngine()

def get_trade_data(path: str, conf_threshold: float = 0.25) -> dict:
    return engine.process_image(path, conf_threshold)

if __name__ == "__main__":
    result = get_trade_data("test.png")
    print(result)
