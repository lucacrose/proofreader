from core.engine import TradeEngine

engine = TradeEngine()

def get_trade_data(path: str) -> dict:
    return engine.process_image(path)

if __name__ == "__main__":
    result = get_trade_data("test.png")
    print(result)
