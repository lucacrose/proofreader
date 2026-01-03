from core.engine import TradeEngine

engine = TradeEngine()

if __name__ == "__main__":
    result = engine.process_image("test.png")
    print(result)
