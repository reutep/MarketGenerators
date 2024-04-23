# to test, run:
# python -m src.main

from .market_generation.market_generator import MarketGenerator
# from features.encoder import Encoder
# from models.market_generator import MarketGenerator
# from models.model import ModelClass
# from visualization.plotting import plot_results

def main():
    generator = MarketGenerator()

if __name__ == "__main__":
    main()