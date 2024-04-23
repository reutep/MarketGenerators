# to test, run:
# python -m src.market_generation.market_generator
from ..data.make_dataset import DataLoader
from ..features.encoder import Encoder
from ..features.decoder import Decoder

class MarketGenerator:
    def __init__(self, data_loader: DataLoader, encoder: Encoder, model=None, decoder: Decoder = None):
        self.data_loader = data_loader
        self.encoder = encoder
        self.model = model
        self.decoder = decoder

    def load_data(self):
        # output format to be adjusted
        data, times = self.data_loader.create_dataset()
        return data, times

    def transform_data(self, data):
        return self.encoder.encode(data)

    def train_model(self, transformed_data):
        self.model.train(transformed_data)

    def evaluate_model(self):
        return self.model.evaluate()

    def transform_back(self, output):
        if self.decoder:
            return self.decoder.decode(output)
        return output
    
    def run_pipeline(self):
        # run the entire pipeline
        # to be adjusted
        self.load_data()
        self.transform_data()
        self.train_model()
        self.evaluate_model()
        self.transform_back()
