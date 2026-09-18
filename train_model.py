import argparse
import json
from pathlib import Path
import tensorflow as tf
from tensorflow.keras.datasets import imdb
from tensorflow.keras.preprocessing.sequence import pad_sequences
from model_io import build_model, save_model

def parse_args() -> argparse.Namespace:
    pass

def load_data(max_len: int, limit: int=0):
    pass

def train_model(model, x_train, y_train, batch_size=32, epochs=3):
    pass

def main() -> None:
    pass
