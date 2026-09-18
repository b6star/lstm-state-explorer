from pathlib import Path
import tensorflow as tf

def build_model(vocab_size: int=10000) -> tf.keras.Model:
    pass

def save_model(model: tf.keras.Model, model_path: Path) -> None:
    pass

def load_model(model_path: Path) -> tf.keras.Model:
    pass
