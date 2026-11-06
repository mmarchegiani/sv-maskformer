"""General callbacks: checkpoint selection, early stopping and the prediction-writer base class."""

from maskformer.callbacks.best_checkpoint import BestCheckpoint
from maskformer.callbacks.early_stopping import EarlyStopping
from maskformer.callbacks.prediction_writer import PredictionWriterBase

__all__ = ["BestCheckpoint", "EarlyStopping", "PredictionWriterBase"]
