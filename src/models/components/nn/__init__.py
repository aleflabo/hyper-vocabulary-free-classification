from src.models.components.nn.classifiers import NearestNeighboursClassifier
from src.models.components.nn.encoders import LanguageTransformer
from src.models.components.nn.hyper_classifier import Hyper_NearestNeighboursClassifier

__all__ = ["NearestNeighboursClassifier", "LanguageTransformer"]

CLASSIFIERS = {
    "nearest_neighbours": NearestNeighboursClassifier,
    "hyperNearestNeighbours": Hyper_NearestNeighboursClassifier,
}

LANGUAGE_ENCODERS = {
    "transformer": LanguageTransformer,
}
