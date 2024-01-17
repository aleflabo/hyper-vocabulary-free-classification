from src.models.cased import CaSED
from src.models.clip import CLIP
from src.models.vocabulary_free_clip import VocabularyFreeCLIP
from src.models.meru import CaSED_MERU
from src.models.meruclip import CaSED_CLIP_MERU

__all__ = ["CaSED", "CLIP", "VocabularyFreeCLIP", "CaSED_MERU", "CaSED_CLIP_MERU"]

MODELS = {
    "cased": CaSED,
    "clip": CLIP,
    "vocabulary_free_clip": VocabularyFreeCLIP,
    "cased_meru": CaSED_MERU,
    "cased_meru": CaSED_CLIP_MERU
}
