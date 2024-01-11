from typing import Callable, Optional

import torch
import torchvision.transforms as T
from torchmetrics import MetricCollection
from torchmetrics.aggregation import MeanMetric

from src import utils
from src.data.components.transforms import TextCompose, default_vocab_transform
from src.models.components.metrics import (
    SemanticClusterAccuracy,
    SentenceIOU,
    SentenceScore,
    UniqueValues,
)
from src.models.vocabulary_free_clip import VocabularyFreeCLIP

# MERU imports
from src.utils.meru_utils.config import LazyConfig, LazyFactory
from src.utils.meru_utils.checkpointing import CheckpointManager
from src.utils.meru_utils.tokenizer import Tokenizer

log = utils.get_logger(__name__)


class CaSED_MERU(VocabularyFreeCLIP):
    """LightningModule for Category Search from External Databases.

    It employs MERU's Visual and Textual encoders to generate embeddings for images and texts.

    Reference:
        Conti et al. Vocabulary-free Image Classification. NeurIPS 2023.

    Args:
        vocabulary (BaseVocabulary): Vocabulary to use.
        vocab_transform (TextCompose, optional): List of transforms to apply to a vocabulary.
        model_name (str): Name of the CLIP model to use.
        pretrained (str): Pretrained weights to use for the CLIP model. Defaults to "openai".

    Extra hparams:
        alpha (float): Weight for the average of the image and text predictions. Defaults to 0.5.
        vocab_prompt (str): Prompt to use for a vocabulary. Defaults to "{}".
        tau (float): Temperature to use for the classifier. Defaults to 1.0.
    """

    def __init__(self, *args, vocab_transform: Optional[TextCompose] = None, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self._vocab_transform = vocab_transform or default_vocab_transform()
        device = (
            torch.cuda.current_device()
            if torch.cuda.is_available()
            else torch.device("cpu")
        )

        meru_config = LazyConfig.load(kwargs["meru_config"])
        meru_ckpt = kwargs["meru_ckpt"]
        self.meru = LazyFactory.build_model(meru_config, device).eval()
        CheckpointManager(model=self.meru).load(meru_ckpt)
        self.meru.eval()
        self.meru_tokenizer = Tokenizer() # Don't know if it should override the main tokenizer or not
        # save hyperparameters
        kwargs["alpha"] = kwargs.get("alpha", 0.5)
        self.save_hyperparameters("alpha", "vocab_transform")

    @property
    def vocab_transform(self) -> Callable:
        """Get image preprocess transform.

        The getter wraps the transform in a map_reduce function and applies it to a list of images.
        If interested in the transform itself, use `self._vocab_transform`.
        """
        vocab_transform = self._vocab_transform

        def vocabs_transforms(texts: list[str]) -> list[torch.Tensor]:
            return [vocab_transform(text) for text in texts] # filtering the sentences

        return vocabs_transforms

    @vocab_transform.setter
    def vocab_transform(self, transform: T.Compose) -> None:
        """Set image preprocess transform.

        Args:
            transform (torch.nn.Module): Transform to use.
        """
        self._vocab_transform = transform

    def batch_step(
        self, images_z: torch.Tensor, vocabularies: list[list]
    ) -> tuple[torch.Tensor, list, list]:
        """Perform a single batch step.

        Args:
            images_z (torch.Tensor): Batch of image embeddings.
            images_fp (list[str]): List of paths to image files.
        """
        unfiltered_words = sum(vocabularies, [])

        # encode unfiltered words
        unfiltered_words_z = self.encode_vocabulary(unfiltered_words).squeeze(0)
        unfiltered_words_z = unfiltered_words_z / unfiltered_words_z.norm(dim=-1, keepdim=True)

        # generate a text embedding for each image from their unfiltered words
        unfiltered_words_per_image = [len(vocab) for vocab in vocabularies]
        texts_z = torch.split(unfiltered_words_z, unfiltered_words_per_image)
        texts_z = torch.stack([word_z.mean(dim=0) for word_z in texts_z])
        texts_z = texts_z / texts_z.norm(dim=-1, keepdim=True)

        # filter the words and embed them
        vocabularies = self.vocab_transform(vocabularies)
        vocabularies = [vocab or ["object"] for vocab in vocabularies]
        words = sum(vocabularies, [])
        words_z = self.encode_vocabulary_meru(words, use_prompts=True)
        # words_z = words_z / words_z.norm(dim=-1, keepdim=True) #! Rimosso to be hyperbolic

        # create a one-hot relation mask between images and words
        words_per_image = [len(vocab) for vocab in vocabularies]
        col_indices = torch.arange(sum(words_per_image))
        row_indices = torch.arange(len(images_z)).repeat_interleave(torch.tensor(words_per_image))
        mask = torch.zeros(len(images_z), sum(words_per_image), device=self.device)
        mask[row_indices, col_indices] = 1

        #! get hyperbolic image embeddings
        images_z = self.meru.encode_image(images_z)
        # get the image and text predictions
        if not self.hparams.alpha == 0:
            images_p = self.classifier(images_z, words_z, mask=mask)
        if not self.hparams.alpha == 1:
            texts_p = self.classifier(texts_z, words_z, mask=mask)

        # average the image and text predictions
        samples_p = self.hparams.alpha * images_p + (1 - self.hparams.alpha) * texts_p

        return samples_p, words, vocabularies

    def test_step(self, batch: dict, batch_idx: int, dataloader_idx: int = 0) -> None:
        """Lightning test step.

        Args:
            batch (dict): Batch of data.
            batch_idx (int): Index of the batch.
            dataloader_idx (int, optional): Index of the dataloader. Defaults to 0.
        """
        images = batch["images_tensor"]
        targets = batch["targets_name"]
        images_fp = batch["images_fp"]

        # get vocabularies for each image
        images_z = self.vision_encoder(images) # [64, 768] # encode images with CLIP for retrieval
        images_vocab = self.vocabulary(images_z=images_z, images_fp=images_fp) # this only find the sentences

        # get predictions for each image
        images_p, words, images_vocab = self.batch_step(images_z, images_vocab)
        preds = images_p.topk(k=1, dim=-1)
        images_words = [[words[idx] for idx in indices.tolist()] for indices in preds.indices]
        images_words_values = preds.values.tolist()
        words = [
            {word: sum([v for w, v in zip(iw, iwv) if w == word]) for word in set(iw)}
            for iw, iwv in zip(images_words, images_words_values)
        ]

        # log metrics
        num_vocabs = torch.tensor([len(image_vocab) for image_vocab in images_vocab])
        num_vocabs = num_vocabs.to(self.device)
        self.metrics["test/num_vocabs_avg"](num_vocabs)
        self.log("test/num_vocabs.avg", self.metrics["test/num_vocabs_avg"])
        self.metrics["test/vocabs_unique"](images_vocab)
        self.log("test/vocabs.unique", self.metrics["test/vocabs_unique"])
        self.metrics["test/vocabs/selected_unique"](sum([list(w.keys()) for w in words], []))
        self.log("test/vocabs/selected.unique", self.metrics["test/vocabs/selected_unique"])

        self.test_outputs.append((words, targets))

    def on_test_epoch_end(self) -> None:
        """Lightning hook called at the end of the test epoch."""
        words, targets = zip(*self.test_outputs)
        words = sum(words, [])
        targets = sum(targets, [])
        self.metrics["test/semantic_metrics"](words, targets)
        self.log_dict(self.metrics["test/semantic_metrics"])

        super().on_test_epoch_end()

    def configure_metrics(self) -> None:
        """Configure metrics."""
        self.metrics["test/num_vocabs_avg"] = MeanMetric()
        self.metrics["test/vocabs_unique"] = UniqueValues()
        self.metrics["test/vocabs/selected_unique"] = UniqueValues()

        semantic_metrics = {}
        semantic_cluster_acc = SemanticClusterAccuracy(task="multiclass", average="micro")
        semantic_metrics["test/semantic_cluster_acc"] = semantic_cluster_acc
        semantic_metrics["test/semantic_iou"] = SentenceIOU()
        semantic_metrics["test/semantic_similarity"] = SentenceScore()
        self.metrics["test/semantic_metrics"] = MetricCollection(semantic_metrics)
    
    def encode_vocabulary_meru(self, vocab: list[str], use_prompts: bool = False) -> torch.Tensor:
        """Encode a vocabulary.

        Args:
            vocab (list): List of words.
        """
        if vocab == self._prev_vocab_words and use_prompts == self._prev_used_prompts:
            return self._prev_vocab_words_z

        prompts = self.vocab_prompts if use_prompts else None
        # encode the text (tokenize and embed)
        texts_z_views = self.encode_text_meru(self.text_preprocess(vocab, prompts=prompts))

        # cache vocabulary
        self._prev_vocab_words = vocab
        self._prev_used_prompts = use_prompts
        self._prev_vocab_words_z = texts_z_views

        return texts_z_views
    
    def encode_text_meru(self, texts_views: list[list[str]]) -> torch.Tensor:
        """Tokenize and encode texts with the language encoder.

        Args:
            texts_views (list[list[str]]): List of texts to encode.
        """
        tokenized_texts_views = [ #! [750(n_words) 77(...)] Vs [2302] 3+ tokens per word <BOS> t1...tk <EOS>
            torch.cat([self.meru_tokenizer(text) for text in text_views]) for text_views in texts_views
        ]
        tokenized_texts_views = torch.stack(tokenized_texts_views).to(self.device)

        T, C, _ = tokenized_texts_views.shape
        texts_z_views = self.language_encoder(tokenized_texts_views.view(T * C, -1))
        texts_z_views = texts_z_views.view(T, C, -1)

        return texts_z_views


if __name__ == "__main__":
    _ = CaSED_MERU()