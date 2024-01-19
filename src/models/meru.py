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
from src.utils.meru_utils import lorentz as L
from src.utils.meru_utils.image_traversal import interpolate, calc_scores, get_text_feats
from src.models.components.nn import Hyper_NearestNeighboursClassifier
from src.models.meru_backup import MERU

from yaml import safe_load

log = utils.get_logger(__name__)

DNAMES = {
    'Caltech101': 'caltech101',
    'DTD': 'dtd',
    'Flowers102': 'flowers102',
    'Food101': 'food101',
    'EuroSAT': 'eurosat',
    'FGVCAircraft': 'aircraft',
    'OxfordPets': 'pets',
    'SUN397': 'sun397'
}

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
        meru_for_classification = kwargs.get("use_meru_for_classification", False)
        meru_for_augmentation = kwargs.get("use_meru_for_augmentation", False)
        use_meru = meru_for_classification or meru_for_augmentation
        
        self.meru_classification = meru_for_classification
        self.meru_augmentation = meru_for_augmentation
        self.augment_image = kwargs.get("augment_img", False)
        self.augment_text = kwargs.get("augment_txt", False)

        if use_meru:
            self.use_meru_prompts = kwargs.get("use_meru_prompts", True)
            datasets_and_prompts_fp = kwargs["datasets_and_prompts"]
            with open(datasets_and_prompts_fp, "r") as f:
                self.datasets_and_prompts = safe_load(f)
            meru_config = LazyConfig.load(kwargs["meru_config"])
            meru_ckpt = kwargs["meru_ckpt"]
            self.meru = LazyFactory.build_model(meru_config, device).eval()
            CheckpointManager(model=self.meru).load(meru_ckpt)
            self.meru.eval()
            self.meru_tokenizer = Tokenizer() # needed for both classification and augmentation with text
            if meru_for_classification:
                tau = kwargs.get("tau", 1.0)
                sim = kwargs.get("similarity", "LIP")
                use_sofmax = kwargs.get("use_softmax", False)
                is_hyper = True if isinstance(self.meru, MERU) else False
                self.classifier = Hyper_NearestNeighboursClassifier(is_hyper=is_hyper, tau=tau, use_softmax=use_sofmax, similarity=sim)
            if meru_for_augmentation:
                if isinstance(self.meru, MERU):
                    self.root_feat = torch.zeros(512, device=device) #! find something like "self.meru.embed_dim" instead of hard-coded 512
                else:
                    # CLIP model checkpoint should have the 'root' embedding.
                    self.root_feat = torch.load(meru_ckpt)["root"].to(device) #! check if it does actually work
                self.text_pool, self.text_feats_pool = get_text_feats(self.meru) # Here we have the small MERU database of words, with their embeddings
                # Add [ROOT] to the pool of text feats.
                self.text_pool.append("[ROOT]") #! we do not need it
                self.text_feats_pool = torch.cat([self.text_feats_pool, self.root_feat[None, ...]])
                self.steps = kwargs.get("steps", 50)

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
        self, images_z: torch.Tensor, vocabularies: list[list],
        images
    ) -> tuple[torch.Tensor, list, list]:
        """Perform a single batch step.

        Args:
            images_z (torch.Tensor): Batch of image embeddings.
            vocabularies (list[list]): List of vocabularies (sentences) for each image.
            images (torch.Tensor): Batch of images.
        """
        if not self.meru_classification:
            # when using MERU for classification, we do not need to encode the sentences
            
            unfiltered_words = sum(vocabularies, [])

            # encode unfiltered words
            unfiltered_words_z = self.encode_vocabulary(unfiltered_words).squeeze(0)
            unfiltered_words_z = unfiltered_words_z / unfiltered_words_z.norm(dim=-1, keepdim=True)

            # generate a text embedding for each image from their unfiltered words
            #! We do not do this since MERU is only visual
            unfiltered_words_per_image = [len(vocab) for vocab in vocabularies]
            texts_z = torch.split(unfiltered_words_z, unfiltered_words_per_image)
            texts_z = torch.stack([word_z.mean(dim=0) for word_z in texts_z])
            texts_z = texts_z / texts_z.norm(dim=-1, keepdim=True)

        # filter the words and embed them. Optionally one can use MERU for augmenting the vocabulary
        if self.meru_augmentation and self.augment_image:
            encoded_images = self.meru.encode_image(images, project=True)
            new_words = self.image_traversal(encoded_images)
            vocabularies = [ db_sentences + traversal_sentences for db_sentences, traversal_sentences in zip(vocabularies, new_words)]

        vocabularies = self.vocab_transform(vocabularies)
        vocabularies = [vocab or ["object"] for vocab in vocabularies]
        words = sum(vocabularies, [])
        if self.meru_classification:
            words_z = self.encode_vocabulary_meru(words, use_prompts=True)
        else:
            words_z = self.encode_vocabulary(words, use_prompts=True)
            words_z = words_z / words_z.norm(dim=-1, keepdim=True)

        # create a one-hot relation mask between images and words
        words_per_image = [len(vocab) for vocab in vocabularies]
        col_indices = torch.arange(sum(words_per_image))
        row_indices = torch.arange(len(images_z)).repeat_interleave(torch.tensor(words_per_image))
        mask = torch.zeros(len(images_z), sum(words_per_image), device=self.device)
        mask[row_indices, col_indices] = 1

        if self.meru_classification:
            #! get hyperbolic image embeddings
            images_z = self.meru.encode_image(images, project=True)
            # get the image and text predictions
            if self.hparams.alpha != 0.0:
                images_p = self.classifier(images_z, words_z, mask=mask, curvature=self.meru.curv.exp())

            samples_p = images_p
        else:
            # get the image and text predictions
            images_p = self.classifier(images_z, words_z, mask=mask)
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

        self.eval_dataset = DNAMES[images_fp[0].split("/")[7]]

        # get vocabularies for each image
        images_z = self.vision_encoder(images) # [64, 768] # encode images with CLIP for retrieval
        images_vocab = self.vocabulary(images_z=images_z, images_fp=images_fp) # this only find the sentences

        if self.meru_augmentation and self.augment_text:
            new_words = self.text_traversal(images_vocab)
            images_vocab = [ db_sentences + traversal_sentences for db_sentences, traversal_sentences in zip(images_vocab, new_words)]


        # get predictions for each image
        images_p, words, images_vocab = self.batch_step(images_z, images_vocab, images=images)
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

        # prompts = self.vocab_prompts if use_prompts else None
        # Let's use the MERU prompts
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
        assert len(texts_views) == 1, "It should be a tuple-singleton [ERR: len(texts_views) != 1]"
        
        self.meru_prompts = self.datasets_and_prompts[self.eval_dataset] if self.use_meru_prompts else ["{}"]
        # Collect text features of each class.
        all_class_feats: list[torch.Tensor] = []
        
        for name in texts_views[0]:
            class_prompts = [_pt.format(name) for _pt in self.meru_prompts]

            class_prompt_tokens = self.meru_tokenizer(class_prompts)
            class_feats = self.meru.encode_text(class_prompt_tokens, project=False)

            # Ensamble in the tangent space, then project back to the hyperboloid
            class_feats = class_feats.mean(dim=0)
            class_feats = class_feats * self.meru.textual_alpha.exp()
            class_feats = L.exp_map0(class_feats, self.meru.curv.exp())
            
            all_class_feats.append(class_feats)
        all_class_feats = torch.stack(all_class_feats)

        return all_class_feats
    
    def image_traversal(self, images_feats: torch.Tensor) -> list[str]:
        """Image traversal.

        Args:
            images_fe
        """
        #! I expect the features to be of shape [bs, embedding_dim] (e.g. [64, 512])
        new_words = []
        for image_feats in images_feats:
            interpolated_feats = interpolate(self.meru, image_feats, self.root_feat, self.steps)
            NN_scores = calc_scores(self.meru, interpolated_feats, self.text_feats_pool, has_root=True)

            NN_scores, NN_idxs = NN_scores.max(dim=-1)
            NN_texts = [self.text_pool[idx.item()] for idx in NN_idxs]

            unique_NN_texts = list(set(NN_texts)) # this is not sorted
            unique_NN_texts = [sentence for sentence in unique_NN_texts if sentence != "[ROOT]"] # remove the root
            new_words.append(unique_NN_texts)

        assert len(new_words) == len(images_feats), "The number of images and the number of sets of new words do not match"

        return new_words
    
    def text_traversal(self, texts: torch.Tensor) -> list[str]:
        """Image traversal.

        Args:
            text_fe
        """
        # we need to perform tokenization and embedding first

        #! I expect the features to be of shape [bs, embedding_dim] (e.g. [64, 512])
        new_words = []
        for txt in texts:
            new_words_per_image = []
            captions_tokens = self.meru_tokenizer(txt) #! tokens of all the sentences related to the image
            captions_feats = self.meru.encode_text(captions_tokens, project=True) #! [num_sentences, embedding_dim]
            
            for txt_feats in captions_feats:
                interpolated_feats = interpolate(self.meru, txt_feats, self.root_feat, self.steps)
                NN_scores = calc_scores(self.meru, interpolated_feats, self.text_feats_pool, has_root=True)

                NN_scores, NN_idxs = NN_scores.max(dim=-1)
                NN_texts = [self.text_pool[idx.item()] for idx in NN_idxs]

                unique_NN_texts = list(set(NN_texts)) # this is not sorted
                unique_NN_texts = [sentence for sentence in unique_NN_texts if sentence != "[ROOT]"] # remove the root
                new_words_per_image.append(unique_NN_texts)
            new_words.append(sum(new_words_per_image, []))

        assert len(new_words) == len(texts), "The number of new_texts and the number of sets of new words do not match"

        return new_words


if __name__ == "__main__":
    _ = CaSED_MERU()