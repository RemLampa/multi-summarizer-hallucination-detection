# ---
# jupyter:
#   jupytext:
#     cell_metadata_filter: -all
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.18.1
# ---

# %% [markdown]
# # Multi-Document Summarization Model
#
# This module defines a multi-document summarization model using a
# transformer-based architecture.
#
# The model is designed to generate concise summaries from multiple input
# documents.
#
# For this, we fine tune the BART model from
# [sshleifer/distilbart-cnn-12-6](https://huggingface.co/sshleifer/distilbart-cnn-12-6)
# which is a distilled version of
# [facebook/bart-large-cnn](https://huggingface.co/facebook/bart-large-cnn).
#
# We also use
# [all-MiniLM-L6-v2](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2)
# as the embedding model for the Maximal Marginal Relevance (MMR) based chunk
# selection during the context building stage of multi-document summarization and
# [alexfabbri/multi_news](https://huggingface.co/datasets/alexfabbri/multi_news)
# dataset for fine-tuning and evaluation.

# %% [markdown]
# ## Dependencies

# %% [markdown]
# ### Install required libraries (for Colab only)
# Uncomment if running in a Colab environment.
# %%
# %pip install -q --upgrade pip setuptools wheel
# Keep Colab-compatible torch stack
# %pip install -q \
#     torch==2.10.0 torchvision==0.25.0 torchaudio==2.10.0 \
#     --index-url https://download.pytorch.org/whl/cu128
# %pip install -q \
#     transformers==4.50.3 \
#     datasets==4.1.1 \
#     evaluate==0.4.3 \
#     accelerate==1.4.0 \
#     rouge-score==0.1.2 \
#     absl-py==2.1.0 \
#     sentence-transformers==5.2.3

# %% [markdown]
# ### Import necessary libraries
# %%
import re
from typing import List, cast

import evaluate
import numpy as np
import torch
from datasets import load_dataset
from evaluate import EvaluationModule
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from transformers import (
    AutoConfig,
    AutoModelForSeq2SeqLM,
    AutoTokenizer,
    BartForConditionalGeneration,
    BatchEncoding,
    DataCollatorForSeq2Seq,
    GenerationConfig,
    PreTrainedTokenizerBase,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
)
from transformers.configuration_utils import PretrainedConfig
from transformers.trainer_utils import EvalPrediction, get_last_checkpoint

# %% [markdown]
# ## Initialization and Setup

# %%
# Set dev mode flag
#
# In dev mode, we will use a smaller subset of the dataset for faster iterations
# during development.

# In production mode, we will use the full dataset for training and evaluation.
DEV_MODE = True

MODEL_NAME = "sshleifer/distilbart-cnn-12-6"
MAX_INPUT_LENGTH = 640
MAX_TARGET_SUMMARY_LENGTH = 256
FORCED_BOS_TOKEN_ID = 0
# for dislbart models, tying input and output word embeddings can save memory without hurting performance
TIE_WORD_EMBEDDINGS = True


# Set random seeds for reproducibility
def set_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


SEED = 42
set_seed(SEED)

if DEV_MODE:
    MODEL_OUTPUT_DIR = "../models/multi_doc_summarizer_dev"
    HF_CACHE = "../hf_cache"
else:
    # Production mode will be using Colab, save to Google Drive instead
    from google.colab import drive

    drive.mount("/content/drive")

    BASE = "/content/drive/MyDrive/rai8001"
    MODEL_OUTPUT_DIR = f"{BASE}/models/multi_doc_summarizer"
    HF_CACHE = f"{BASE}/hf_cache"

# Check for GPU availability
device = "cpu"

if torch.cuda.is_available():
    device = "cuda"

if torch.backends.mps.is_available():
    device = "mps"

print(f"Using device: {device}")

# %% [markdown]
# ### Clean up existing model output directory
#
# **!!!!!!DANGER!!!!!!**
#
# Uncomment the following code block to delete the existing model output
# directory before training.
# %%
# import os

# if os.path.exists(MODEL_OUTPUT_DIR):
#     import shutil

#     shutil.rmtree(MODEL_OUTPUT_DIR)


# %% [markdown]
# ## Context Builder for Stage A
# %%
class MultiDocContextBuilder:
    """
    Implements the context building logic for Stage A of the multi-document
    summarization process. It takes a list of input documents, splits them into
    smaller chunks, ranks the chunks using Maximal Marginal Relevance (MMR),
    and selects the top-ranked chunks that fit within the token budget to create
    a context string for Stage B summarization.
    """

    def __init__(
        self,
        tokenizer,
        token_budget: int,
        stage_a_embedder: SentenceTransformer | None = None,
        max_chars_per_chunk: int = 500,
        min_chars_per_chunk: int = 40,
    ):
        self.tokenizer = tokenizer
        self.token_budget = token_budget
        self.stage_a_embedder = (
            stage_a_embedder
            if stage_a_embedder is not None
            else SentenceTransformer("all-MiniLM-L6-v2", cache_folder=HF_CACHE)
        )
        self.max_chars_per_chunk = max_chars_per_chunk
        self.min_chars_per_chunk = min_chars_per_chunk

    @staticmethod
    def normalize_text(text: str) -> str:
        """
        Normalizes the input text by stripping whitespace and collapsing
        multiple newlines and spaces into single spaces.
        """
        text = text.strip()
        text = re.sub(r"\n+", " ", text)
        text = re.sub(r"\s+", " ", text)

        return text

    def split_raw_docs(self, text: str) -> List[str]:
        # The original dataset uses "|||||" as a delimiter between
        # source articles in the multi-document input.
        parts = re.split(r"\s*\|{3,}\s*", text.strip())

        return [self.normalize_text(part) for part in parts if part.strip()]

    def split_into_chunks(
        self,
        docs: List[str],
        max_chars_per_chunk: int = 500,
        min_chars_per_chunk: int = 40,
    ) -> List[str]:
        """Splits a list of documents into smaller chunks based on character count."""
        chunks: List[str] = []

        for d in docs:
            d = (d or "").strip()

            if not d:
                continue

            # Split the document into sentences using regex that looks for
            # sentence-ending punctuation followed by whitespace
            sentences = re.split(r"(?<=[.!?])\s+", d)

            cur = ""

            for s in sentences:
                s = s.strip()

                if not s:
                    continue

                # If adding the next sentence would not exceed the max
                # character limit, add it to the current chunk.
                if len(cur) + len(s) + 1 <= self.max_chars_per_chunk:
                    cur = f"{cur} {s}".strip()

                    continue

                # If the current chunk has enough characters,
                # add it to the list of chunks
                if len(cur) >= self.min_chars_per_chunk:
                    chunks.append(cur)

                # Start a new chunk with the current sentence
                cur = s

            if len(cur) >= self.min_chars_per_chunk:
                chunks.append(cur)

        return chunks

    def _select_chunks(
        self,
        chunks: List[str],
        k: int = 12,  # number of chunks to select
        lambda_rel: float = 0.7,  # relevance-diversity trade-off parameter for MMR
    ) -> List[str]:
        """
        Selects the top-k chunks based on Maximal Marginal Relevance (MMR) ranking.
        """
        if not chunks or len(chunks) <= k:
            return chunks

        embs = self.stage_a_embedder.encode(chunks, normalize_embeddings=True)

        centroid = embs.mean(axis=0, keepdims=True)

        # Compute relevance scores as cosine similarity to the centroid
        rel = cosine_similarity(embs, centroid).ravel()

        selected: List[int] = []

        candidates = list(range(len(chunks)))

        while candidates and len(selected) < k:
            best_i, best_score = -1, -1e9

            for i in candidates:
                diversity = 0.0

                if selected:
                    # Compute diversity as the maximum cosine similarity to already selected chunks
                    diversity = float(
                        np.max(cosine_similarity(embs[i : i + 1], embs[selected]))
                    )

                # Compute MMR score as a weighted combination of relevance and diversity
                score = lambda_rel * rel[i] - (1.0 - lambda_rel) * diversity

                if score > best_score:
                    best_i, best_score = i, score

            selected.append(best_i)

            candidates.remove(best_i)

        return [chunks[i] for i in selected]

    def build_context_from_docs(
        self,
        docs: List[str],
        token_budget: int = MAX_INPUT_LENGTH,
        max_chunks: int = 12,
        lambda_rel: float = 0.7,
        doc_sep_token: str = " <SEP> ",
    ) -> str:
        """Build a token-budgeted context string for Stage B summarization."""
        chunks = self.split_into_chunks(docs)
        ranked = self._select_chunks(chunks, k=max_chunks, lambda_rel=lambda_rel)

        kept: List[str] = []

        used = 0

        for c in ranked:
            # Calculate the number of tokens that would be added by including this chunk
            n = len(self.tokenizer.encode(c, add_special_tokens=False))

            # Calculate the number of tokens that would be added by including the separator
            # token if this is not the first chunk
            sep_n = (
                len(self.tokenizer.encode(doc_sep_token, add_special_tokens=False))
                if kept
                else 0
            )

            # If adding this chunk and the separator would exceed the token budget, skip it
            if used + sep_n + n > token_budget:
                continue

            # If we can include this chunk without exceeding the token budget,
            # add it to the list of kept chunks and update the token count
            kept.append(c)

            used += sep_n + n

        return doc_sep_token.join(kept) if kept else doc_sep_token.join(docs)

    def build_context_from_raw(
        self,
        raw_multi_doc: str,
        max_chunks: int = 12,
        lambda_rel: float = 0.7,
        doc_sep_token: str = " <SEP> ",
    ) -> str:
        """
        Convenience method that builds a context string for Stage B summarization
        directly from the raw multi-document input.
        """
        docs = self.split_raw_docs(raw_multi_doc)

        if not docs:
            return ""

        return self.build_context_from_docs(
            docs=docs,
            max_chunks=max_chunks,
            lambda_rel=lambda_rel,
            doc_sep_token=doc_sep_token,
        )


# %% [markdown]
# ## Dataset Preparation
# %%
class MultiNewsDataProcessor:
    """
    MultiNewsDataProcessor is a class that handles the loading and preprocessing
    of the multi-document summarization dataset. It takes care of normalizing the
    input documents and summaries, tokenizing them using the BART tokenizer, and
    preparing the data for training the model.
    """

    def __init__(
        self,
        tokenizer: PreTrainedTokenizerBase,
        context_builder: MultiDocContextBuilder,
        # the older version of the dataset is "alexfabbri/multi_news",
        # but it has compabitility issues with datasets 4
        # so we're using the updated one
        dataset_name: str = "Awesome075/multi_news",
        max_input_length: int = MAX_INPUT_LENGTH,
        max_target_summary_length: int = MAX_TARGET_SUMMARY_LENGTH,
        doc_sep_token: str = " <SEP> ",
    ):
        """Initializes the data processor for multi-document summarization."""
        self.dataset_name = dataset_name
        self.tokenizer = tokenizer
        self.max_input_length = max_input_length
        self.max_target_summary_length = max_target_summary_length
        self.doc_sep_token = doc_sep_token
        self.context_builder = context_builder or MultiDocContextBuilder(
            tokenizer=self.tokenizer,
            token_budget=max_input_length,
        )

    def _preprocess_batch(self, batch: dict) -> BatchEncoding:
        sources = [
            self.context_builder.build_context_from_raw(doc)
            for doc in batch["document"]
        ]

        targets = [
            self.context_builder.normalize_text(summary) for summary in batch["summary"]
        ]

        model_inputs = self.tokenizer(
            sources,
            max_length=self.max_input_length,
            padding=False,
            truncation=True,
        )

        labels = self.tokenizer(
            targets,
            max_length=self.max_target_summary_length,
            padding=False,
            truncation=True,
        )

        model_inputs["labels"] = labels["input_ids"]

        return model_inputs

    def load_and_preprocess_data(self, split: str | None = None) -> tuple:
        """
        Loads and preprocesses the multi-document summarization dataset.

        Args:
        split (str, optional):
            The dataset split(s) to load (e.g., 'train', 'validation', 'test').
            If None, the default split will be loaded. Defaults to None.
        """

        # Load the dataset and preprocess it based on the specified split
        if split is None:
            # Default to loading the entire dataset if no split is specified
            raw_dataset = load_dataset(self.dataset_name, cache_dir=HF_CACHE)
            base_cols = next(iter(raw_dataset.values())).column_names
            tokenized_dataset = raw_dataset.map(
                self._preprocess_batch, batched=True, remove_columns=base_cols
            )

            return raw_dataset, tokenized_dataset, self.tokenizer

        if isinstance(split, str):
            # If only a single split is specified, load and preprocess that split
            raw_dataset = load_dataset(
                self.dataset_name, split=split, cache_dir=HF_CACHE
            )
            base_cols = raw_dataset.column_names
            tokenized_dataset = raw_dataset.map(
                self._preprocess_batch, batched=True, remove_columns=base_cols
            )

            return raw_dataset, tokenized_dataset, self.tokenizer

        raise TypeError("Split must be a string or None.")


# %% [markdown]
# ## Model Definition
# %%
class MultiDocumentSummarizer:
    """
    MultiDocumentSummarizer is a class that defines a transformer-based model for
    multi-document summarization.
    """

    @staticmethod
    def _apply_stable_model_config(model: BartForConditionalGeneration) -> None:
        """Apply config values that must remain stable across all checkpoints."""
        if model.generation_config is None:
            model.generation_config = GenerationConfig.from_model_config(model.config)

        model.generation_config.forced_bos_token_id = FORCED_BOS_TOKEN_ID

    def __init__(
        self,
        tokenizer: PreTrainedTokenizerBase,
        context_builder: MultiDocContextBuilder,
        model_name: str = MODEL_NAME,
        max_input_length: int = MAX_INPUT_LENGTH,
    ):
        """Initializes the multi-document summarization model."""
        self.model_name: str = model_name
        self.tokenizer = tokenizer
        self.context_builder = context_builder
        cfg: PretrainedConfig = AutoConfig.from_pretrained(
            model_name, cache_dir=HF_CACHE
        )
        cfg.tie_word_embeddings = TIE_WORD_EMBEDDINGS
        self.model: BartForConditionalGeneration = (
            AutoModelForSeq2SeqLM.from_pretrained(
                model_name, config=cfg, cache_dir=HF_CACHE
            ).to(device)
        )
        self.model = cast(BartForConditionalGeneration, self.model)
        self._apply_stable_model_config(self.model)
        self.rouge_metric: EvaluationModule = evaluate.load("rouge")

        # We will use a separate sentence transformer model to compute embeddings
        # for the MMR-based chunk selection during inference. This allows us to
        # keep the summarization model focused on generation while leveraging a
        # lightweight embedding model for relevance and diversity scoring.
        self.stage_a_embedder: SentenceTransformer = SentenceTransformer(
            "all-MiniLM-L6-v2", cache_folder=HF_CACHE
        )

    def _compute_metrics(self, eval_preds: EvalPrediction) -> dict[str, float]:
        """Computes ROUGE metrics for evaluation."""
        predictions = eval_preds.predictions
        label_ids = eval_preds.label_ids

        # Some versions return tuple(preds, ...) instead of just preds
        if isinstance(predictions, tuple):
            predictions = predictions[0]

        # If predictions are logits, we need to convert them to token ids
        if isinstance(predictions, np.ndarray) and predictions.ndim == 3:
            predictions = np.argmax(predictions, axis=-1)

        # Ensure integer ids are in valid range for decoding
        predictions = np.asarray(predictions, dtype=np.int64)
        labels = np.asarray(label_ids, dtype=np.int64)

        vocab_size = self.tokenizer.vocab_size
        if not isinstance(vocab_size, int):
            raise ValueError(
                "Unexpected tokenizer vocab size type: {}".format(type(vocab_size))
            )

        vocab_max = vocab_size - 1
        predictions = np.clip(predictions, 0, vocab_max)

        # Replace ignore index (-100) in predictions with pad_token_id for decoding
        pad_id = self.tokenizer.pad_token_id
        if pad_id is None:
            pad_id = 0

        labels = np.where(labels != -100, labels, pad_id)
        labels = np.clip(labels, 0, vocab_max)

        decoded_preds = self.tokenizer.batch_decode(
            predictions, skip_special_tokens=True
        )
        decoded_labels = self.tokenizer.batch_decode(labels, skip_special_tokens=True)

        # Compute ROUGE scores
        rouge_scores = self.rouge_metric.compute(
            predictions=[pred.strip() for pred in decoded_preds],
            references=[label.strip() for label in decoded_labels],
            use_stemmer=True,
        )

        if rouge_scores is None:
            return {}

        return {k: round(float(v), 4) for k, v in rouge_scores.items()}

    def train(
        self,
        train_dataset: torch.utils.data.Dataset,
        eval_dataset: torch.utils.data.Dataset,
        output_dir: str = MODEL_OUTPUT_DIR,
        learning_rate: float = 2e-5,
        train_batch_size: int = 1,
        eval_batch_size: int = 2,
        num_train_epochs: int = 3,
        weight_decay: float = 0.01,
        logging_steps: int = 500,
        report_to: str = "none",
        fp16: bool = (device == "cuda"),  # enable only if using cuda for training
    ) -> Seq2SeqTrainer:
        data_collator = DataCollatorForSeq2Seq(
            self.tokenizer,
            model=self.model,
            pad_to_multiple_of=8 if fp16 else None,
        )

        args = Seq2SeqTrainingArguments(
            output_dir=output_dir,
            eval_strategy="epoch",
            save_strategy="epoch",
            learning_rate=learning_rate,
            per_device_train_batch_size=train_batch_size,
            per_device_eval_batch_size=eval_batch_size,
            num_train_epochs=num_train_epochs,
            weight_decay=weight_decay,
            logging_steps=logging_steps,
            save_total_limit=3,
            predict_with_generate=True,
            load_best_model_at_end=True,
            metric_for_best_model="eval_rougeL",
            greater_is_better=True,
            fp16=fp16,
            report_to=report_to,
            dataloader_pin_memory=(device == "cuda"),
            # Accumulate gradients over 8 steps to effectively increase
            # batch size without running out of memory
            gradient_accumulation_steps=8,
        )

        # Disable caching for training to save memory
        self.model.config.use_cache = False

        trainer = Seq2SeqTrainer(
            model=self.model,
            args=args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            data_collator=data_collator,
            compute_metrics=self._compute_metrics,
        )

        # resume if checkpoint exists in the output directory
        last_checkpoint = get_last_checkpoint(output_dir)
        if last_checkpoint is not None:
            print(f"Resuming training from checkpoint: {last_checkpoint}")
            trainer.train(resume_from_checkpoint=last_checkpoint)
        else:
            print("No checkpoint found. Starting training from scratch.")
            trainer.train()

        # re-enable caching after training
        self.model.config.use_cache = True

        return trainer

    def generate_summary(
        self,
        doc: str,
        max_new_tokens: int = 256,
        min_new_tokens: int = 30,
        num_beams: int = 4,
    ) -> str:
        """
        Generates a summary for the given input documents.

        Args:
            doc (str): The input document(s) to summarize. For multi-document summarization,
                multiple documents should be concatenated into a single string with a separator token (e.g., " <SEP> ").
            max_length (int): The maximum length of the generated summary. Defaults to 256.
            min_length (int): The minimum length of the generated summary. Defaults to 30.
            num_beams (int): The number of beams to use for beam search during generation. Defaults to 4.
        """
        encoding: BatchEncoding = self.tokenizer(
            doc,
            max_length=MAX_INPUT_LENGTH,
            truncation=True,
            return_tensors="pt",
        ).to(device)

        # Move input tensors to the same device as the model
        model_inputs: dict[str, torch.Tensor] = {
            k: v.to(device) for k, v in encoding.items()
        }

        input_ids = model_inputs["input_ids"]
        attention_mask = model_inputs["attention_mask"]

        self.model.eval()
        with torch.no_grad():
            summary_ids = self.model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_new_tokens=max_new_tokens,
                min_new_tokens=min_new_tokens,
                num_beams=num_beams,
                early_stopping=True,
            )

            decoded = self.tokenizer.decode(summary_ids[0], skip_special_tokens=True)

            if isinstance(decoded, list):
                decoded = " ".join(decoded)

            return decoded.strip()

    def batch_generate_summaries(
        self,
        docs: List[str],
        max_new_tokens: int = 256,
        min_new_tokens: int = 30,
        num_beams: int = 4,
    ) -> List[str]:
        """Generates summaries for a batch of input documents."""
        encoding: BatchEncoding = self.tokenizer(
            docs,
            max_length=MAX_INPUT_LENGTH,
            padding=True,
            truncation=True,
            return_tensors="pt",
        ).to(device)

        # Move input tensors to the same device as the model
        model_inputs: dict[str, torch.Tensor] = {
            k: v.to(device) for k, v in encoding.items()
        }

        input_ids = model_inputs["input_ids"]
        attention_mask = model_inputs["attention_mask"]

        self.model.eval()
        with torch.no_grad():
            raw_summaries = self.model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_new_tokens=max_new_tokens,
                min_new_tokens=min_new_tokens,
                num_beams=num_beams,
                early_stopping=True,
            )

            seqs = (
                raw_summaries
                if isinstance(raw_summaries, torch.Tensor)
                else raw_summaries.sequences
            )

            return self.tokenizer.batch_decode(seqs, skip_special_tokens=True)

    def generate_multi_doc_summary(
        self,
        docs: List[str],
        doc_sep_token: str = " <SEP> ",
        max_new_tokens: int = 256,
        min_new_tokens: int = 30,
        num_beams: int = 4,
    ) -> str:
        """
        Generates a single summary for multiple input documents.

        This method implements a two-stage approach to handle multi-document summarization
        within the token budget of the model. In Stage A, it splits the input documents
        into smaller chunks, ranks them based on relevance and diversity using Maximal Marginal
        Relevance (MMR), and selects the top-ranked chunks that fit within the token budget.

        In Stage B, it concatenates the selected chunks into a single context string and feeds it
        to the model to generate the final summary.
        """
        stage_a_context = self.context_builder.build_context_from_docs(
            docs=docs,
            doc_sep_token=doc_sep_token,
        )

        return self.generate_summary(
            stage_a_context,
            max_new_tokens=max_new_tokens,
            min_new_tokens=min_new_tokens,
            num_beams=num_beams,
        )

    def save(self, save_directory: str = MODEL_OUTPUT_DIR):
        """Saves the fine-tuned model and tokenizer to the specified output directory."""
        self.model.save_pretrained(save_directory)
        self.tokenizer.save_pretrained(save_directory)

    @classmethod
    def load(cls, model_directory: str = MODEL_OUTPUT_DIR):
        """Loads a fine-tuned model and tokenizer from the specified directory."""
        cfg = AutoConfig.from_pretrained(model_directory, cache_dir=HF_CACHE)
        cfg.tie_word_embeddings = TIE_WORD_EMBEDDINGS

        model = AutoModelForSeq2SeqLM.from_pretrained(
            model_directory, config=cfg, cache_dir=HF_CACHE
        ).to(device)
        cls._apply_stable_model_config(model)

        tokenizer = AutoTokenizer.from_pretrained(model_directory, cache_dir=HF_CACHE)

        context_builder = MultiDocContextBuilder(
            tokenizer=tokenizer,
            token_budget=MAX_INPUT_LENGTH,
        )

        summarizer = cls(
            tokenizer=tokenizer,
            context_builder=context_builder,
        ).__new__(cls)  # Create an uninitialized instance

        summarizer.model_name = model_directory
        summarizer.model = model
        summarizer.rouge_metric = evaluate.load("rouge")

        return summarizer


# %% [markdown]
# ## Run the Training and Evaluation Pipeline
# We need to wrap the training and evaluation code in a main guard to prevent it
# from executing when the module is imported, which allows us to reuse the
# `MultiDocumentSummarizer` class in other contexts without triggering the training
# pipeline automatically.
# %%
if __name__ == "__main__":

    def load_and_preprocess_data(
        tokenizer: PreTrainedTokenizerBase, context_builder: MultiDocContextBuilder
    ):
        multiNewsProcessor = MultiNewsDataProcessor(
            tokenizer=tokenizer,
            context_builder=context_builder,
        )

        if DEV_MODE:
            print(
                "DEV MODE: Using a smaller subset of the dataset for faster iterations."
            )
            train_raw, train_tokenized, tokenizer = (
                multiNewsProcessor.load_and_preprocess_data(
                    split="train[:100]"  # Use only the first 100 examples for training in dev mode
                )
            )
            val_raw, val_tokenized, _ = multiNewsProcessor.load_and_preprocess_data(
                split="validation[:1%]"
            )
            test_raw, test_tokenized, _ = multiNewsProcessor.load_and_preprocess_data(
                split="test[:1%]"
            )
        else:
            print(
                "PRODUCTION MODE: Using the full dataset for training and evaluation."
            )
            train_raw, train_tokenized, tokenizer = (
                multiNewsProcessor.load_and_preprocess_data(split="train")
            )
            val_raw, val_tokenized, _ = multiNewsProcessor.load_and_preprocess_data(
                split="validation"
            )
            test_raw, test_tokenized, _ = multiNewsProcessor.load_and_preprocess_data(
                split="test"
            )

        return (
            train_raw,
            train_tokenized,
            val_raw,
            val_tokenized,
            test_raw,
            test_tokenized,
        )

    def train_model(
        tokenizer: PreTrainedTokenizerBase, context_builder: MultiDocContextBuilder
    ):
        summarizer = MultiDocumentSummarizer(
            tokenizer=tokenizer,
            context_builder=context_builder,
        )

        print("train_tokenized", train_tokenized)

        if DEV_MODE:
            print(
                "DEV MODE: Training on a smaller subset of the dataset for faster iterations."
            )

            # In dev mode, we will train for only 1 epoch with a smaller batch size to
            # speed up the training process.
            trainer = summarizer.train(
                train_dataset=train_tokenized,
                eval_dataset=val_tokenized,
                num_train_epochs=1,
                train_batch_size=1,
                eval_batch_size=1,
                logging_steps=10,
            )
        else:
            print(
                "PRODUCTION MODE: Training on the full dataset for better performance."
            )

            trainer = summarizer.train(
                train_dataset=train_tokenized,
                eval_dataset=val_tokenized,
            )

            summarizer.save()  # Save the fine-tuned model after training
            trainer.save_state()

        print(
            "Training complete. Best model saved to:",
            trainer.state.best_model_checkpoint,
        )
        print("Best ROUGE-L score:", trainer.state.best_metric)
        return summarizer, trainer

    def generate_sample_summaries(summarizer: MultiDocumentSummarizer):
        print("Single document summary generation example:")
        print(
            "Generated summary:", summarizer.generate_summary(train_raw[0]["document"])
        )
        print("\n")
        print("Reference summary:", train_raw[0]["summary"])

        print("\n")
        print("Multi-document summary generation example:")

        # Each document is a concatenation of multiple news articles, separated by
        # "|||||" in the original dataset. We will split them into individual documents
        # and then generate a summary for the combined input.
        sample_docs = train_raw[0]["document"].split("|||||")
        for i, doc in enumerate(sample_docs):
            print(f"Document {i + 1}:", doc[:500], "...\n")

        sample_multi_doc_summary = summarizer.generate_multi_doc_summary(sample_docs)

        print("\n")
        print("Generated summary:", sample_multi_doc_summary)
        print("\n")
        print("Reference summary:", train_raw[0]["summary"])

        return sample_multi_doc_summary

    def evaluate_on_test_set(trainer):
        test_results = trainer.predict(test_tokenized, metric_key_prefix="test")

        print("Test set evaluation results:")

        if test_results.metrics is None:
            print("No metrics were computed during evaluation.")
        else:
            for metric_name, metric_value in test_results.metrics.items():
                print(f"{metric_name}: {metric_value:.4f}")

    # Run the preprocessing and training pipeline
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, cache_dir=HF_CACHE)

    context_builder = MultiDocContextBuilder(
        tokenizer=tokenizer,
        token_budget=MAX_INPUT_LENGTH,
    )

    (train_raw, train_tokenized, val_raw, val_tokenized, test_raw, test_tokenized) = (
        load_and_preprocess_data(tokenizer, context_builder)
    )

    summarizer, trainer = train_model(tokenizer, context_builder)

    sample_multi_doc_summary = generate_sample_summaries(summarizer)

    evaluate_on_test_set(trainer)
