from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path
from typing import Iterable, Literal, Protocol, cast

import torch
from dotenv import load_dotenv
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

from scripts.summarization_model_v2 import MultiDocumentSummarizer

REPO_ROOT = Path(__file__).resolve().parent.parent
load_dotenv(REPO_ROOT / ".env")

DEFAULT_MODEL_DIR_V1 = REPO_ROOT / "models" / "multi_doc_summarizer_dev_v1"
DEFAULT_MODEL_DIR_V2 = REPO_ROOT / "models" / "multi_doc_summarizer_dev_v2"

ModelVariant = Literal["v1", "v2"]


class SupportsSummarize(Protocol):
    """
    Protocol for summarizer implementations.

    Ensures both v1 and v2 summarizers can be used interchangeably in the app.
    """

    def generate_multi_doc_summary(self, docs: list[str]) -> str:
        """Generate a summary for multiple documents."""
        ...


class LegacyV1Summarizer:
    """
    App adapter for v1 behavior without importing scripts/summarization_model.py directly.

    The v1 module is Jupyter notebook-style and not safe for imports.
    """

    def __init__(self, model_dir: str):
        self.tokenizer = AutoTokenizer.from_pretrained(model_dir)

        self.model = AutoModelForSeq2SeqLM.from_pretrained(model_dir)

        self.max_input_length = 640

        self.device = (
            "cuda"
            if torch.cuda.is_available()
            else ("mps" if torch.backends.mps.is_available() else "cpu")
        )

        self.model.to(self.device)

    def _generate_summary(self, doc: str) -> str:
        inputs = self.tokenizer(
            doc,
            max_length=self.max_input_length,
            truncation=True,
            return_tensors="pt",
        ).to(self.device)

        self.model.eval()

        with torch.no_grad():
            summary_ids = self.model.generate(
                **inputs,
                max_new_tokens=256,
                min_new_tokens=30,
                num_beams=4,
                early_stopping=True,
            )

        return self.tokenizer.decode(summary_ids[0], skip_special_tokens=True)

    def generate_multi_doc_summary(self, docs: list[str]) -> str:
        combined = " <SEP> ".join(d.strip() for d in docs if d and d.strip())

        return self._generate_summary(combined)


def _resolve_effective_model_dir(base_dir: str) -> str:
    """
    Resolve to a loadable directory.

    Accept either a model root with config.json or a parent containing checkpoint-*.
    """
    base = Path(base_dir)
    if (base / "config.json").exists():
        return str(base)

    checkpoint_dirs = [
        p
        for p in base.glob("checkpoint-*")
        if p.is_dir() and p.name.split("-")[-1].isdigit()
    ]

    # Sort by checkpoint number to ensure the latest is last
    checkpoint_dirs.sort(key=lambda p: int(p.name.split("-")[-1]))

    if checkpoint_dirs:
        latest = checkpoint_dirs[-1]

        if (latest / "config.json").exists():
            return str(latest)

    raise FileNotFoundError(f"No valid models found in '{base_dir}'")


def _resolve_model_dir(model_variant: ModelVariant) -> str:
    specific = os.getenv(f"MODEL_DIR_{model_variant.upper()}")

    if specific:
        print(f"Using {model_variant} model from {specific}.")

        return _resolve_effective_model_dir(specific)

    print(f"Using {model_variant} model from default directory.")

    default_dir = (
        DEFAULT_MODEL_DIR_V1 if model_variant == "v1" else DEFAULT_MODEL_DIR_V2
    )

    return _resolve_effective_model_dir(str(default_dir))


@lru_cache(maxsize=2)  # Cache both v1 and v2 summarizer instances
def get_summarizer(model_variant: ModelVariant = "v2") -> SupportsSummarize:
    model_dir = _resolve_model_dir(model_variant)

    if model_variant == "v1":
        return LegacyV1Summarizer(model_dir)

    # For v2, we need to cast to the protocol type since MultiDocumentSummarizer
    # doesn't explicitly declare it even though we know it implements the
    # required method.
    return cast(SupportsSummarize, MultiDocumentSummarizer.load(model_dir))


def summarize_documents(docs: Iterable[str], model_variant: ModelVariant = "v2") -> str:
    cleaned = [d.strip() for d in docs if d and d.strip()]

    if not cleaned:
        raise ValueError("No valid input documents were provided.")

    if len(cleaned) > 10:
        raise ValueError("At most 10 documents are allowed.")

    summarizer = get_summarizer(model_variant)

    return summarizer.generate_multi_doc_summary(cleaned)
