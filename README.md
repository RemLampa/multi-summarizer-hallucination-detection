# Multi-Document Summarizer

NLP project for multi-document summarization with two serving variants:

- `v1`: direct concatenation summarization
- `v2`: Stage-A context selection + Stage-B summarization

## Requirements

- Python 3.11+
- Dependencies in `requirements.txt`

## Install

```bash
pip install -r requirements.txt
```

## Run Web App

```bash
streamlit run streamlit_app.py
```

The app accepts up to **10** `.txt` files and can switch between `v1` and
`v2` summarization behavior.

## Training Dataset and Foundational Models

### Training Dataset

- Primary dataset: [`Awesome075/multi_news_parquet`][multi_news_parquet]
  which is a parquet version of [`alexfabbri/multi_news`][multi-news]
- Task format: each example contains multiple related news documents and one
  reference summary
- Usage:
  - `v1`: direct multi-document concatenation + seq2seq summarization
  - `v2`: Stage-A context selection + Stage-B seq2seq summarization

### Foundational Models

- Stage-B summarization backbone (v1 and v2):
  [`sshleifer/distilbart-cnn-12-6`][distilbart]
  - Distilled BART model used as the generation model and fine-tuned in this
    project
- Stage-A semantic selection model (v2):
  [`sentence-transformers/all-MiniLM-L6-v2`][minilm]
  - Used for embedding-based relevance/diversity scoring during context
    construction

## Model Training

Train either variant directly from the project root:

```bash
python scripts/summarization_model_v1.py
python scripts/summarization_model_v2.py
```

Each script loads `alexfabbri/multi_news`, tokenizes train/validation/test
splits, then fine-tunes the configured backbone.

## Model Directory Configuration

Model paths can be configured via `.env` in the project root:

```env
MODEL_DIR_V1=/absolute/path/to/v1/model_or_parent_dir
MODEL_DIR_V2=/absolute/path/to/v2/model_or_parent_dir
```

If a path points to a parent directory, the app automatically resolves the
`checkpoint-*` directory that contains `config.json`.

Default directories used when `.env` values are not set:

- `models/multi_doc_summarizer_dev_v1`
- `models/multi_doc_summarizer_dev_v2`

## Project Files

- `streamlit_app.py`: Streamlit UI
- `app/service.py`: model loading, variant routing, summarization service
- `scripts/`: training pipelines for `v1` and `v2`
- `notebooks/`: Jupyter training/dev notebooks (via
  [Jupytext](https://jupytext.readthedocs.io/en/latest/))

[multi_news_parquet]: https://huggingface.co/datasets/Awesome075/multi_news_parquet
[multi-news]: https://huggingface.co/datasets/alexfabbri/multi_news
[distilbart]: https://huggingface.co/sshleifer/distilbart-cnn-12-6
[minilm]: https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2
