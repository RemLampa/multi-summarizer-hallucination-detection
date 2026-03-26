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

The app supports:

- Uploading up to **10** `.txt` files
- Switching between `v1` and `v2` model behavior
- Generating one combined summary

## Model Directory Configuration

Model paths can be configured via `.env` in the project root:

```env
MODEL_DIR_V1=/absolute/path/to/v1/model_or_parent_dir
MODEL_DIR_V2=/absolute/path/to/v2/model_or_parent_dir
```

If a path points to a parent directory, the app automatically resolves the latest `checkpoint-*` directory that contains `config.json`.

Default directories used when `.env` values are not set:

- `models/multi_doc_summarizer_dev_v1`
- `models/multi_doc_summarizer_dev_v2`

## Project Files

- `streamlit_app.py`: Streamlit UI
- `app/service.py`: model loading, variant routing, summarization service
- `scripts/summarization_model.py`: original notebook-style training script
- `scripts/summarization_model_v2.py`: refactored training/inference pipeline
