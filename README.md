# Mastodon topic modeling

An automated pipeline for identifying and summarizing the most popular topics of daily discourse on Mastodon using advanced NLP techniques.

## Overview

This project implements an end-to-end pipeline for discovering and summarizing trending topics on the Mastodon social network. It combines BERTopic for topic modeling with large language models for generating interpretable summaries, providing daily insights into what the Mastodon community is discussing.

## Architecture

![Pipeline diagram](./pipeline-diagram.svg)

The system consists of five main components:

1. **collect.py** - Data collection
   - Fetches public posts from mastodon.social API
   - Anonymizes data and stores it in MongoDB

2. **preprocess.py** - Data preprocessing
   - Filters english-language posts
   - Aggregates textual content from multiple fields

3. **model.py** - Topic modeling
   - Generates Gemma-300m embeddings
   - Trains BERTopic models with optimized parameters

4. **summarize.py** - LLM summarization
   - Selects top topics by popularity
   - Generates structured summaries using Qwen3 8b

5. **streamlit.py** - Streamlit app
   - Displays topic summaries

## Requirements

### Dependencies
- Python 3.11
- Pipeline python dependencies listed in pipeline/requirements.txt
- Streamlit python dependencies listed in app/requirements.txt
- MongoDB
- Ollama

### Environment variables
- MASTODON_HASHING_KEY for data anonymization
- HUGGINGFACE_TOKEN from your HuggingFace account for Gemma embeddings

## Acknowledgments

This project builds upon the work of several amazing open-source projects and research teams:

- **[Mastodon](https://joinmastodon.org/)**: decentralized social network
- **[Gemma embeddings](https://ai.google.dev/gemma)**: embedding models from Google DeepMind
- **[BERTopic](https://maartengr.github.io/BERTopic/)**: topic modeling framework by Maarten Grootendorst
- **[Qwen3](https://qwen.ai/)**: LLM from Alibaba Cloud

Please make a donation to Mastodon if you use its API to offset the server costs.

## Contact

LinkedIn: [Terenci Claramunt](https://www.linkedin.com/in/terenci/)

This project was developed as part of my Applied Data Science Degree thesis.

Feel free to reach out for questions about the methodology or implementation.
