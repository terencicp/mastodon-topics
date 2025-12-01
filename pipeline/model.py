# Trains a BERTopic model on yesterday's preprocessed data
# Saves documents with topic assignments as CSV files

# Run with Python 3.11, install packages in requirements.txt
# Using Gemma embeddings requires Hugging Face login
# and approving ToS at https://huggingface.co/google/embeddinggemma-300m

import os
import json
from pathlib import Path
import pandas as pd
import numpy as np
from bertopic import BERTopic
from umap import UMAP
from sentence_transformers import SentenceTransformer
from huggingface_hub import login
from utils import setup_logging_format, get_date_twodaysago

os.environ['TOKENIZERS_PARALLELISM'] = 'false'

logger = setup_logging_format()

# Huggingface login for Gemma embeddings
token = os.getenv('HUGGINGFACE_TOKEN')
if token:
    login(token=token)
else:
    logger.warning('HUGGINGFACE_TOKEN environment variable not found.')

# Create embeddings with reduced memory usage
def create_embedding_model():
    os.environ['PYTORCH_MPS_HIGH_WATERMARK_RATIO'] = '0.0'
    embedding_model = SentenceTransformer('google/embeddinggemma-300m')
    original_encode = embedding_model.encode
    embedding_model.encode = lambda docs, **kw: original_encode(
        docs, **{**kw, 'batch_size': 8}
    )
    return embedding_model

# Create embedding model or load it from disk
def load_or_create_embeddings(texts, cache_path):
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    if cache_path.exists():
        logger.info('Loading cached embeddings.')
        embeddings = np.load(cache_path)
    else:
        logger.info('Creating new embeddings.')
        embedding_model = create_embedding_model()
        embeddings = embedding_model.encode(texts, show_progress_bar=True)
        np.save(cache_path, embeddings)
    return embeddings

# Train BERTopic model with optimized parameters
def train_model(texts, embeddings):
    logger.info('Training BERTopic model.')
    topic_model = BERTopic(
        umap_model=UMAP(
            n_components=10,
            n_neighbors=45,
            min_dist=0.3,
            metric='cosine',
            random_state=293
    ))
    topic_labels, _ = topic_model.fit_transform(texts, embeddings)
    return topic_model, topic_labels

# Get representative document rankings for all documents
def get_representative_docs_rank(topic_model, texts, embeddings, topic_labels, n=150):
    # Get representative docs from BERTopic
    representative_docs, _, _, _ = topic_model._extract_representative_docs(
        c_tf_idf=topic_model.c_tf_idf_,
        documents=pd.DataFrame({
            'Document': texts,
            'ID': range(len(texts)),
            'Topic': topic_labels
        }),
        topics=topic_model.topic_representations_,
        nr_repr_docs=n,
        nr_samples=len(texts)
    )
    # Representative docs ranks inside each topic
    rankings = [None] * len(texts)
    for topic_label, representative_docs in representative_docs.items():
        for rank, doc_text in enumerate(representative_docs, 1):
            rankings[texts.index(doc_text)] = rank
    return rankings

# Save statuses with topic labels to CSV (ignoring outliers)
def save_topics_csv(statuses, topic_labels, output_path, representative_rank):
    outlier = -1
    statuses_topics = statuses.copy()
    statuses_topics['topic'] = topic_labels
    statuses_topics['representative_rank'] = representative_rank
    statuses_topics = statuses_topics[statuses_topics['topic'] != outlier]
    output_path.parent.mkdir(parents=True, exist_ok=True)
    statuses_topics.to_csv(output_path, index=False)
    logger.info(f'Saved topics to {output_path}.')

# Data directories
data_path = Path('../data')
yesterday_dt, yesterday_ymd = get_date_twodaysago()

# Get all CSV files in documents folder
documents_dir = data_path / yesterday_ymd / 'documents'
csv_files = sorted(documents_dir.glob('*.csv'))
logger.info(f'Found {len(csv_files)} files to process for {yesterday_ymd}.')

# Process each file
for csv_path in csv_files:
    
    try:
        file_name = csv_path.stem
        logger.info(f'Processing "{file_name}.csv".')
        
        # Load preprocessed data
        statuses = pd.read_csv(csv_path)
        texts = statuses['text'].tolist()
        logger.info(f'Loaded {len(texts):,} documents.')
        
        # Load or create embeddings
        embeddings_cache_path = data_path / yesterday_ymd / 'embeddings' / f'{file_name}.npy'
        embeddings = load_or_create_embeddings(texts, embeddings_cache_path)
        
        # Train model
        topic_model, topic_labels = train_model(texts, embeddings)
        representative_rank = get_representative_docs_rank(topic_model, texts, embeddings, topic_labels)
        output_path = data_path / yesterday_ymd / 'topics' / f'{file_name}.csv'
        save_topics_csv(statuses, topic_labels, output_path, representative_rank)
        
    except Exception as e:
        logger.error(f'Error processing {csv_path.name}: {e}.')
        continue

logger.info('Model training complete.')
