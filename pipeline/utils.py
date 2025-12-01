import logging
import signal
import sys
from datetime import datetime, timedelta
from pymongo import MongoClient
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

MONGODB_URI = 'mongodb://localhost:27017/'
DB_NAME = 'mastodon_production'

def setup_logging_format():
    logging.basicConfig(
        level=logging.INFO,
        format='[%(asctime)s] %(levelname)s: %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    return logging.getLogger(__name__)

def get_mongodb():
    client = MongoClient(MONGODB_URI)
    db = client[DB_NAME]
    return client, db

def register_shutdown_handler(client):
    def shutdown_db(sig, frame):
        client.close()
        sys.exit(0)
    signal.signal(signal.SIGINT, shutdown_db)
    signal.signal(signal.SIGTERM, shutdown_db)

def get_date_twodaysago():
    yesterday_dt = datetime.now() - timedelta(days=2)
    yesterday_ymd = yesterday_dt.strftime('%Y-%m-%d')
    return yesterday_dt, yesterday_ymd

def remove_similar_documents(statuses, similarity_threshold, sort_by='reblogs_count'):
    statuses = statuses.sort_values(sort_by, ascending=False).reset_index(drop=True)
    vectorizer = TfidfVectorizer(ngram_range=(1, 2))
    tfidf_matrix = vectorizer.fit_transform(statuses['text'])
    similarity_matrix = cosine_similarity(tfidf_matrix)
    np.fill_diagonal(similarity_matrix, 0)
    to_keep = []
    for i in range(len(statuses)):
        if i == 0 or not any(similarity_matrix[i, j] > similarity_threshold for j in to_keep):
            to_keep.append(i)
    return statuses.iloc[to_keep].reset_index(drop=True)
