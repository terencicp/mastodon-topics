# Connects to a MongoDB database with a collection named 'statuses'
# Processes yesterday's data for the specified timezone
# And saves it as CSV in the documents folder

# Run with Python 3.11, install packages in requirements.txt

import os
import re
import sys
import urllib.request
from datetime import datetime, timedelta, timezone
from zoneinfo import ZoneInfo
import pandas as pd
from bs4 import BeautifulSoup
import emoji
import fasttext
import spacy
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
from pathlib import Path
from utils import (
    setup_logging_format,
    get_mongodb,
    register_shutdown_handler,
    get_date_twodaysago,
    remove_similar_documents
)

TIMEZONES = [
    'Europe/Madrid',
    'America/New_York'
]

# Set up logging
logger = setup_logging_format()
logger.info('Setting up the environment.')

# Download FastText language identification model
fasttext_model_path = 'lid.176.bin'
if not Path(fasttext_model_path).exists():
    urllib.request.urlretrieve('https://dl.fbaipublicfiles.com/fasttext/supervised-models/lid.176.bin', fasttext_model_path)

# Load FastText model
fasttext_model = fasttext.load_model(fasttext_model_path)

# Connect to MongoDB
client, db = get_mongodb()
register_shutdown_handler(client)
collection = db['statuses']

# Get statuses in time window from MongoDB
def get_statuses_in_time_window(collection, start_datetime, end_datetime):
    statuses = collection.aggregate([
        {'$addFields': {'created_at_dt': {'$toDate': '$created_at'}}},
        {'$match': {
            'created_at_dt': {
                '$gt': start_datetime.astimezone(timezone.utc),
                '$lt': end_datetime.astimezone(timezone.utc)
            },
        }}
    ])
    return list(statuses)

# Aggregate all textual fields from statuses
def aggregate_textual_fields(statuses):
    rows = []
    for status in statuses:
        # Post with username mentions already removed
        text = status['content_nomention']
        # Add link preview
        card = status.get('card')
        if card:
            text += '. ' + card['title'] + '; ' + card['description']
        # Add poll options
        poll = status.get('poll')
        if poll:
            options = [option['title'] for option in poll['options']]
            text += '. ' + ', '.join(options)
        # Add hashtags if not already present
        for tag in status['tags']:
            if tag['name'].lower() not in text.lower():
                text += f". #{tag['name']} "
        rows.append({
            'id_hash': status['_id'],
            'in_reply_to_id_hash': status['in_reply_to_id_hash'],
            'account_id_hash': status['account_id_hash'],
            'reblogs_count': status['reblogs_count'],
            'text': text,
            'created_at': datetime.fromisoformat(status['created_at'])
        })
    return pd.DataFrame(rows)

# Text cleaning function
def clean_text(text):
    # Remove HTML links (except hashtags)
    soup = BeautifulSoup(text, 'html.parser')
    for a_tag in soup.find_all('a'):
        tag_classes = a_tag.get('class', []) if a_tag.attrs else []
        if 'hashtag' not in tag_classes:
            a_tag.decompose()
    # Strip HTML tags
    text = soup.get_text(separator=' ')
    # Remove text URLs
    text = re.sub(r'http\S+', '', text)
    # Remove Mastodon custom emojis
    text = re.sub(r':\w+:', '', text)
    # Remove emojis and meaningless characters
    text = emoji.replace_emoji(text, replace='')
    text = re.sub(r'[^\w\s.,!?;:\'’"\-&%()$€£]', ' ', text)
    # Normalize whitespace
    text = re.sub(r'\s+', ' ', text).strip()
    return text

# Merge author replies into root statuses
def merge_author_replies(statuses):
    def get_author_replies(status_id_hash, author_id_hash, statuses):
        direct_replies = statuses[
            (statuses['in_reply_to_id_hash'] == status_id_hash) & 
            (statuses['account_id_hash'] == author_id_hash)
        ]
        all_replies = []
        for _, reply in direct_replies.iterrows():
            all_replies.append(reply)
            all_replies.extend(get_author_replies(reply['id_hash'], author_id_hash, statuses))
        return all_replies
    # Statuses that are not replies
    root_statuses = statuses[statuses['in_reply_to_id_hash'].isna()].copy()
    # Merge each author's replies into their original status
    statuses_with_replies = []
    for _, status in root_statuses.iterrows():
        replies = get_author_replies(status['id_hash'], status['account_id_hash'], statuses)
        if replies:
            replies.sort(key=lambda status: status['created_at'])
            reply_texts = [r['text'] for r in replies]
            status['text'] = status['text'] + '. ' + '. '.join(reply_texts)
        statuses_with_replies.append(status)
    return pd.DataFrame(statuses_with_replies)

# Language detection
def is_english(text):
    predictions = fasttext_model.predict(text, k=1)
    language_code = predictions[0][0].replace('__label__', '')
    probability = predictions[1][0]
    return language_code == 'en' and probability >= 0.7

# Save statuses CSV in /data/yyyy-mm-dd/documents
def save_csv(statuses, date_str, timezone_str):
    output_dir = Path('../data') / date_str / 'documents'
    output_dir.mkdir(parents=True, exist_ok=True)
    filename = output_dir / f'{timezone_str.replace("/", "-")}.csv'
    statuses[['text', 'reblogs_count']].to_csv(filename, index=False)
    return filename

for tz in TIMEZONES:

    try:
        tz_info = ZoneInfo(tz)
        # Calculate yesterday's date range in the given timezone
        yesterday_dt, yesterday_ymd = get_date_twodaysago()
        start_datetime = yesterday_dt.replace(hour=0, minute=0, second=0, microsecond=0)
        start_datetime = datetime(
            yesterday_dt.year, yesterday_dt.month, yesterday_dt.day, 0, 0, 0,
            tzinfo=tz_info
        )
        end_datetime = start_datetime + timedelta(days=1)
        logger.info(f'Processing data for {yesterday_ymd} in timezone {tz_info}.')
        # Get statuses in the time window
        statuses_in_time_window = get_statuses_in_time_window(collection, start_datetime, end_datetime)
        logger.info(f'Found {len(statuses_in_time_window):,} statuses in the current date range.')
        # Aggregate all textual fields
        statuses = aggregate_textual_fields(statuses_in_time_window)
        # Text cleaning
        statuses['text'] = statuses['text'].apply(clean_text)
        # Add replies from the same author
        statuses = merge_author_replies(statuses)
        # Filter short texts
        statuses = statuses[statuses['text'].str.split().str.len() >= 10]
        # Keep only english texts
        statuses = statuses.loc[statuses['text'].apply(is_english)].copy()
        # Remove very similar texts
        statuses = remove_similar_documents(statuses, similarity_threshold=0.8)
        # Save as CSV
        filename = save_csv(statuses, yesterday_ymd, tz)
        logger.info(f'Saved {len(statuses):,} processed statuses to {filename}.')

    except Exception as e:
        logger.error(f'Error processing timezone {tz}: {e}')
        continue

client.close()

logger.info('Data preprocessing complete.')