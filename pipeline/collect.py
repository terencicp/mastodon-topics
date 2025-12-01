# Fetch statuses from the Mastodon API that are at least 1 day old,
# filter and anonymize them and store them in MongoDB

# Run with Python 3.11, install packages in requirements.txt

# Requires "min_id", a mastodon.social status id that is ~1 day old
# that can be passed as command-line argument
# or from the last seen value stored in the db

import time
import requests
import hmac
import hashlib
import os
import sys
from datetime import datetime, timezone, timedelta
from bs4 import BeautifulSoup
from utils import setup_logging_format, get_mongodb, register_shutdown_handler

# Set up logging
logger = setup_logging_format()

# Get hashing key from environment
HASHING_KEY = os.environ.get('MASTODON_HASHING_KEY')
if not HASHING_KEY:
    raise ValueError('MASTODON_HASHING_KEY environment variable not set.')

# Connect to MongoDB
client, db = get_mongodb()
register_shutdown_handler(client)

# Start collecting data at min_id
if len(sys.argv) > 1:
    # Get min_id from the command line arguments
    min_id = sys.argv[1]
    db['min_id'].replace_one({}, {'id': min_id}, upsert=True)
else:
    # Get last seen min_id from db
    min_id_doc = db['min_id'].find_one()
    if not min_id_doc or 'id' not in min_id_doc:
        raise ValueError('No min_id found in db. Run: python collect.py [status_id]')
    min_id = min_id_doc['id']
logger.info(f'Inital id: {min_id}.')

# Fetch statuses created after the given status id from the API
def fetch_timeline(min_id):
    params = {
        'limit': 40,
        'min_id': min_id
    }
    try:
        timeline = 'https://mastodon.social/api/v1/timelines/public'
        response = requests.get(timeline, params=params, timeout=10)
        response.raise_for_status()
        return response.json()
    except Exception as error:
        logger.error(f'Error fetching data: {error}')
        return []

# Get the highest id and its created_at timestamp
def get_latest_min_id(current_min_id, statuses):
    next_min_id = current_min_id
    latest_created_at = None
    for status in statuses:
        if int(status['id']) > int(next_min_id):
            next_min_id = status['id']
            latest_created_at = datetime.fromisoformat(status['created_at'])
    # Store the last min_id in the database
    db['min_id'].replace_one({}, {'id': next_min_id}, upsert=True)
    return next_min_id, latest_created_at

# Choose which statuses to keep
def filter_statuses(statuses):
    filtered = []
    for status in statuses:
        # Keep only statuses from the instance mastodon.social
        if not status['uri'].startswith('https://mastodon.social/'):
            continue
        # Keep only statuses with content (no reblogs)
        if status['content'] == '':
            continue
        # Exclude automated posts
        if status['account']['bot']:
            continue
        filtered.append(status)
    return filtered

# Remove user mentions from content
def remove_mentions(statuses):
    for status in statuses:
        status['mentions_count'] = len(status['mentions'])
        if not status['mentions']:
            status['content_nomention'] = status['content']
        else:
            soup = BeautifulSoup(status['content'], 'html.parser')
            for mention_link in soup.find_all('a', class_='mention'):
                mention_link.decompose()
            status['content_nomention'] = str(soup)
    return statuses

# Anonymize user and status ids
def hash_ids(statuses):
    def anonymize(id):
        if not id:
            return None
        hash = hmac.new(HASHING_KEY.encode('utf-8'), id.encode('utf-8'), hashlib.sha256)
        return hash.hexdigest()[:18]
    for status in statuses:
        status['id_hash'] = anonymize(status['id'])
        status['account_id_hash'] = anonymize(status['account']['id'])
        status['in_reply_to_id_hash'] = anonymize(status['in_reply_to_id'])
    return statuses

# Keep only the data required for topic modeling
def select_fields(statuses):
    statuses_subset = []
    for status in statuses:
        statuses_subset.append({
            '_id': status['id_hash'],
            'in_reply_to_id_hash': status['in_reply_to_id_hash'],
            'account_id_hash': status['account_id_hash'],
            'created_at': status['created_at'],
            'content_nomention': status['content_nomention'],
            'card': status['card'],
            'poll': status['poll'],
            'tags': status['tags'],
            'reblogs_count': status['reblogs_count']
        })
    return statuses_subset

# Store processed statuses to MongoDB
def store_statuses(db, statuses):
    for status in statuses:
        try:
            db['statuses'].update_one(
                {'_id': status['_id']},
                {'$set': status},
                upsert=True
            )
        except Exception as e:
            logger.error(f'Error saving to MongoDB: {e}')
            continue

# Fetch statuses from the Mastodon API continuously
while True:

    try:
        statuses = fetch_timeline(min_id)
        min_id, last_status_datetime = get_latest_min_id(min_id, statuses)
        age = datetime.now(timezone.utc) - last_status_datetime
        if age >= timedelta(days=1):
            statuses = filter_statuses(statuses)
            statuses = remove_mentions(statuses)
            statuses = hash_ids(statuses)
            statuses = select_fields(statuses)
            store_statuses(db, statuses)
            logger.info(f'Saved {len(statuses)} statuses')
            time.sleep(2)
        else:
            logger.info(f'Last status is <1 day old, waiting.')
            time.sleep(30)

    except Exception as e:
        logger.error(f'Unexpected error: {e}')
        time.sleep(60)
