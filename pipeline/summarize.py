# Generates JSON summaries for yesterday's topics using an LLM

# Run with Python 3.11, install packages in requirements.txt
# Requires installing Ollama from https://ollama.com/

import os
import json
import subprocess
import time
from pathlib import Path
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from ollama import chat
from pydantic import BaseModel
import tiktoken
from utils import setup_logging_format, get_date_twodaysago, remove_similar_documents

# Topic selection
PROPORTION_OF_TOPIC_DOCS_COVERED = 0.4

# Topic document selection
MAX_POPULAR_DOC_SIMILARITY = 0.45
MAX_REPRESENTATIVE_DOC_SIMILARITY = 0.1
MAX_DOCS = 100
MAX_CHARS_PER_DOC = 1000

logger = setup_logging_format()

tokenizer = tiktoken.get_encoding('o200k_base')

# Start Ollama server if not running
try:
    subprocess.run(['pgrep', '-x', 'ollama'], check=True, capture_output=True)
except subprocess.CalledProcessError:
    logger.info('Starting Ollama server.')
    subprocess.Popen(['ollama', 'serve'], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    time.sleep(5)

# Pydantic schema for structured LLM output
class TopicSection(BaseModel):
    section_title: str
    section_content: str
class TopicSummary(BaseModel):
    title: str
    summary: str
    description: list[TopicSection]

# Generate LLM summary for a topic
def generate_llm_summary(docs):
    formatted_docs = '\n'.join([f'- {doc}' for i, doc in enumerate(docs)])
    _, yesterday_ymd = get_date_twodaysago()
    
    prompt = f"""
This subset of user posts from a social network have been clustered as a topic.
Your task is to provide a thoughtful in-depth analysis of its content to help
readers understand the general themes of the messages.
Your summary for this topic and others will be shown in a page titled:
'What are people talking about today?'

Topic user posts:

{formatted_docs}

Answer only with valid JSON including the fields:
* title: Concise title describing the topic.
* summary: Short paragraph describing the content of the topic.
* description: Array of sections summarizing different aspects of the topic.
  It may have one to four sections depending on topic heterogeneity.
  Each section is an object with section_title and section_content."""

    logger.info(f'Prompt tokens: {len(tokenizer.encode(prompt)):,}')
    
    for attempt in range(2):
        try:
            response = chat(
                model='qwen3:8b',
                messages=[{'role': 'user', 'content': prompt}],
                format=TopicSummary.model_json_schema(),
                think=True,
                options={
                    'temperature': 0.6,
                    'top_p': 0.95,
                    'top_k': 20,
                    'min_p': 0,
                    'num_predict': 32768
                }
            )
            summary = TopicSummary.model_validate_json(response.message.content)
            return summary
        except Exception as e:
            if attempt == 1:
                logger.error(f'LLM call failed after 2 attempts: {e}')
                raise

# Select popular and representative documents to summarize
def select_documents_to_summarize(topic_statuses):
    selected_docs = pd.DataFrame()
    # Select popular documents
    popular = topic_statuses[topic_statuses['reblogs_count'] > 0].copy()
    popular_distinct = remove_similar_documents(popular, MAX_POPULAR_DOC_SIMILARITY)
    selected_docs = pd.concat([selected_docs, popular_distinct])
    logger.info(f'Filtered {len(popular_distinct)} of {len(popular)} reblogged statuses.')
    # Select representative documents not already included
    representative = topic_statuses[topic_statuses['representative_rank'].notna()].copy()
    representative_distinct = remove_similar_documents(
        representative[~representative.index.isin(selected_docs.index)],
        MAX_REPRESENTATIVE_DOC_SIMILARITY,
        sort_by='representative_rank'
    )
    selected_docs = pd.concat([selected_docs, representative_distinct])
    # Limit documents
    return selected_docs.head(MAX_DOCS)

# Summarize a single topic
def summarize_topic(topic_statuses):
    if len(topic_statuses) > MAX_DOCS:
        topic_statuses = select_documents_to_summarize(topic_statuses)
    selected_texts_trimmed = topic_statuses['text'].apply(lambda x: x[:MAX_CHARS_PER_DOC])
    return generate_llm_summary(selected_texts_trimmed.tolist())

# Dictionary for JSON output
def create_topic_output(summary, topic_stats, topic_id):
    return {
        'topic_id': topic_id,
        'title': summary.title,
        'summary': summary.summary,
        'description': [
            {'section_title': s.section_title, 'section_content': s.section_content}
            for s in summary.description
        ],
        'total_documents': int(topic_stats.loc[topic_id, 'doc_count']),
        'total_reblogs': int(topic_stats.loc[topic_id, 'total_reblogs'])
    }

# Add topic number of documents and popularity
def add_topic_stats(statuses_with_topics):
    topic_stats = statuses_with_topics.groupby('topic').agg(
        doc_count=('topic', 'size'),
        total_reblogs=('reblogs_count', 'sum')
    )
    topic_stats['popularity'] = topic_stats['doc_count'] * topic_stats['total_reblogs']
    return topic_stats

# Select the most popular topics, assuring a minimum document coverage
def select_topics_by_coverage(topic_stats):
    sorted_topics = topic_stats.sort_values('popularity', ascending=False)
    total_docs = sorted_topics['doc_count'].sum()
    cumulative_coverage = (sorted_topics['doc_count'].cumsum() / total_docs)
    exceeds_threshold = (cumulative_coverage >= PROPORTION_OF_TOPIC_DOCS_COVERED)
    selected_topics = sorted_topics.iloc[:exceeds_threshold.argmax()+1]
    return selected_topics.index.tolist()

# Data directories
data_path = Path('../data')
_, yesterday_ymd = get_date_twodaysago()
output_dir = data_path / yesterday_ymd / 'summaries'
output_dir.mkdir(parents=True, exist_ok=True)

# Get all CSV files in topics folder
topics_dir = data_path / yesterday_ymd / 'topics'
csv_files = sorted(topics_dir.glob('*.csv'))
logger.info(f'Found {len(csv_files)} files to process for {yesterday_ymd}.')

for i, csv_path in enumerate(csv_files, 1):

    logger.info(f'Processing file {i}/{len(csv_files)}: "{csv_path.name}".')
    
    try:
        # Check if CSV file has already been summarized
        output_path = output_dir / f'{csv_path.stem}.json'
        if output_path.exists():
            logger.info(f'Skipping {csv_path.name}, summary already exists.')
            continue
        
        # Load data with topic assignments
        statuses = pd.read_csv(csv_path)
        logger.info(f'Loaded {len(statuses):,} documents.')
        
        # Add topic number of documents and popularity
        topic_stats = add_topic_stats(statuses)
        
        # Select N most popular topics to summarize
        selected_topics = select_topics_by_coverage(topic_stats)
        logger.info(f'Selected {len(selected_topics)} topics for summarization.')
        
        # Generate summaries for the selected topics
        summaries = {}
        for rank, topic_id in enumerate(selected_topics, 1):
            logger.info(f'Summarizing topic {rank}/{len(selected_topics)}.')
            topic_statuses = statuses[statuses['topic'] == topic_id].copy()
            summary = summarize_topic(topic_statuses)
            summaries[rank] = create_topic_output(summary, topic_stats, topic_id)
        
        # Save JSON to summaries folder
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(summaries, f, ensure_ascii=False, indent=2)
        logger.info(f'Saved summaries to {output_path}.')
        
    except Exception as e:
        logger.error(f'Error processing "{csv_path.name}": {e}.')
        continue

logger.info('Summarization complete.')
