import json
from pathlib import Path
from datetime import datetime

# Get last 7 days from folder names
def get_available_dates():
    data_path = Path('data')
    dates = [d.name for d in data_path.iterdir() if d.is_dir()]
    return sorted(dates, reverse=True)[:7]

# Get files for a date
def get_files_for_date(date):
    summaries_path = Path(f'data/{date}/summaries')
    return list(summaries_path.glob('*.json'))

# Extract unique timezones from files
def get_timezones(date):
    files = get_files_for_date(date)
    timezones = set()
    for file in files:
        parts = file.stem.split('-')
        timezone = f'{parts[0]}/{parts[1]}'
        timezones.add(timezone)
    return sorted(timezones, reverse=True)

# Load JSON file
def load_json(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        return json.load(file)

# Format date for display
def format_date(date_str):
    dt = datetime.strptime(date_str, '%Y-%m-%d')
    return dt.strftime('%B %d, %Y')

# Build file path from dropdown selections
def build_file_path(date, timezone):
    tz_parts = timezone.split('/')
    filename = f'{tz_parts[0]}-{tz_parts[1]}.json'
    return Path(f'data/{date}/summaries/{filename}')
