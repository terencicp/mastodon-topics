# Push new data to GitHub

import requests
import subprocess
from pathlib import Path
from utils import setup_logging_format, get_date_twodaysago

logger = setup_logging_format()
_, yesterday_ymd = get_date_twodaysago()
repository = Path(__file__).parent.parent

try:
    subprocess.run(['git', 'add', '.'], cwd=repository, check=True)
    subprocess.run(['git', 'commit', '-m', f'Add data for {yesterday_ymd}'], cwd=repository, check=True)
    subprocess.run(['git', 'push'], cwd=repository, check=True)
    logger.info(f'Pushed data to GitHub.')
except subprocess.CalledProcessError as e:
    logger.error(f'Git command failed: {e}.')

try:
    requests.get('https://mastodon-topics.streamlit.app/', timeout=10)
except:
    pass