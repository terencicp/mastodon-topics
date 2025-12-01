# Pipeline orchestrator, runs with 1 retry. Run using:
# launchctl load ~/Library/LaunchAgents/com.terencicp.pipeline.plist

import subprocess
import sys
from pathlib import Path
from utils import setup_logging_format

logger = setup_logging_format()

SCRIPTS = ['preprocess.py', 'model.py', 'summarize.py']

def run_script(script_name):

    logger.info(f'Running {script_name}.')
    script_path = Path(__file__).parent / script_name
    
    try:
        subprocess.run([sys.executable, str(script_path)], check=True)
        return True
        
    except subprocess.CalledProcessError:
        logger.error(f'{script_name} failed, retrying.')
        try:
            subprocess.run([sys.executable, str(script_path)], check=True)
            return True
        except subprocess.CalledProcessError:
            logger.error(f'{script_name} failed after retry, stopping.')
            return False

for script in SCRIPTS:
    if not run_script(script):
        sys.exit(1)
