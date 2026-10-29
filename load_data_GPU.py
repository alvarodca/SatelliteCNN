import os
import re
import zipfile
from pathlib import Path
import pandas as pd
import requests
from dotenv import load_dotenv

BASE_DIR = Path(__file__).resolve().parent
ENV_PATH = BASE_DIR / '.env'

load_dotenv(dotenv_path=ENV_PATH, override=True)
GITHUB_TOKEN = os.getenv('GITHUB_TOKEN')

if (not GITHUB_TOKEN or not GITHUB_TOKEN.startswith(('ghp_', 'github_pat_'))) and ENV_PATH.is_file():
    text = ENV_PATH.read_text(encoding='utf-8', errors='ignore')
    match = re.search(r'(ghp_[A-Za-z0-9_]+|github_pat_[A-Za-z0-9_]+)', text)
    if match:
        GITHUB_TOKEN = match.group(1)

if not GITHUB_TOKEN:
    raise ValueError(f"Valid GITHUB_TOKEN starting with 'ghp_' or 'github_pat_' not found in {ENV_PATH}")

REPO = 'alvarodca/SatelliteCNN'
ASSET_NAME = 'db.zip'

SEED = 42
CLASS_NAMES = ['Water', 'Very Low', 'Low', 'Moderate', 'High', 'Very High', 'Non-burnable']
NUM_CLASSES = len(CLASS_NAMES)
MAX_TRAIN_SAMPLES = 0
NUM_WORKERS = 0
CUSTOM_EPOCHS = 5
FT_EPOCHS = 5
CUSTOM_LR = 1e-2
FT_LR = 1e-3

PIXEL_MEAN = [0.485, 0.456, 0.406]
PIXEL_STD = [0.229, 0.224, 0.225]

DATA_ARCHIVE = BASE_DIR / 'db.zip'
DATA_ROOT = BASE_DIR / 'data'
DB_DIR = DATA_ROOT / 'db'
OUTPUT_DIR = BASE_DIR / 'submissions'

if not DATA_ARCHIVE.is_file():
    headers = {
        'Authorization': f'token {GITHUB_TOKEN}',
        'Accept': 'application/vnd.github+json'
    }
    
    releases_url = f'https://api.github.com/repos/{REPO}/releases'
    response = requests.get(releases_url, headers=headers)
    
    if response.status_code == 404:
        raise PermissionError(
            f"GitHub API 404: Token lacks access to '{REPO}'. "
            "Ensure your token has 'repo' scope (Classic) or repository access with 'Contents: Read-only' (Fine-Grained)."
        )
    elif response.status_code != 200:
        raise RuntimeError(f"GitHub API Error {response.status_code}: {response.json().get('message')}")
        
    releases = response.json()
    asset = None
    for release in releases:
        for a in release.get('assets', []):
            if a['name'] == ASSET_NAME:
                asset = a
                break
        if asset:
            break
            
    if not asset:
        raise FileNotFoundError(f"Asset '{ASSET_NAME}' was not found in any release of {REPO}")
    
    dl_headers = {**headers, 'Accept': 'application/octet-stream'}
    with requests.get(asset['url'], headers=dl_headers, stream=True) as stream:
        stream.raise_for_status()
        with open(DATA_ARCHIVE, 'wb') as f:
            for chunk in stream.iter_content(chunk_size=8192):
                f.write(chunk)

required_csvs = [DB_DIR / f'{split}_reduced.csv' for split in ('train', 'val', 'test')]

if DATA_ARCHIVE.is_file():
    DATA_ROOT.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(DATA_ARCHIVE) as archive:
        for member in archive.infolist():
            if not member.filename.startswith('db/'):
                continue
            target = DATA_ROOT / member.filename
            if not member.is_dir() and target.is_file() and target.stat().st_size == member.file_size:
                continue
            if not target.resolve().is_relative_to(DATA_ROOT.resolve()):
                raise ValueError('The archive contains an invalid dataset path.')
            if member.is_dir():
                target.mkdir(parents=True, exist_ok=True)
            elif not target.is_file() or target.stat().st_size != member.file_size:
                archive.extract(member, DATA_ROOT)

if not all(path.is_file() for path in required_csvs):
    raise FileNotFoundError(
        'Dataset CSV files were not found. Start the kernel in the notebook directory '
        'or update DATA_ARCHIVE and DATA_ROOT in the setup cell.')

for csv_path in required_csvs:
    ids = pd.read_csv(csv_path, usecols=['id'], dtype={'id': str})['id']
    for image_id in ids:
        image_path = DB_DIR / 'images' / f'{image_id}.jpg'
        if not image_path.is_file() or image_path.stat().st_size == 0:
            raise FileNotFoundError(
                f'Missing or empty image: {image_path}. Supply the original db.zip '
                'and rerun this setup cell to restore it.')