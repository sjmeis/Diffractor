import requests
from pathlib import Path
from tqdm import tqdm

# Registry of your hosted filtered files
REMOTE_ASSETS = {
    "conceptnet-numberbatch-19-08-300": "https://zenodo.org/records/19701515/files/numberbatch-en-19.08_filtered.txt?download=1",
    "glove-twitter-200": "https://zenodo.org/records/19701515/files/glove.twitter.27B.200d_filtered.txt?download=1",
    "glove-wiki-gigaword-300": "https://zenodo.org/records/19701515/files/glove.6B.300d_filtered.txt?download=1",
    "glove-commoncrawl-30": "https://zenodo.org/records/19701515/files/glove.840B.300d_filtered.txt?download=1",
    "word2vec-google-news-300": "https://zenodo.org/records/19701515/files/GoogleNews-vectors-negative300_filtered.txt?download=1"
}

class Downloader:
    def __init__(self, cache_dir: str = None):
        self.cache_dir = Path(cache_dir or Path.home() / ".cache" / "diffractor")
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    def fetch(self, model_name: str) -> Path:
        if model_name not in REMOTE_ASSETS:
            raise ValueError(f"Unknown model: {model_name}. Available: {list(REMOTE_ASSETS.keys())}")
        
        file_path = self.cache_dir / f"{model_name}.txt"
        
        if not file_path.exists():
            print(f"Downloading {model_name} to {file_path}...")
            self._download_file(REMOTE_ASSETS[model_name], file_path)
            
        return file_path

    def _download_file(self, url: str, dest: Path):
        response = requests.get(url, stream=True)
        total_size = int(response.headers.get('content-length', 0))
        
        with open(dest, 'wb') as f, tqdm(
            total=total_size, unit='B', unit_scale=True, desc=dest.name
        ) as pbar:
            for data in response.iter_content(1024):
                f.write(data)
                pbar.update(len(data))

def ensure_embeddings(model_names: list = None) -> list:
    dl = Downloader()
    models = model_names or list(REMOTE_ASSETS.keys())
    return [dl.fetch(m) for m in models]