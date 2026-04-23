import multiprocessing as mp
import numpy as np
import nltk
from typing import List, Union, Optional
from dataclasses import dataclass
from tqdm.auto import tqdm
from nltk.tokenize.treebank import TreebankWordDetokenizer

from .downloader import ensure_embeddings
from .utils import EmbeddingManager

@dataclass
class DiffractorConfig:
    gamma: int = 5
    epsilon: float = 1.0
    sensitivity: float = 1.0 
    method: str = "geometric" 
    replace_stopwords: bool = False
    verbose: bool = True
    seed: int = 42

class Diffractor:
    def __init__(self, config: Optional[DiffractorConfig] = None, model_names: Optional[List[str]] = None):
        self.config = config or DiffractorConfig()
        self.detok = TreebankWordDetokenizer()
        
        self.paths = ensure_embeddings(model_names)
        self.mgr = EmbeddingManager(self.paths)
        
        self._pool = None
        self._rng = np.random.default_rng(self.config.seed)
        self._stop_words = set(nltk.corpus.stopwords.words("english"))

    def __enter__(self):
        self.start()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()

    def start(self):
        if self._pool is not None:
            return

        num_workers = len(self.mgr.lists)
        ss = np.random.SeedSequence(self.config.seed)
        child_seeds = ss.spawn(num_workers)

        self._pool = mp.Pool(
            processes=num_workers,
            initializer=self._worker_initializer,
            initargs=(self.mgr.lists, self.mgr.word_to_idx, self.config, child_seeds)
        )

    @staticmethod
    def _worker_initializer(lists, mapping, config, seeds):
        global _RESOURCES
        worker_id = mp.current_process()._identity[0] - 1
        _RESOURCES = {
            "list": lists[worker_id],
            "mapping": mapping[worker_id],
            "config": config,
            "rng": np.random.default_rng(seeds[worker_id])
        }

    def rewrite(self, texts: Union[str, List[str]], epsilon: Optional[Union[float, List[float]]] = None) -> List[str]:
        if isinstance(texts, str):
            texts = [texts.lower()]
        else:
            texts = [x.lower() for x in texts]

        if self._pool is None:
            self.start()

        results = []
        for text in tqdm(texts, disable=not self.config.verbose, desc="Rewriting"):
            tokens = nltk.word_tokenize(text)

            if isinstance(epsilon, list):
                if len(epsilon) != len(tokens):
                    raise ValueError(f"Epsilon list length ({len(epsilon)}) must match token count ({len(tokens)}).")
                token_epsilons = epsilon
            else:
                active_eps = epsilon if epsilon is not None else self.config.epsilon
                token_epsilons = [active_eps] * len(tokens)

            perturbed_tokens = []
            for token, token_eps in zip(tokens, token_epsilons):
                args = [(token, token_eps)] * len(self.mgr.lists)
                candidates = self._pool.starmap(apply_privacy_mechanism, args)
                final_word = self._pick_final_word(token, candidates)
                perturbed_tokens.append(final_word)
            results.append(self.detok.detokenize(perturbed_tokens))
            
        return results

    def _pick_final_word(self, original_token: str, candidates: List[Optional[str]]) -> str:
        if candidates is None:
            return original_token

        valid_cands = [c for c in candidates if c is not None]

        if not valid_cands:
            return original_token
            
        return self._rng.choice(valid_cands)

    def close(self):
        if self._pool:
            self._pool.close()
            self._pool.join()
            self._pool = None

def apply_privacy_mechanism(token: str, epsilon: float) -> Optional[str]:
    res = _RESOURCES
    config = res["config"]
    rng = res["rng"]
    mapping = res["mapping"]
    s_list = res["list"]

    if not config.replace_stopwords and token.lower() in nltk.corpus.stopwords.words("english"):
        return None
    
    if token not in mapping:
        return None

    idx = mapping[token]

    if config.method == "geometric":
        p = 1 - np.exp(-epsilon / config.sensitivity)
        noise = rng.geometric(p) - rng.geometric(p)
        new_idx = int(np.clip(idx + noise, 0, len(s_list) - 1))
        return s_list[new_idx]
    elif config.method == "TEM":
        start = max(0, int(idx - config.gamma))
        end = min(len(s_list), int(idx + config.gamma + 1))
        
        neighborhood_indices = list(range(start, end))
        neighborhood_words = s_list[start:end]

        scores = [-1.0 * abs(idx - i) for i in neighborhood_indices]
        
        support_size = len(s_list) - len(neighborhood_indices)
        if support_size <= 0: support_size = 1 # Avoid log(0)
        
        perp_score = (-1.0 * config.gamma) + (2.0 * config.sensitivity * np.log(support_size) / epsilon)
        
        scale = (2.0 * config.sensitivity) / epsilon
        noisy_scores = [s + rng.gumbel(loc=0, scale=scale) for s in scores]
        noisy_perp = perp_score + rng.gumbel(loc=0, scale=scale)
        
        max_idx = np.argmax(noisy_scores)
        if noisy_scores[max_idx] > noisy_perp:
            return neighborhood_words[max_idx]
        else:
            while True:
                rand_idx = rng.integers(0, len(s_list))
                if rand_idx < start or rand_idx >= end:
                    return s_list[rand_idx]