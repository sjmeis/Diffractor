import faiss
import numpy as np
from typing import List
from gensim.models import KeyedVectors
import importlib_resources as impresources
from pathlib import Path

from .downloader import Downloader

class EmbeddingManager:
    """Handles the transformation of vectors into searchable FAISS indices and sorted lists."""
    
    def __init__(self, file_paths: List[str]):
        self.lists = []
        self.word_to_idx = []
        self._load_all(file_paths)

    def _load_all(self, paths):
        for path in paths:
            print(f"Processing {path}...")
            wv = KeyedVectors.load_word2vec_format(path, binary=False)
            vectors = wv.vectors.astype('float32')
            words = wv.index_to_key
            
            sorted_words = self.create_sorted_list(vectors, words)
            self.lists.append(sorted_words)
            
            self.word_to_idx.append({w: i for i, w in enumerate(sorted_words)})

    @staticmethod
    def create_sorted_list(vectors: np.ndarray, words: List[str]) -> List[str]:
        """Greedy nearest neighbor search."""
        dim = vectors.shape[1]
        index = faiss.IndexFlatL2(dim)
        index.add(vectors)
        
        sorted_list = []
        current_idx = 0
        
        remaining_indices = set(range(len(words)))
        
        for _ in range(len(words)):
            sorted_list.append(words[current_idx])
            remaining_indices.remove(current_idx)
            
            if not remaining_indices:
                break
                
            # Find nearest neighbor
            query_vec = vectors[current_idx].reshape(1, -1)
            k = min(50, len(remaining_indices) + 1)
            distances, indices = index.search(query_vec, k)
            
            found = False
            for neighbor_idx in indices[0]:
                if neighbor_idx in remaining_indices:
                    current_idx = neighbor_idx
                    found = True
                    break
            
            if not found:
                current_idx = next(iter(remaining_indices))
                
        return sorted_list

def prepare_custom_embeddings(input_path: str, output_path: str):
    """
    Utility for users to clean their own embedding files.
    Removes words not in the Diffractor vocab.
    """
    with open(impresources.files("diffractor.data") / "vocab.txt", 'r') as f:
        vocab = set([x.strip() for x in f.readlines()])

    dl = Downloader()
    if output_name is None:
        # Default to the input filename but marked as filtered
        output_name = Path(input_path).stem + "_filtered.txt"
    
    output_path = dl.cache_dir / output_name

    count = 0
    lines_to_keep = []
    
    print(f"Filtering {input_path} against internal vocabulary...")
    
    with open(input_path, 'r', encoding='utf-8', errors='ignore') as f:
        first_line = f.readline()
        if len(first_line.split()) > 2:
            f.seek(0)
        
        for line in f:
            word = line.split(maxsplit=1)[0]
            if word in vocab:
                lines_to_keep.append(line)
                count += 1
                
    if not lines_to_keep:
        raise ValueError("No words from the input file matched the internal vocabulary.")

    with open(output_path, 'w', encoding='utf-8') as fout:
        dim = len(lines_to_keep[0].split()) - 1
        fout.write(f"{count} {dim}\n")
        fout.writelines(lines_to_keep)

    print(f"Success! {count} words saved to cache: {output_path}")
    return output_path

def prepare_custom_embeddings_cli():
    import sys
    if len(sys.argv) < 2:
        print("Usage: diffractor-clean <path_to_raw_embeddings.txt>")
        return
    prepare_custom_embeddings(sys.argv[1])