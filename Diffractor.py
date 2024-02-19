from typing import *
from time import time
import json
from pathlib import Path
import faiss
from collections import defaultdict, Counter
import nltk
nltk.download("stopwords", quiet=True)
from nltk.corpus import stopwords
import string
import numpy as np
import random
from enum import Enum
import scipy.stats
import math
import gensim.models
import multiprocessing as mp
from tqdm.auto import tqdm
import diffprivlib

import importlib_resources as impresources

########################### GLOBALS ###############################
stop = set([x for x in stopwords.words("english")])
punct = string.punctuation

def worker_init(lists, mapping_word, func, st, rn, sc):
    global my_idx
    global my_list
    global my_mapping
    #global epsilon
    global function
    global rep_stop
    global my_rng
    global scoring

    my_idx = mp.current_process()._identity[0] % len(lists)
    my_list = lists[my_idx].copy()
    my_mapping = mapping_word.copy()
    #epsilon = eps
    function = func
    rep_stop = st
    my_rng = np.random.default_rng(rn[my_idx])
    scoring = sc

def get_cand(arg):
    global my_idx
    global my_list
    global my_mapping
    #global epsilon
    global function
    global rep_stop
    global my_rng
    global scoring

    tokens = arg[0]
    epsilon = arg[1]

    cands = []
    for t in tokens:
        if t in stop and rep_stop == False:
            cands.append(t)
            continue

        if t in punct or t not in my_mapping:
            cands.append(t)
            continue

        start = my_mapping[t][my_idx]
        if start is None:
            cands.append(None)
        else:
            cands.append(function(my_list, start, epsilon, my_idx, my_rng, scoring))
    return cands
#######################################################################

class Algorithm(Enum):
    Flat = "flat"
    FlatOutside = "flat-outside"
    Partitioned = "partitioned"
    PartitionedOutside = "partitioned-outside"

class Lists():
    home = None

    algorithm = None
    num_lists = None
    model_names = None
    return_list = True
    save_directory = None

    with open(impresources.files("Diffractor") / "data" / "vocab.txt", 'r') as f:
        vocab = set([x.strip() for x in f.readlines()])
    lists = None
    mapping_word = None
    mapping_idx = None

    def __init__(self, home, algorithm=Algorithm.Flat, num_lists=1, return_list=True, save_directory=None, vocab=None, model_names=None):
        print("INIT")
        self.home = home
        self.algorithm = Algorithm(algorithm)
        self.num_lists = num_lists
        self.return_list = return_list
        self.model_names = model_names
        self.save_directory = save_directory

        if self.return_list == False and self.save_directory is None:
            self.save_directory = "./"

        if vocab is not None:
            self.vocab = vocab

        self.lists = self.get_all_lists()
        self.create_mappings()

        print("INIT Finished.")
        pass

    def create_mappings(self):
        print("Creating word mappings...")
        self.mapping_word = defaultdict(dict)
        self.mapping_idx = defaultdict(dict)
        for token in self.vocab:
            for i, l in enumerate(self.lists):
                try:
                    idx = l.index(token)
                    self.mapping_word[token][i] = idx
                except ValueError:
                    self.mapping_word[token][i] = None
                    continue
                self.mapping_idx[idx][i] = token
        return

    def get_all_lists(self):
        all_lists = []
        
        models = [
            ('conceptnet-numberbatch-19-08-300', 300, "{}/numberbatch-en-19.08_filtered.txt".format(self.home)), 
            ('glove-twitter-200', 200, "{}/glove.twitter.27B.200d_filtered.txt".format(self.home)), 
            ('glove-wiki-gigaword-300', 300, "{}/glove.6B.300d_filtered.txt".format(self.home)),
            ('glove-commoncrawl-300', 300, "{}/glove.840B.300d_filtered.txt".format(self.home)),
            ('word2vec-google-news-300', 300, "{}/GoogleNews-vectors-negative300_filtered.txt".format(self.home))
        ]
        
        for m in models:
            if self.model_names is not None and m[0] not in self.model_names:
                print("Skipping {}".format(m[0]))
                continue

            start_time = time()
            
            dim = m[1]
            model_name = m[0]
            model_path = m[2]
            print("Creating {} list(s) for {}...".format(self.num_lists, model_name))
            
            print("Loading {}...".format(model_name))
            wv = gensim.models.KeyedVectors.load_word2vec_format(model_path, binary=False, unicode_errors="ignore")
        
            idx = 0
            indices = []
            idx_word = []
            word_idx = {}

            print("Creating indices...")
            for _, w in enumerate(self.vocab):
                if w in wv.index_to_key:
                    j = wv.key_to_index[w]
                    indices.append(j)
                    idx_word.append(w)
                    word_idx[w] = idx
                    idx += 1    
            idx_vec = np.take(wv.vectors, indices, axis=0)
            assert(idx_vec.shape == (len(indices), dim))

            print("Creating lists...")
            sl = self.get_lists(idx_vec=idx_vec, idx_word=idx_word, dim=dim)
            all_lists.extend(sl)
            
            end_time = time()
            print("Time elapsed: {} seconds".format(round(end_time - start_time, 2)))
            del wv
            
        if self.return_list == True:
            print("All lists created.")
            return all_lists
        else:
            save_path = (Path(self.save_directory) / "lists.json").as_posix()
            with open(save_path, 'w') as out:
                json.dump(all_lists, out, indent=3)
            print("All lists created and saved to {}.".format(save_path))
            return

    def get_lists(self, idx_vec, idx_word, dim=300):
        lists = []

        # Select functions according to algorithm
        faiss_index_fn = self.get_faiss_index_fn(self.algorithm)
        sort_fn = self.get_1dsort_fn(self.algorithm)
        starting_idx_fn = self.get_starting_idx_fn(self.algorithm)

        for i in range(self.num_lists):
            # Init RNGs
            #rng_seed = 42
            rng_indices = np.random.default_rng()#seed = rng_seed)
            indices = starting_idx_fn(dim, idx_vec, self.num_lists, rng_indices)

            # Init faiss
            index = faiss_index_fn(dim)
            index.train(idx_vec)

            # Add our vectors to faiss
            index.add(idx_vec)

            # Search neighbourhood; create the list
            sorted_word_list = sort_fn(index, idx_vec, idx_word, indices[i])
            lists.append(sorted_word_list)

            # Delete remaining elements from the faiss search space
            index.reset()
        
        return lists
    
    def get_faiss_index_fn(self, algorithm):
        def index_for_flat(dim):
            quantizer = faiss.IndexFlatL2(dim)
            nlist = 100
            m = 25
            index = faiss.IndexIVFFlat(quantizer, dim, nlist)
            return index

        def index_for_partition(dim):
            quantizer = faiss.IndexFlatL2(dim)
            nlist = 50
            m = 10
            index = faiss.IndexIVFFlat(quantizer, dim, nlist)
            return index

        if algorithm == Algorithm.Flat:
            return index_for_flat
        elif algorithm == Algorithm.FlatOutside:
            return index_for_flat
        elif algorithm == Algorithm.Partitioned:
            return index_for_partition
        elif algorithm == Algorithm.PartitionedOutside:
            return index_for_partition

    def get_1dsort_fn(self, algorithm):
        def generate_with_flat_algo(index, idx_vec, idx_word, start_index):
            index.nprobe = 20
            sorted_list = []
            idx = start_index
            for _ in range(0, len(idx_vec)):
                vec = idx_vec[idx]
                sorted_list.append(idx_word[idx])
                index.remove_ids(np.asarray([idx]))

                D, I = index.search(vec[None, :], k=1)
                idx = I[0][0]
            return sorted_list

        def generate_with_partitioned_algo(index, idx_vec, idx_word, start_index):
            sorted_list = []
            idx = start_index
            for _ in range(0, len(idx_vec)):
                vec = idx_vec[idx]
                sorted_list.append(idx_word[idx])
                index.remove_ids(np.asarray([idx]))

                for nprobe in range(1, index.nlist + 1):
                    index.nprobe = nprobe
                    _, I = index.search(vec[None, :], k=1)
                    idx = I[0][0]
                    if idx != -1:
                        break
            return sorted_list

        if algorithm == Algorithm.Flat:
            return generate_with_flat_algo
        elif algorithm == Algorithm.FlatOutside:
            return generate_with_flat_algo
        elif algorithm == Algorithm.Partitioned:
            return generate_with_partitioned_algo
        elif algorithm == Algorithm.PartitionedOutside:
            return generate_with_partitioned_algo

    def get_starting_idx_fn(self, algorithm):
        def calculate_starting_indices(dim, idx_vec, num_lists, rng):
            index = faiss.IndexFlatL2(dim + 1)
            index.add(self.augment_database(idx_vec))
            origin = np.zeros_like(idx_vec[0])
            _, I = index.search(self.augment_queries(origin[None, :]), k=num_lists)
            return I[0]

        def default_starting_indices(dim, idx_vec, num_lists, rng):
            return rng.integers(low=0, high=len(idx_vec)-1, size=num_lists)

        if algorithm == Algorithm.FlatOutside:
            return calculate_starting_indices
        elif algorithm == Algorithm.PartitionedOutside:
            return calculate_starting_indices
        else:
            return default_starting_indices

    # https://gist.github.com/mdouze/c7653aaa8c3549b28bad75bd67543d34#file-demo_farthest_l2-ipynb
    def augment_queries(self, xq):
        extra_column = np.ones((len(xq), 1), dtype=xq.dtype)
        return np.hstack((xq, extra_column))

    def augment_database(self, xb):
        norms2 = (xb ** 2).sum(1)
        return np.hstack((-2 * xb, norms2[:, None]))
    ################################################################

class Diffractor():
    RNG = None
    SS = None

    L = None
    mu = None
    scale = None
    sensitivity = None
    gamma = None
    rep_stop = None
    pool = None
    epsilon = None
    scoring = None

    geo_mech = None

    progress_bar = None

    def __init__(self, L, gamma=5, epsilon=5, rep_stop=False, method="geometric", scoring="distance", progress_bar=True):
        self.L = L
        self.mu = 0
        self.scale = 1
        self.sensitivity = 1
        self.gamma = gamma
        self.rep_stop = rep_stop
        self.epsilon = epsilon
        self.scoring = scoring

        self.progress_bar = not progress_bar

        self.RNG = np.random.default_rng(42)
        self.SS = np.random.SeedSequence(42)

        #self.geo_mech = diffprivlib.mechanisms.GeometricTruncated(epsilon=self.epsilon, lower=0, upper=min([len(x)-1 for x in L.lists]))
        if method == "TEM":
            METHOD = self.truncated_exponential
        else:
            METHOD = self.geometric

        num_workers = len(self.L.lists)
        rns = self.SS.spawn(num_workers)
        self.pool = mp.Pool(num_workers, initializer=worker_init, initargs=(self.L.lists, self.L.mapping_word, METHOD, self.rep_stop, rns, self.scoring))
        pass

    def cleanup(self):
        self.pool.close()
        del self.L
        print("Cleanup complete.")

    def rewrite(self, texts, epsilon=None):
        if not isinstance(texts, list):
            texts = [texts]
        
        if epsilon is not None and not isinstance(epsilon, list):
            print("Invalid epsilon format.")
            return
        elif epsilon is None:
            epsilon = [epsilon for _ in range(tokens)]

        total_lists = len(self.L.lists)
        perturbed = []
        num_perturbed = 0
        num_diff = 0
        total = 0
        support = []
        for i, s in enumerate(tqdm(texts, disable=self.progress_bar)):
            tokens = nltk.word_tokenize(s)
            eps = epsilon[i]
            cands = self.pool.map(get_cand, [(tokens, eps) for _ in range(total_lists)])
            temp_replace = []
            for i in range(len(cands[0])):
                i_cands = [x[i] for x in cands if x[i] is not None]
                if len(i_cands) == 0:
                    new_word = tokens[i]
                else:
                    new_word = self.RNG.choice(i_cands)
                    num_perturbed += 1
                    if tokens[i] != new_word:
                        num_diff += 1
                support.append(len(set(i_cands)))
                total += 1
                temp_replace.append(new_word)
            perturbed.append(" ".join(temp_replace))
        return perturbed, num_perturbed, num_diff, total, round(np.mean(support), 3)

    def geometric(self, l, idx, epsilon, l_id, rng=None, scoring=None):
        geo_mech = diffprivlib.mechanisms.GeometricTruncated(epsilon=epsilon, lower=0, upper=min([len(x)-1 for x in self.L.lists]))
        new_idx = geo_mech.randomise(idx)
        return l[new_idx]

    def truncated_exponential(self, l, idx, epsilon, l_id, rng, scoring="distance"):
        scores, Lw, rest, sup = self.score(l, idx, l_id, scoring)
        perp = (-1 * self.gamma) + (2 * self.sensitivity * np.log(sup) / epsilon)
        Lw.append("⊥")
        scores.append(perp)
        noisy = [x + rng.gumbel(loc=0, scale=(2*self.sensitivity)/epsilon) for x in scores]
        w_hat = Lw[np.argmax(noisy)]
        if w_hat != "⊥":
            return w_hat
        else:
            return rng.choice(rest)

    def score(self, l, start, l_id, scoring):
        part = l[max(0, math.ceil(start-self.gamma)):min(len(l), math.floor(start+self.gamma))+1]
        rest = list(set(l) - set(part))
        score = [-1 * abs(start-self.L.mapping_word[x][l_id]) for _, x in enumerate(part)]
        if scoring != "distance":
            score = [scipy.stats.norm.pdf(x, loc=self.mu, scale=self.scale) for x in score]
        return score, part, rest, len(l)-len(part)