<div align="left">

  [![PyPI version](https://img.shields.io/pypi/v/diffractor.svg)](https://pypi.org/project/diffractor/)
  [![License](https://img.shields.io/github/license/sjmeis/diffractor.svg)](https://github.com/sjmeis/diffractor/blob/main/LICENSE)

</div>

# 1-Diffractor
`1-Diffractor` is a high-performance library for word-level text perturbation leveraging Metric Differential Privacy. It maps text into 1D sorted embedding spaces to apply noise, ensuring privacy guarantees while maintaining semantic utility.


## Key Features
 - **Metric DP Implementation**: Support for both Truncated Geometric and Truncated Exponential (TEM) mechanisms.
 - **Automated Embedding Management**: Automatically downloads and caches filtered embedding models (GloVe, Word2Vec, Numberbatch).
 - **Parallel Processing**: Uses optimized multiprocessing to perturb large batches of text quickly.
 - **BYOE (Bring Your Own Embeddings)**: CLI tools to clean and integrate custom embedding files into the `1-Diffractor`.

## Quickstart Guide
### Installation
```bash
pip install dp-diffractor
```

### Basic Usage
```python
from diffractor import Diffractor, DiffractorConfig

# Configure the privacy mechanism
config = DiffractorConfig(
    method="geometric", 
    epsilon=1.0, 
    verbose=True
)

with Diffractor(config) as df:
    texts = ["Differential Privacy is really cool!", "Hello world."]
    perturbed = df.rewrite(texts)
    print(perturbed)
```

## Advanced Configuration
The `DiffractorConfig` object allows you to customize the privatization parameters:

| Parameter         | Default       | Description                                              |
|-------------------|---------------|----------------------------------------------------------|
| `method`            | `geometric`   | The DP mechanism: `geometric` or `TEM`.                  |
| `epsilon`           | `1.0`           | Privacy budget (ε). Lower is more private.               |
| `gamma`             | `5`             | Neighborhood radius for the `TEM` scoring function.      |
| `sensitivity`       | `1.0`           | Sensitivity of the scoring function.                     |
| `replace_stopwords` | `False`         | If False, keeps common stopwords unchanged.              |
| `verbose`           | `True`          | Enables progress bars and status logging.                |
| `seed`              | `42`            | Global seed for reproducible perturbations.              |


---

## Managing Embeddings

`1-Diffractor` keeps a local cache (by default, `~/.cache/diffractor`) to store embedding files.

### Custom Embeddings (BYOE)
If you have your own embedding file, you must filter it against the internal vocabulary to ensure it works with the privatization mechanism:

```
# In your terminal
diffractor-clean path/to/my_vectors.txt
```

Then, use it during startup:

```python
df = Diffractor(model_names=["my_vectors_filtered"])
```

### Default Models
By default, `1-Diffractor` fetches and uses the following embedding models:
 - `conceptnet-numberbatch-19-08-300`
 - `glove-twitter-200`
 - `glove-wiki-gigaword-300`
 - `glove-commoncrawl-30`
 - `word2vec-google-news-300`
 
---

## Citation
If you find `1-Diffractor` useful or make use of it in your research, please be sure to cite the original paper:

```
@inproceedings{10.1145/3643651.3659896,
author = {Meisenbacher, Stephen and Chevli, Maulik and Matthes, Florian},
title = {1-Diffractor: Efficient and Utility-Preserving Text Obfuscation Leveraging Word-Level Metric Differential Privacy},
year = {2024},
isbn = {9798400705564},
publisher = {Association for Computing Machinery},
address = {New York, NY, USA},
url = {https://doi.org/10.1145/3643651.3659896},
doi = {10.1145/3643651.3659896},
booktitle = {Proceedings of the 10th ACM International Workshop on Security and Privacy Analytics},
pages = {23–33},
numpages = {11},
keywords = {data privacy, differential privacy, natural language processing},
location = {Porto, Portugal},
series = {IWSPA '24}
}
```

Please also consider citing the hosted embedding files:

```
@dataset{meisenbacher_2026_19701515,
  author       = {Meisenbacher, Stephen},
  title        = {Filtered Embedding Files for 1-Diffractor},
  month        = apr,
  year         = 2026,
  publisher    = {Zenodo},
  doi          = {10.5281/zenodo.19701515},
  url          = {https://doi.org/10.5281/zenodo.19701515},
}
```