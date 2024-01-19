# Diffractor
Code repository for `1-Diffractor`, a highly efficient word-level Metric Differential Privacy mechanism.

## Quick Start
`1-Diffractor` is made up of two parts: (1) lists, and (2) the Diffractor. In order to use the Diffractor mechanism, you must initiate a `Lists` object and pass this to a `Diffractor` object.

### Creating Lists
In order to create a Lists object, use must initialize with the following parameters:
- *num_lists*: number of lists per embedding model (default: 1)
- *model_names*: which specific models to use (default: use all)
- *home*: location of the embedding files (default: current directory)

To create the Lists, simply call `L = Diffractor.Lists(**args)`. This process will take a short while, depending on the above parameters.

NOTE: for the default set of lists you must download the models from the following [https://drive.google.com/drive/folders/1ExL4XIxYCK1_9oiy5PwwMlxwqCImYW9W?usp=sharing](directory). Then you must specify the corresponding *home* argument.

### Initalializing the Diffractor
With the Lists object ready, you can now set up a Diffractor. Some parameters here:
- *epsilon*: the privacy parameter (default: 5)
- *method*: the exact underlying method, either "TEM" or "geometric" (default: geometric)
- *rep_stop*: whether to replace stop words or not (default: False == do not replace)

With these, simply call `D = Diffractor(L=L, **args)`.

### Text Privatization
`1-Diffractor` is optimized to run on multiple cores, and to process input texts in parallel. Therefore, the optimal usage is:

`private_texts = D.rewrite(input_texts)`

with *input_texts* as a list of inputs texts, i.e., a list of sentences / documents.
