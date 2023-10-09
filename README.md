# BLAsTR

### **This is an unpublished prototype for internal use only.**

> NOTE: The latest version of BLAsTR no longer supports Python versions earlier than 3.9.

> NOTE: The latest version of BLAsTR is a beta-chain only model. An updated paired-chain model is returning very soon.

**B**ERT **L**everaged for the **A**nalysi**s** of **T** cell **R**eceptors (**BLAsTR**) is a BERT-like attention model trained on T cell receptor (TCR) data.
It maps TCRs to vector representations, which can be used for downstream TCR and TCR repertoire analysis such as TCR clustering or classification.

## Installation

1. Clone this repository.
2. Open a terminal window, and activate a [python](https://www.python.org/) environment (e.g. via [venv](https://docs.python.org/3/library/venv.html) or [conda](https://conda.io/))
3. With your chosen environment active, run the following command inside the cloned repository directory. This will install a copy of `blastr` into that environment, and you should be able to use `blastr` in any python script as long as the same python environment is active.

```bash
$ python -m pip install .
```

## Prescribed data format

> NOTE: The latest version of BLAsTR is back to being a beta-chain only model, which means that even if you supply the model with data including alpha chains, it will not use this data.

BLASTR expects to receive TCR data in the form of [pandas](https://pandas.pydata.org/) [`DataFrame`](https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.html?highlight=dataframe#pandas.DataFrame) instances.
Therefore, all TCR data should be represented as a `DataFrame` with the following structure and data types.
The column order is irrelevant.
Each row should represent one TCR.

For easier cleaning and standardisation of TCR data, check out [tidytcells](https://pypi.org/project/tidytcells/).

| Column name | Column datatype | Column contents |
|---|---|---|
|TRAV|`str`|IMGT symbol for the alpha chain V gene (with allele specifier)|
|CDR3A|`str`|Amino acid sequence of the alpha chain CDR3, including the first C and last W/F residues, in all caps|
|TRAJ|`str`|IMGT symbol for the alpha chain J gene (with allele specifier)|
|TRBV|`str`|IMGT symbol for the beta chain V gene (with allele specifier)|
|CDR3B|`str`|Amino acid sequence of the beta chain CDR3, including the first C and last W/F residues, in all caps|
|TRBJ|`str`|IMGT symbol for the beta chain J gene (with allele specifier)|

## Usage

### Functional API (`blastr.blastr`)

The eponymously named `blastr` submodule is the easiest way to use BLAsTR.
It loads the default BLAsTR variant (currently `beta_cdr_bert_unsupervised_large`) and exposes its methods directly as module-level functions.

---

#### `blastr.blastr.calc_vector_representations(tcrs: DataFrame) -> ndarray`

Map a table of TCRs provided as a pandas `DataFrame` in the above format to a set of vector representations.

Parameters:

- tcrs (`DataFrame`): DataFrame in the presribed format.

Returns:

A 2D [numpy](https://numpy.org/) [`ndarray`](https://numpy.org/doc/stable/reference/generated/numpy.ndarray.html) object where every row vector corresponds to a row in the original TCR `DataFrame`.
The returned array will have shape (N, D) where N is the number of TCRs in the input data and D is the dimensionality of the blastr model.

---

#### `blastr.blastr.calc_cdist_matrix(anchor_tcrs: DataFrame, comparison_tcrs: DataFrame) -> ndarray`

Generate a cdist matrix between two collections of TCRs.

Parameters:

- anchor_tcrs (`DataFrame`): DataFrame in the prescribed format, representing TCRs from collection A.
- comparison_tcrs (`DataFrame`): DataFrame in the prescribed format, representing TCRs from collection B.

Returns:

A 2D numpy `ndarray` representing a cdist matrix between TCRs from collection A and B.
The returned array will have shape (X, Y) where X is the number of TCRs in collection A and Y is the number of TCRs in collection B.

---

#### `blastr.blastr.calc_pdist_vector(tcrs: DataFrame) -> ndarray`

Generate a pdist set of distances between each pair of TCRs in the input data.

Parameters:

- tcrs (`DataFrame`): DataFrame in the prescribed format.

Returns

A 2D numpy `ndarray` representing a pdist vector of distances between each pair of TCRs in the input data.
The returned array will have shape (1/2 * N * (N-1),), where N is the number of TCRs in the input data.

---

### Loading specific BLAsTR variants (`blastr.variant`)

Because BLAsTR is still a project in development, there exist multiple variants of the model.
For more curious users, these model variants will be available to load and use through the `blastr.variant` submodule.

The module exposes functions, each named after a particular model variant, which when called, will return a `BLAsTR` object corresponding to the selected model variant.
This `BLAsTR` object will then have the methods: `calc_pdist_vector`, `calc_cdist_matrix`, and `calc_vector_representations` available to use, with function signatures exactly as defined above for the functional API in the `blastr.blastr` submodule.

Currently available variants:

- `blastr.variant.beta_cdr_bert_unsupervised`
- `blastr.variant.beta_cdr_bert_unsupervised_large`