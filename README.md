# SCEPTR

### **This is an unpublished prototype for internal use only.**

> NOTE: The latest version of SCEPTR no longer supports Python versions earlier than 3.9.

**S**imple **C**ontrastive **E**mbedding of the **P**eptide sequence of **T** cell **R**eceptors (**SCEPTR**) is a BERT-like attention model trained on T cell receptor (TCR) data.
It maps TCRs to vector representations, which can be used for downstream TCR and TCR repertoire analysis such as TCR clustering or classification.

## Installation

From your Python environment, run the following replacing `<VERSION_TAG>` with the appropriate version specifier.
The latest release tags can be found by checking the 'releases' section on the github repository page.

```bash
pip install git+https://github.com/yutanagano/sceptr.git@<VERSION_TAG>
```

## Prescribed data format

SCEPTR expects to receive TCR data in the form of [pandas](https://pandas.pydata.org/) [`DataFrame`](https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.html?highlight=dataframe#pandas.DataFrame) instances.
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

### Functional API (`sceptr.sceptr`)

The eponymous `sceptr` submodule is the easiest way to use SCEPTR.
It loads the default SCEPTR variant (currently `ab_sceptr`) and exposes its methods directly as module-level functions.

> NOTE: To use the functional API, import the `sceptr` submodule like so:
> ```
> from sceptr import sceptr
> ```
> Attempting to access the submodule as an attribute of the top level module
> ```
> import sceptr
> 
> sceptr.sceptr.calc_vector_representations() #...do something...
> ```
> will result in an error.

---

#### `sceptr.sceptr.calc_vector_representations(instances: DataFrame) -> ndarray`

Map a table of TCRs provided as a pandas `DataFrame` in the above format to a set of vector representations.

Parameters:

- tcrs (`DataFrame`): DataFrame in the presribed format.

Returns:

A 2D [numpy](https://numpy.org/) [`ndarray`](https://numpy.org/doc/stable/reference/generated/numpy.ndarray.html) object where every row vector corresponds to a row in the original TCR `DataFrame`.
The returned array will have shape (N, D) where N is the number of TCRs in the input data and D is the dimensionality of the SCEPTR model.

---

#### `sceptr.sceptr.calc_cdist_matrix(anchors: DataFrame, comparisons: DataFrame) -> ndarray`

Generate a cdist matrix between two collections of TCRs.

Parameters:

- anchor_tcrs (`DataFrame`): DataFrame in the prescribed format, representing TCRs from collection A.
- comparison_tcrs (`DataFrame`): DataFrame in the prescribed format, representing TCRs from collection B.

Returns:

A 2D numpy `ndarray` representing a cdist matrix between TCRs from collection A and B.
The returned array will have shape (X, Y) where X is the number of TCRs in collection A and Y is the number of TCRs in collection B.

---

#### `sceptr.sceptr.calc_pdist_vector(instances: DataFrame) -> ndarray`

Generate a pdist set of distances between each pair of TCRs in the input data.

Parameters:

- tcrs (`DataFrame`): DataFrame in the prescribed format.

Returns

A 2D numpy `ndarray` representing a pdist vector of distances between each pair of TCRs in the input data.
The returned array will have shape (1/2 * N * (N-1),), where N is the number of TCRs in the input data.

---

### Loading specific SCEPTR variants (`sceptr.variant`)

Because SCEPTR is still a project in development, there exist multiple variants of the model.
For more curious users, these model variants will be available to load and use through the `sceptr.variant` submodule.

The module exposes functions, each named after a particular model variant, which when called, will return a `Sceptr` object corresponding to the selected model variant.
This `Sceptr` object will then have the methods: `calc_pdist_vector`, `calc_cdist_matrix`, and `calc_vector_representations` available to use, with function signatures exactly as defined above for the functional API in the `sceptr.sceptr` submodule.

Currently available variants:

- `sceptr.variant.ab_sceptr` (default model used by the functional API)
- `sceptr.variant.ab_sceptr_large` (larger variant of the paired-chain model, with model dimensionality 128)
- `sceptr.variant.ab_sceptr_blosum` (variant using BLOSUM62 embeddings instead of one-hot)
- `sceptr.variant.a_sceptr` (alpha-chain only variant)
- `sceptr.variant.b_sceptr` (beta-chain only variant)
