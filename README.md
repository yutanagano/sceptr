# SCEPTR

### **This is an unpublished prototype for internal use only.**

> [!NOTE]
> The latest version of SCEPTR no longer supports Python versions earlier than 3.9.

**S**imple **C**ontrastive **E**mbedding of the **P**eptide sequence of **T** cell **R**eceptors (**SCEPTR**) is a BERT-like attention model trained on T cell receptor (TCR) data.
It maps TCRs to vector representations, which can be used for downstream TCR and TCR repertoire analysis such as TCR clustering or classification.

## Installation

### Prerequisites

> [!IMPORTANT]
> To install `sceptr` properly, you must have [`git-lfs`](https://git-lfs.com/) installed and set up on your system.
> This is because you must be able to download the trained model weights properly during your install.
> The trained model weight files are relatively large, and are therefore not tracked directly by `git` and `github`.
> Instead, the version control system tracks a stub file which references a file hosted on the `git-lfs` servers.
> To properly de-reference these stub files at install time, you need a copy of `git-lfs`.
>
> The library code that powers `sceptr` is now outsourced to a separate package, `libtcrlm`, which is also a private repository (both this repo and `libtcrlm` will become public once SCEPTR is published).
> This means that to install `sceptr`, **users must also be granted access to the `libtcrlm` repository on github.**
> Please notify @yutanagano if you would like to continue using the latest version of `sceptr` and have not yet been granted access to this repository.
> This was done to avoid code duplication between this `sceptr` deployment repo and the development/training repo.
> Apologies to anyone inconvenienced!

> [!NOTE]
> The following prerequisites will disappear once all repositories are made public and a copy of all the install files are uploaded to PyPI.

1. [`git-lfs`](https://git-lfs.com/) must be installed and set up on your system.
2. You must have access to the `libtcrlm` repo (contact @yutanagano to request access).

### Using `pip`

From your Python environment, run the following replacing `<VERSION_TAG>` with the appropriate version specifier (e.g. `v1.0.0-alpha.1`).
The latest release tags can be found by checking the 'releases' section on the github repository page.

```bash
pip install git+https://github.com/yutanagano/sceptr.git@<VERSION_TAG>
```

### Manual install

You can also clone the repository, and from within your Python environment, navigate to the project root directory and run:

```bash
pip install .
```

Note that even for manual installation, you still need `git-lfs` to properly de-reference the stub files at `git-clone`-ing time.

## Prescribed data format

SCEPTR expects to receive TCR data in the form of [pandas](https://pandas.pydata.org/) [`DataFrame`](https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.html?highlight=dataframe#pandas.DataFrame) instances.
Therefore, all TCR data should be represented as a `DataFrame` with the following structure and data types.
The column order is irrelevant.
Each row should represent one TCR.
Incomplete rows are allowed (e.g. only beta chain data available) as long as the SCEPTR variant that is being used has at least some partial information to go on.

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

> [!TIP]
> To use the functional API, import the `sceptr` submodule like so:
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

> [!TIP]
> If your machine doesn't have a CUDA-capable GPU, or a smaller GPU, the larger variants may take a long time to run.
> The regular variants have comparable performance and should run quicker.

Because SCEPTR is still a project in development, there exist multiple variants of the model.
For more curious users, these model variants will be available to load and use through the `sceptr.variant` submodule.

The module exposes functions, each named after a particular model variant, which when called, will return a `Sceptr` object corresponding to the selected model variant.
This `Sceptr` object will then have the methods: `calc_pdist_vector`, `calc_cdist_matrix`, and `calc_vector_representations` available to use, with function signatures exactly as defined above for the functional API in the `sceptr.sceptr` submodule.

Currently available variants:

- `sceptr.variant.ab_sceptr` (default model used by the functional API)
- `sceptr.variant.ab_sceptr_large` (larger variant of the paired-chain model, with model dimensionality 128)
- `sceptr.variant.ab_sceptr_blosum` (variant using BLOSUM62 embeddings instead of one-hot)
- `sceptr.variant.ab_sceptr_cdr3_only` (only uses the CDR3 loops as input)
- `sceptr.variant.ab_sceptr_cdr3_only_mlm_only` (only uses CDR3 loops as input, and did not receive contrastive learning)
- `sceptr.variant.ab_sceptr_xlarge_cdr3_only_mlm_only` (extra larger variant using only the CDR3 sequences as input, only trained on MLM, with model dimensionality 768)
- `sceptr.variant.a_sceptr` (alpha-chain only variant)
- `sceptr.variant.b_sceptr` (beta-chain only variant)
