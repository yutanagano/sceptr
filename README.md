# SCEPTR

> [!NOTE]
> The latest version of SCEPTR no longer supports Python versions earlier than 3.9.

**S**imple **C**ontrastive **E**mbedding of the **P**rimary sequence of **T** cell **R**eceptors (**SCEPTR**) is a BERT-like attention model trained on T cell receptor (TCR) data.
It maps TCRs to vector representations, which can be used for downstream TCR and TCR repertoire analysis such as TCR clustering or classification.

## Installation

### From PyPI (Recommended)

Coming soon.

### From Source

> [!IMPORTANT]
> To install `sceptr` from source, you must have [`git-lfs`](https://git-lfs.com/) installed and set up on your system.
> This is because you must be able to download the trained model weights directly from the Git LFS servers during your install.

#### Using `pip`

From your Python environment, run the following replacing `<VERSION_TAG>` with the appropriate version specifier (e.g. `v1.0.0-alpha.1`).
The latest release tags can be found by checking the 'releases' section on the github repository page.

```bash
pip install git+https://github.com/yutanagano/sceptr.git@<VERSION_TAG>
```

#### Manual install

You can also clone the repository, and from within your Python environment, navigate to the project root directory and run:

```bash
pip install .
```

Note that even for manual installation, you still need `git-lfs` to properly de-reference the stub files at `git-clone`-ing time.

#### Troubleshooting

A recent security update to `git` has resulted in some difficulties cloning repositories that rely on `git-lfs`.
This can result in an error message with a message along the lines of:

```
fatal: active `post-checkout` hook found during `git clone`
```

If this happens, you can temporarily set the `GIT_CLONE_PROTECTION_ACTIVE` environment variable to `false` by prepending `GIT_CLONE_PROTECTION_ACTIVE=false` before the install command like below:

```bash
GIT_CLONE_PROTECTION_ACTIVE=false pip install git+https://github.com/yutanagano/sceptr.git@<VERSION_TAG>
```

This is [a known issue](https://github.com/git-lfs/git-lfs/issues/5749) for `git` version `2.45.1` and [is fixed](https://lore.kernel.org/git/xmqqr0dheuw5.fsf@gitster.g/T/#u) from version `2.45.2`.

## Prescribed data format

> [!IMPORTANT]
> SCEPTR only recognises TCR V/J gene symbols that are IMGT-compliant, and also known to be functional (i.e. known pseudogenes or ORFs are not allowed).
> For easy standardisation of TCR gene nomenclature in your data, as well as filtering your data for functional V/J genes, check out [tidytcells](https://pypi.org/project/tidytcells/).

SCEPTR expects to receive TCR data in the form of [pandas](https://pandas.pydata.org/) [`DataFrame`](https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.html?highlight=dataframe#pandas.DataFrame) instances.
Therefore, all TCR data should be represented as a `DataFrame` with the following structure and data types.
The column order is irrelevant.
Each row should represent one TCR.
Incomplete rows are allowed (e.g. only beta chain data available) as long as the SCEPTR variant that is being used has at least some partial information to go on.

| Column name | Column datatype | Column contents |
|---|---|---|
|TRAV|`str`|IMGT symbol for the alpha chain V gene|
|CDR3A|`str`|Amino acid sequence of the alpha chain CDR3, including the first C and last W/F residues, in all caps|
|TRAJ|`str`|IMGT symbol for the alpha chain J gene|
|TRBV|`str`|IMGT symbol for the beta chain V gene|
|CDR3B|`str`|Amino acid sequence of the beta chain CDR3, including the first C and last W/F residues, in all caps|
|TRBJ|`str`|IMGT symbol for the beta chain J gene|

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

Because SCEPTR is still a project in development, there exist multiple variants of the model.
For more curious users, these model variants will be available to load and use through the `sceptr.variant` submodule.

The module exposes functions, each named after a particular model variant, which when called, will return a `Sceptr` object corresponding to the selected model variant.
This `Sceptr` object will then have the methods: `calc_pdist_vector`, `calc_cdist_matrix`, and `calc_vector_representations` available to use, with function signatures exactly as defined above for the functional API in the `sceptr.sceptr` submodule.

#### Paired-chain variants

|Name|Description|
|---|---|
|`sceptr.variant.default`|default model used by the functional API|
|`sceptr.variant.mlm_only`|default model trained without autocontrastive learning|
|`sceptr.variant.left_aligned`|similar to default model but with learnable token embeddings and a sinusoidal position information embedding method more similar to the original NLP BERT/transformer models|
|`sceptr.variant.left_aligned_mlm_only`|left-aligned variant trained without autocontrastive learning|
|`sceptr.variant.cdr3_only`|only uses the CDR3 loops as input|
|`sceptr.variant.cdr3_only_mlm_only`|only uses CDR3 loops as input, and did not receive autocontrastive learning|
|`sceptr.variant.large`|larger variant with model dimensionality 128|
|`sceptr.variant.small`|smaller variant with model dimensionality 32|
|`sceptr.variant.tiny`|smaller variant with model dimensionality 16|
|`sceptr.variant.blosum`|variant using BLOSUM62 embeddings instead of one-hot|
|`sceptr.variant.average_pooling`|variant using the average-pooling method to generate the TCR representation vector|
|`sceptr.variant.shuffled_data`|variant trained on the Tanno et al. dataset with randomised alpha/beta pairing|
|`sceptr.variant.synthetic_data`|variant trained using synthetic TCR sequences generated by OLGA|
|`sceptr.variant.dropout_noise_only`|variant trained without residue/chain dropping during autocontrastive learning|
|`sceptr.variant.finetuned`|variant fine-tuned using supervised contrastive learning for six pMHCs with peptides GILGFVFTL, NLVPMVATV, SPRWYFYYL, TFEYVSQPFLMDLE, TTDPSFLGRY and YLQPRTFLL (from [VDJdb](https://vdjdb.cdr3.net/))|

#### Single-chain variants

|Name|Description
|---|---|
|`sceptr.variant.a_sceptr`|alpha-chain only variant|
|`sceptr.variant.b_sceptr`|beta-chain only variant|
