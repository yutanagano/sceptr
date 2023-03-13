# BLASTR

### **This is a prototype for internal use only.**

> NOTE: Current models only support beta chain data. Support for alpha and paired-chain is coming soon.

## **B**ERT **L**everaged for the **A**nalysis of **T** cell **R**eceptors

BLASTR is a BERT-like attention model trained on T cell receptor (TCR) data.
It maps TCRs to vector representations, which can be used for downstream TCR and TCR repertoire analysis such as TCR clustering or classification.

## Installation

1. Clone this repository.
2. Open a terminal window, and activate a [python](https://www.python.org/) environment (e.g. via [venv](https://docs.python.org/3/library/venv.html) or [conda](https://conda.io/))
3. With your chosen environment active, run the following command inside the cloned repository directory. This will install a copy of `blastr` into that environment, and you should be able to use `blastr` in any python script as long as the same python environment is active.

```bash
$ python -m pip install .
```

## Prescribed data format

BLASTR expects to receive TCR data in the form of [pandas](https://pandas.pydata.org/) [`DataFrame`](https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.html?highlight=dataframe#pandas.DataFrame) instances.
Therefore, all TCR data should be represented as a `DataFrame` with the following structure and data types.
The column order is irrelevant.
Each row should represent one TCR.

For easier cleaning and standardisation of TCR data, check out [tidytcells](https://pypi.org/project/tidytcells/).

| Column name | Column datatype | Column contents |
|---|---|---|
|TRBV|`str`|IMGT symbol for the beta chain V gene (with allele specifier)|
|CDR3B|`str`|Amino acid sequence of the beta chain CDR3, including the first C and last W/F residues, in all caps|
|TRBJ|`str`|IMGT symbol for the beta chain J gene (with allele specifier)|

## Usage

TODO

### Default functions

#### `blastr.embed()`

#### `blastr.cdist()`

#### `blastr.pdist()`