Usage
=====

Functional API (:py:mod:`sceptr.sceptr`)
----------------------------------------

.. tip::
	To use the functional API, import the `sceptr` submodule like so:

	>>> from sceptr import sceptr

	Attempting to access the submodule as an attribute of the top level module

	>>> import sceptr
	>>> # ... load data, etc ...
	>>> sceptr.sceptr.calc_vector_representations(df)

	will result in an error.

The eponymous :py:mod:`sceptr.sceptr` submodule is the easiest way to use SCEPTR.
It loads the default SCEPTR variant and exposes its methods directly as module-level functions.

Model Variants (:py:mod:`sceptr.variant`)
-----------------------------------------

For more curious users, model variants are available to load and use through the :py:mod:`sceptr.variant` submodule.
The module exposes functions, each named after a particular model variant, which when called, will return a :py:class:`~sceptr.model.Sceptr` object corresponding to the selected model variant.
This :py:class:`~sceptr.model.Sceptr` object will then have the methods: `calc_pdist_vector`, `calc_cdist_matrix`, and `calc_vector_representations` available to use, with function signatures exactly as defined above for the functional API in the :py:mod:`sceptr.sceptr` submodule.

.. _data_format:

Prescribed data format
----------------------

.. important::
	SCEPTR only recognises TCR V/J gene symbols that are IMGT-compliant, and also known to be functional (i.e. known pseudogenes or ORFs are not allowed).
	For easy standardisation of TCR gene nomenclature in your data, as well as filtering your data for functional V/J genes, check out `tidytcells <https://pypi.org/project/tidytcells/>`_.

SCEPTR expects to receive TCR data in the form of `pandas <https://pandas.pydata.org/>`_ `DataFrame <https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.html?highlight=dataframe#pandas.DataFrame>`_ instances.
Therefore, all TCR data should be represented as a `DataFrame` with the following structure and data types.
The column order is irrelevant.
Each row should represent one TCR.
Incomplete rows are allowed (e.g. only beta chain data available) as long as the SCEPTR :py:mod:`~sceptr.variant` that is being used has at least some partial information to go on.

+-------------+-----------------+-----------------------------------------------------------------------------------------------------+
| Column name | Column datatype | Column contents                                                                                     |
+=============+=================+=====================================================================================================+
|TRAV         |`str`            |IMGT symbol for the alpha chain V gene                                                               |
+-------------+-----------------+-----------------------------------------------------------------------------------------------------+
|CDR3A        |`str`            |Amino acid sequence of the alpha chain CDR3, including the first C and last W/F residues, in all caps|
+-------------+-----------------+-----------------------------------------------------------------------------------------------------+
|TRAJ         |`str`            |IMGT symbol for the alpha chain J gene                                                               |
+-------------+-----------------+-----------------------------------------------------------------------------------------------------+
|TRBV         |`str`            |IMGT symbol for the beta chain V gene                                                                |
+-------------+-----------------+-----------------------------------------------------------------------------------------------------+
|CDR3B        |`str`            |Amino acid sequence of the beta chain CDR3, including the first C and last W/F residues, in all caps |
+-------------+-----------------+-----------------------------------------------------------------------------------------------------+
|TRBJ         |`str`            |IMGT symbol for the beta chain J gene                                                                |
+-------------+-----------------+-----------------------------------------------------------------------------------------------------+
