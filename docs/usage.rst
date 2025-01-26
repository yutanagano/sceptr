Usage
=====

.. _functional_api:

Functional API (Recommended)
----------------------------

The functional :ref:`API <root>` is accessible from the root module, and is the easiest way to use SCEPTR.
When using the functional API, you will be using the default SCEPTR model (see the :ref:`model variants <model_variants>` section below).
To begin analysing TCR data with sceptr, you must first load the TCR data into memory in the :ref:`prescribed format <data_format>` using `pandas <https://pandas.pydata.org/>`_.

.. tip::
	SCEPTR only recognises TCR V gene symbols that are IMGT-compliant, and also known to be functional (i.e. known pseudogenes or ORFs are not allowed).
	For easy standardisation of TCR gene nomenclature in your data, as well as filtering your data for functional V/J genes, check out `tidytcells <https://pypi.org/project/tidytcells/>`_.

>>> from pandas import DataFrame
>>> tcrs = DataFrame(
... 	data = {
... 		"TRAV": ["TRAV38-1*01", "TRAV3*01", "TRAV13-2*01", "TRAV38-2/DV8*01"],
... 		"CDR3A": ["CAHRSAGGGTSYGKLTF", "CAVDNARLMF", "CAERIRKGQVLTGGGNKLTF", "CAYRSAGGGTSYGKLTF"],
... 		"TRBV": ["TRBV2*01", "TRBV25-1*01", "TRBV9*01", "TRBV2*01"],
... 		"CDR3B": ["CASSEFQGDNEQFF", "CASSDGSFNEQFF", "CASSVGDLLTGELFF", "CASSPGTGGNEQYF"],
... 	},
... 	index = [0,1,2,3]
... )
>>> print(tcrs)
              TRAV                 CDR3A         TRBV            CDR3B
0      TRAV38-1*01     CAHRSAGGGTSYGKLTF     TRBV2*01   CASSEFQGDNEQFF
1         TRAV3*01            CAVDNARLMF  TRBV25-1*01    CASSDGSFNEQFF
2      TRAV13-2*01  CAERIRKGQVLTGGGNKLTF     TRBV9*01  CASSVGDLLTGELFF
3  TRAV38-2/DV8*01     CAYRSAGGGTSYGKLTF     TRBV2*01   CASSPGTGGNEQYF


``calc_cdist_matrix``
*********************

As the name suggests, :py:func:`~sceptr.calc_cdist_matrix` gives you an easy way to calculate a cross-distance matrix between two sets of TCRs.

>>> import sceptr
>>> cdist_matrix = sceptr.calc_cdist_matrix(tcrs.iloc[:2], tcrs.iloc[2:])
>>> print(cdist_matrix)
[[1.2849895 0.7521934]
 [1.4653425 1.4646544]]

``calc_pdist_vector``
*********************

If you're only interested in calculating distances within a set, :py:func:`~sceptr.calc_pdist_vector` gives you a one-dimensional array of within-set distances.

>>> pdist_vector = sceptr.calc_pdist_vector(tcrs)
>>> print(pdist_vector)
[1.4135991  1.2849894  0.75219345 1.4653426  1.4646543  1.287208  ]

.. tip::
	The end result of using the :py:func:`~sceptr.calc_cdist_matrix` and :py:func:`~sceptr.calc_pdist_vector` functions are equivalent to generating sceptr's TCR representations first with :py:func:`~sceptr.calc_vector_representations`, then using `scipy <https://scipy.org/>`_'s `cdist <https://docs.scipy.org/doc/scipy/reference/generated/scipy.spatial.distance.cdist.html>`_ or `pdist <https://docs.scipy.org/doc/scipy/reference/generated/scipy.spatial.distance.pdist.html#scipy.spatial.distance.pdist>`_ functions to get the corresponding matrix or vector, respectively.
	But on machines with `CUDA-enabled GPUs <https://en.wikipedia.org/wiki/CUDA>`_, directly using sceptr's :py:func:`~sceptr.calc_cdist_matrix` and :py:func:`~sceptr.calc_pdist_vector` functions will run faster, as it internally runs all computations on the GPU.

``calc_vector_representations``
*******************************

If you want to directly operate on sceptr's TCR representations, you can use :py:func:`~sceptr.calc_vector_representations`.

>>> reps = sceptr.calc_vector_representations(tcrs)
>>> print(reps.shape)
(4, 64)

``calc_residue_representations``
********************************

The package also provides the user with an easy way to get access to SCEPTR's internal representations of each individual amino acid residue in the tokenised form of its input TCRs, as outputted by the penultimate layer of its self-attention stack.
Interested users can use :py:func:`~sceptr.calc_residue_representations`.
Please refer to the documentation for the :py:class:`~sceptr.model.ResidueRepresentations` class for details on how to interpret the output.

>>> res_reps = sceptr.calc_residue_representations(tcrs)
>>> print(res_reps)
ResidueRepresentations[num_tcrs: 4, rep_dim: 64]

.. _model_variants:

Model variants
--------------

The :py:mod:`sceptr.variant` submodule allows users access a variety of non-default SCEPTR model variants, and use them for TCR analysis.
The submodule exposes functions which return :py:class:`~sceptr.model.Sceptr` objects with the model state of the chosen variant loaded.
These model instances expose the same functions as those used in the functional API, so you can just plug and play.
For example:

>>> from sceptr import variant
>>> sceptr_tiny = variant.tiny()
>>> tiny_reps = sceptr_tiny.calc_vector_representations(tcrs)
>>> print(tiny_reps.shape)
(4, 16)

.. _data_format:

Prescribed data format
----------------------

SCEPTR expects to receive TCR data in the form of `pyrepseq standard format <https://pyrepseq.readthedocs.io/en/latest/api.html#pyrepseq.io.standardize_dataframe>`_-compliant `pandas <https://pandas.pydata.org/>`_ `DataFrame <https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.html?highlight=dataframe#pandas.DataFrame>`_\ s.
All TCR data should be represented as a DataFrame with the following structure and data types.
The column order is irrelevant.
Each row should represent one TCR.
Incomplete rows are allowed (e.g. only beta chain data available) as long as the SCEPTR :py:mod:`~sceptr.variant` that is being used has at least some partial information to go on.
Extra columns are also allowed.

+-------------+-----------------+-----------------------------------------------------------------------------------------------------+
| Column name | Column datatype | Column contents                                                                                     |
+=============+=================+=====================================================================================================+
|TRAV         |str              |IMGT symbol for the alpha chain V gene                                                               |
+-------------+-----------------+-----------------------------------------------------------------------------------------------------+
|CDR3A        |str              |Amino acid sequence of the alpha chain CDR3, including the first C and last W/F residues, in all caps|
+-------------+-----------------+-----------------------------------------------------------------------------------------------------+
|TRBV         |str              |IMGT symbol for the beta chain V gene                                                                |
+-------------+-----------------+-----------------------------------------------------------------------------------------------------+
|CDR3B        |str              |Amino acid sequence of the beta chain CDR3, including the first C and last W/F residues, in all caps |
+-------------+-----------------+-----------------------------------------------------------------------------------------------------+

Hardware acceleration / device selection
----------------------------------------

By default, SCEPTR will detect any available devices with harware-acceleration capabilities and automatically load models onto those devices.
`CUDA- <https://developer.nvidia.com/cuda-zone>`_ and `MPS-enabled <https://developer.apple.com/documentation/metalperformanceshaders>`_ devices are supported.
In cases where you would like to explicitly limit SCEPTR to using the CPU, the functions :py:func:`sceptr.disable_hardware_acceleration` is available.
The setting can also be manually toggled back with :py:func:`sceptr.enable_hardware_acceleration`.
