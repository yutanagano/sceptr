Installation
============

From `PyPI <https://pypi.org/project/sceptr/>`_ (Recommended)
-------------------------------------------------------------

.. code-block:: console

	$ pip install sceptr

From `Source <https://github.com/yutanagano/sceptr>`_
-----------------------------------------------------

.. important::
	 To install ``sceptr`` versions 1.1.0 and below from source, you must have
	 `git-lfs <https://git-lfs.com/>`_ installed and set up on your system. This
	 is because you must be able to download the trained model weights directly
	 from the Git LFS servers during your install.

From your Python environment, run the following replacing ``<VERSION_TAG>``
with the appropriate version specifier (e.g. ``v1.0.0``). The latest release
tags can be found by checking the 'releases' section on the github repository
page.

.. code-block:: console

	$ pip install git+https://github.com/yutanagano/sceptr.git@<VERSION_TAG>

You can also clone the repository, and from within your Python environment,
navigate to the project root directory and run:

.. code-block:: console

	$ pip install .

Note that even for manual installation, you still need ``git-lfs`` to properly
de-reference the stub files at ``git-clone``-ing time.
