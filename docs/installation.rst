Installation
============

From `PyPI <https://pypi.org/project/sceptr/>`_ (Recommended)
-------------------------------------------------------------

.. code-block:: bash

	$ pip install sceptr

From `Source <https://github.com/yutanagano/sceptr>`_
-----------------------------------------------------

.. important::
	To install `sceptr` from source, you must have `git-lfs <https://git-lfs.com/>`_ installed and set up on your system.
	This is because you must be able to download the trained model weights directly from the Git LFS servers during your install.

Using `pip`
...........

From your Python environment, run the following replacing `<VERSION_TAG>` with the appropriate version specifier (e.g. `v1.0.0-beta.1`).
The latest release tags can be found by checking the 'releases' section on the github repository page.

.. code-block:: bash

	$ pip install git+https://github.com/yutanagano/sceptr.git@<VERSION_TAG>

Manual install
..............

You can also clone the repository, and from within your Python environment, navigate to the project root directory and run:

.. code-block:: bash

	$ pip install .

Note that even for manual installation, you still need `git-lfs` to properly de-reference the stub files at `git-clone`-ing time.

Troubleshooting
...............

A recent security update to `git` has resulted in some difficulties cloning repositories that rely on `git-lfs`.
This can result in an error message with a message along the lines of:

.. code-block:: bash

	$ fatal: active `post-checkout` hook found during `git clone`

If this happens, you can temporarily set the `GIT_CLONE_PROTECTION_ACTIVE` environment variable to `false` by prepending `GIT_CLONE_PROTECTION_ACTIVE=false` before the install command like below:

.. code-block:: bash

	$ GIT_CLONE_PROTECTION_ACTIVE=false pip install git+https://github.com/yutanagano/sceptr.git@<VERSION_TAG>

This is `a known issue <https://github.com/git-lfs/git-lfs/issues/5749>`_ for `git` version `2.45.1` and is fixed from version `2.45.2`.
