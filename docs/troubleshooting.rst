Troubleshooting
===============

Installing on Python :math:`\geq` 3.13 on Windows
-------------------------------------------------

User reports / CI testing have suggested that up to version ``1.1.1``,
installing SCEPTR on Python :math:`\geq` 3.13 on Windows results in an error.
This issue is fixed from version ``1.2.0`` onwards. If Windows users notice any
persisting problems despite using the latest version of SCEPTR, please submit
an issue on the `GitHub repository
<https://github.com/yutanagano/sceptr/issues/new>`_ to notify the maintainers.

*Mus musculus* TR genes / alleles not properly recognised as functional
-----------------------------------------------------------------------

This was `a known bug <https://github.com/yutanagano/sceptr/issues/58>`_ with
old versions of the `libtcrlm <https://github.com/yutanagano/libtcrlm>`_
dependency. This has now been fixed thanks to a contribution from `@celinebalaa
<https://github.com/celinebalaa>`_ (thank you!). If you are still having this issue,
please make sure your Python environment has libtcrlm updated to version 1.1.2
or newer, and if the issue is persisting please open an `issue on the GitHub
repository <https://github.com/yutanagano/sceptr/issues/new>`_.

Error from git when installing from source (versions :math:`\leq` ``1.1.0``)
----------------------------------------------------------------------------

A recent security update to ``git`` has resulted in some difficulties cloning
repositories that rely on ``git-lfs``. This can result in an error message with
a message along the lines of:

.. code-block:: console

	$ fatal: active `post-checkout` hook found during `git clone`

If this happens, you can temporarily set the ``GIT_CLONE_PROTECTION_ACTIVE``
environment variable to ``false`` by prepending
``GIT_CLONE_PROTECTION_ACTIVE=false`` before the install command like below:

.. code-block:: console

	$ GIT_CLONE_PROTECTION_ACTIVE=false pip install git+https://github.com/yutanagano/sceptr.git@<VERSION_TAG>

This is `a known issue <https://github.com/git-lfs/git-lfs/issues/5749>`_ for
``git`` version ``2.45.1`` and is fixed from version ``2.45.2``.
