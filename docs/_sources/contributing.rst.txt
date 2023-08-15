Contributing
============


The main branch on the GitHub repository is `protected <https://docs.github.com/en/repositories/configuring-branches-and-merges-in-your-repository/managing-protected-branches/managing-a-branch-protection-rule>`_ so that commits to the main branch are managed through pull requests rather than direct commits. To contribute to the codebase you will need to first need to create a development branch. Any proposed changes can be managed through a pull request. It is often useful to first create an `issue <https://github.com/NewcastleRSE/xray-spectroscopy-ml/issues>`_ for the bug fix or feature and then link a development branch to this issue. Users can also add issues without contributing to the code and there are multiple templates available to guide the formatting of these.

-------------
Documentation
-------------

This documentation is build and deployed to GitHub Pages automatically using GitHub actions. Content is stored in the ``sphinx/*.rst`` files using the `reStructuredText <https://www.sphinx-doc.org/en/master/usage/restructuredtext/basics.html>`_ syntax. To edit or add to the documentation you will need to follow these steps,

* Install the requirements. Building the documentation requires the ``sphinx`` and ``sphix_rtd_theme``  Python packages to be installed. These are now included in the requirements file:

	.. code-block::

		pip install -r requirements.txt

* To add a new page to the documentation, add a new file to the ``sphinx`` directory with the extension ``.rst``. Otherwise, open up the file you wish to edit in your editor of choice and make your changes.

* To check your build locally, run the following command in the ``sphinx`` directory:

 	.. code-block::

 		make html

	This will build a local copy you can preview by clicking on one of the created html files in the ``sphinx/_build/html`` directory

* When you are happy with your changes, commit and push them to your development branch. This will trigger the GitHub action and deploy to GitHub pages.


----
Code
----

**Adding a new model**

	Coming soon!