.. _contributing:

Contributing
============

Thank you for wanting to contribute to this library! We will try to make this 
an easy process for you. It is recommended that you read 
the :ref:`development<development>` page so that you can lint 
and test before submitting code. 
Checking that your PR passes the required testing and linting procedures will speed up
the acceptance of your PR.

Issues and bug reporting
------------------------

To report issues or bugs please create a new issue on 
the `pvops issues page <https://github.com/sandialabs/pvops/issues>`_.
Before submitting your bug report, please perform a cursory search 
to see if the problem has been already reported. If it has been reported, 
and the issue is still open, add a comment to the existing issue instead of opening a new issue.

Guidelines for effective bug reporting
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

- Use a clear descriptive title for the issue.

- Describe the steps to reproduce the problem, 
  the behavior you observed after following the steps, and the expected behavior.

- If possible, provide a simple example of the bug using pvOps example data.

- When relevant, provide information on your computing environment
  (operating system, python version, pvOps version or commit).

- For runtime errors, provide a function call stack.

Contributing code
-----------------

Software developers, within the core development team and external collaborators, 
are expected to follow standard practices to document and test new code. 
Software developers interested in contributing to the project are encouraged 
to create a Fork of the project and submit a Pull Request (PR) using GitHub.
Pull requests will be reviewed by the core development team.
Create a PR or help with other PRs which are in the library 
by referencing `pvops PR page <https://github.com/sandialabs/pvops/pulls>`_. 

Guidelines for preparing and submitting pull-requests
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

- Use a clear descriptive title for your pull-requests

- Describe if your submission is a bugfix, documentation update, or a feature
  enhancement. Provide a concise description of your proposed changes. 
  
- Provide references to open issues, if applicable, to provide the necessary
  context to understand your pull request
  
- Make sure that your pull-request merges cleanly with the `master` branch of
  pvOps. When working on a feature, always create your feature branch off of
  the latest `master` commit
  
- Ensure that appropriate documentation and tests accompany any added features.
  
  
Once a pull-request is submitted you will iterate with NuMAD maintainers
until your changes are in an acceptable state and can be merged in. You can push
addditional commits to the branch used to create the pull-request to reflect the
feedback from maintainers and users of the code.

Questions and feature requests
--------------------------------

For any questions regarding pvOps or requests for additional features,
users may open an issue on the `pvops issues page <https://github.com/sandialabs/pvops/issues>`_
or contact the package maintainer found in ``setup.py``.

Guidelines for effective bug reporting
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

- Use a clear, descriptive title for the question or feature request.

- If submitting a feature request, please provide context, examples, and
  references when relevant.