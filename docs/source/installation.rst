Installation
============


To use ``hmfast``, you'll need the source code and the auxiliary emulator files.  
The recommended way to get started is as follows:

1. **Clone the repository:**

   .. code-block:: bash

      git clone https://github.com/hmfast/hmfast.git


2. **Download emulator data files:**

   The fastest and easiest way is via Python. The following example will download the recommended emulator models (ede-v2) to your computer:

   .. code-block:: python

      import hmfast
      hmfast.download_emulators()

   By default, emulator files are saved to ``~/hmfast_data``.  
   If you want to change the location, pass a different path:

   .. code-block:: python

      hmfast.download.download_emulators(target_dir="/custom/path/for/hmfast_data")

   Or, set the environment variable ``HMFAST_EMULATOR_PATH`` before running Python:

   .. code-block:: bash

      export HMFAST_EMULATOR_PATH=/custom/path/for/hmfast_data

   To download *all* available emulator models, use:

   .. code-block:: python

      hmfast.download.download_emulators(models="all")

   You may also pass a list of model names, for example:

   .. code-block:: python

      hmfast.download.download_emulators(models=["ede-v2", "lcdm"])

After these steps, you are ready to use ``hmfast``.

