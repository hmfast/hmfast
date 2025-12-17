Installation
============


To use ``hmfast``, you'll need to install the package and make sure you have the relevant data, such as the emulator files.
The recommended way to get started is as follows:

1. **Install the repository:**

   To install the latest stable version, simply pip install it.

   .. code-block:: bash

      pip install hmfast

   If you wish to install the developer version, clone the main repo (``https://github.com/hmfast/hmfast``) and pip install it locally.

  

2. **Import the package and begin:**

   You may now import the package. 
   ``hmfast`` relies on pre-trained cosmological emulators perform quick calculations.
   These files are quite large, so ``hmfast`` will automatically download a subset of them at import if they are not already present.
   
   By default, they are stored in ``~/hmfast_data``.  

   If you want to use a different location for these files, simply uncomment the first two lines below and set your preferred path.

   If you want to download all emulators (~1 GB), simply uncomment the last line below.

   .. code-block:: python

      # import os
      # os.environ["HMFAST_DATA_PATH"] = "path/to/hmfast_data"
      import hmfast
      # hmfast.download_emulators(models="all")

   
After these steps, you should be all set to start using ``hmfast``.

