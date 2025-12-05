Installation
============


To use ``hmfast``, you'll need to install the package and make sure you have the relevant data, such as the emulator files.
The recommended way to get started is as follows:

1. **Install the repository:**

   To install the latest stable version, simply pip install it.

   .. code-block:: bash

      pip install hmfast

   To install the developer version, clone the repository as follows and pip install it locally.

   .. code-block:: bash

      git clone https://github.com/hmfast/hmfast.git
      pip install /your/path/to/hmfast

  

2. **Import the package and begin:**

   You may now import the package. 
   Since the emulators are large, ``hmfast`` will download them at import if they do not already exist.
   By default, they will be downloaded to ~/hmfast_data. 
   If you wish to change this, simply uncomment the first two lines below and set your desired path.

   .. code-block:: python

      # import os
      # os.environ["HMFAST_DATA_PATH"] = "path/to/hmfast_data"
      import hmfast
      

After these steps, you should be all set to start using ``hmfast``.

