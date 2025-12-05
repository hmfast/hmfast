Installation
============


To use ``hmfast``, you'll need the source code and the auxiliary emulator files.  
The recommended way to get started is as follows:

1. **Clone the repository:**

   .. code-block:: bash

      git clone https://github.com/hmfast/hmfast.git

   Then, make sure you pip install the local repository.

   .. code-block:: bash

      pip install /your/path/to/hmfast


2. **Import the package and begin:**

   You may now import the package. 
   Since the emulators are large, ``hmfast`` will download them at import if it they do not already exist.
   By default, they will be downloaded to ~/hmfast_data. 
   If you wish to change this, simply uncomment the first line below and set your desired path.

   .. code-block:: python

      # os.environ["HMFAST_DATA_PATH"] = "path/to/hmfast_data"
      import hmfast
      

After these steps, you should be all set to start using ``hmfast``.

