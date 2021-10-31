.. Installation doc

Installation
============

Installation using Conda
------------------------
The easiest way to install :code:`HARDy` is using :code:`conda`::
 
   conda install -c pozzorg hardy

Installation using Git
----------------------
:code:`HARDy` can also be installed using Git. Currently version 1.0
is the most stable version. To install version v1.0, follow the following
steps::

* In your terminal, run::
    git clone https://github.com/EISy-as-Py/hardy.git
* Change the directory to hardy root directory, by running::
    cd hardy
* Run::
    git checkout v1.0
* Run::
    python setup.py install</code>

To check installation run the following command in your terminal::
    
    python -c "import hardy"


Installation using ``evironment.yml`` (Recommended)
---------------------------------------------------
To avoid installing each dependency one by one, we recommend using
environment.yml provided in the github repository. To install the
environment run the following code in your terminal::

    conda install --name hardy --file environment.yml

To proceed with the installation, the hardy environment needs to be
acitivated through::

    conda activate hardy
    
The final step is run the installation command for :code:`HARDy`::

    conda install -c pozzorg hardy

To check installation run the following command in your terminal::
    
    python -c "import hardy"

Installation using pip
----------------------
:code:`HARDy` is also configured to be installed using pip. Currently
version 1.0 is the most stable version. To install version 1.0, run the
following commands in the terminal::

    git clone https://github.com/EISy-as-Py/hardy.git
    cd HARDy
    git checkout v1.0
    pip install .

To check installation run the following command in your terminal::
    
    python -c "import hardy"


.. toctree::
    :maxdepth: 1
    :glob: