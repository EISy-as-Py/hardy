Guide to numerical transformations
==================================

Following numerical transformations are currently defined in the
:code:`HARDy`

.. automodule:: hardy.arbitrage.transformations
   :members:

Defining your own numerical transformations
===========================================

User defined numerical transformations can be integrated in the
:code:`HARDy` by integrating the transformation definition
inside the :code:`hardy.arbitrage.transformations` module. The
example transformation definition is shown below::

    def transformation_function(args):
        y = f(x)

Most of the transformations defined in :code:`HARDy` are one
dimensional transformations i-e they require only one arguments.
If the user want to include more than one argument and metadata
(exponents), refer to function definition of :code:`power` or
:code:`derivative_2d` defined in :code:`hardy.arbitrage.transformations`
module. If the metadata is different than the length of metadata in
:code:`power`, function :code:`apply_tform` in
:code:`hardy.arbitrage.arbitrage` needs to be modified
accordingly as well.


