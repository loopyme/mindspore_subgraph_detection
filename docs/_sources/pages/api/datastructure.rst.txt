==============
DataStructures
==============

.. currentmodule:: SubgraphDetection.DataStructure
.. toctree::

.. autosummary::
   :toctree: _autosummary

   SNode
   SMSGraph
   Subgraph
   SubgraphCore

*****
SNode
*****

.. autoclass:: SubgraphDetection.DataStructure.SNode
   :members:
   :private-members:
   :special-members:

********
SMSGraph
********

.. autoclass:: SubgraphDetection.DataStructure.SMSGraph
   :members:
   :private-members:
   :special-members:

********
Subgraph
********
.. Warning:: 

   May contrary to intuition, but subgraph is not a subclass of SMSGraph, it's actually efficiently store all infomations about a set of isomorphic subgraphs. See `算法简述-名词解释 <../algorithm.html#id2>`_ for more info.

.. autoclass:: SubgraphDetection.DataStructure.Subgraph
   :members:
   :private-members:
   :special-members:

************
SubgraphCore
************

.. Note:: 

   A Subgraph Core won't transform into a subgraph object untill finish growing.

.. autoclass:: SubgraphDetection.DataStructure.SubgraphCore
   :members:
   :private-members:
   :special-members:
