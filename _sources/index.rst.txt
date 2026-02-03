.. aspsim documentation master file, created by
   sphinx-quickstart on Wed Sep 13 10:22:13 2023.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Audio Signal Processing Simulator (aspsim)
==========================================
The package simulates sound in a reverberant space. The simulator supports moving microphones and moving sources. The simulator supports adaptive processing, where the sound of the sources are determined by the sound in the room. This is useful for instance for active noise control and similar adaptive processing tasks.


.. toctree::
   :maxdepth: 1
   :caption: User Guide
   
   output_files
   moving_mics_and_sources
   logger


.. autosummary::
   :toctree: _autosummary
   :template: new-module-template.rst
   :caption: API Reference
   :recursive:

   aspsim.diagnostics
   aspsim.room
   aspsim.signal
   aspsim.array
   aspsim.configutil
   aspsim.counter
   aspsim.fileutilities
   aspsim.processor
   aspsim.saveloadsession
   aspsim.simulator
   aspsim.utilities

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`