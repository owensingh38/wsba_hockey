Quickstart
==========

Installation
------------

.. code-block:: bash

   pip install wsba_hockey

Import
------

.. code-block:: python

   import wsba_hockey as wsba

Scrape a game
-------------

.. code-block:: python

   pbp = wsba.nhl_scrape_game(2024020918, split_shifts=False, remove=["game-end"])

Add expected goals
------------------

.. code-block:: python

   pbp = wsba.nhl_apply_xG(pbp)

