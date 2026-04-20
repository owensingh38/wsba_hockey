Plotting
========

Event plots (recommended)
-------------------------

`nhl_plot_events` is the unified plotting entrypoint for event maps.

.. autofunction:: wsba_hockey.wsba_main.nhl_plot_events

Examples
^^^^^^^^

Plot events for a single game:

.. code-block:: python

   figs = wsba.nhl_plot_events(
       pbp,
       group="game",
       entities=[2025030131],
       events=["goal", "shot-on-goal", "hit"],
       strengths="all",
       season_types=[2, 3],
   )

Plot events for skaters by NHL API player_id:

.. code-block:: python

   figs = wsba.nhl_plot_events(
       pbp,
       group="skater",
       entities=[8473533, 8476921],
       season=20252026,
       events=["goal", "shot-on-goal"],
       strengths=["5v5"],
       season_types=[2, 3],
   )

Heatmaps
--------

.. autofunction:: wsba_hockey.wsba_main.nhl_plot_heatmap

