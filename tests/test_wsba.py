import os
import matplotlib.pyplot as plt
import polars as pl
import wsba_hockey as wsba

def test_wsba():
    sample_dir = os.path.join(os.path.dirname(os.path.realpath(__file__)), "samples")

    # Reuse one session across every network scrape in this test.
    with wsba.make_pooled_session() as session:
        # Test scrape of random games and exported rosters
        pbp, rosters = wsba.nhl_scrape_game(
            ["random", 3, 2007, 2024],
            xg=True,
            export_roster=True,
            session=session,
        )
        pbp.write_csv(f"{sample_dir}/sample_random_pbp.csv")
        rosters.write_csv(f"{sample_dir}/sample_random_game_rosters.csv")
        assert pbp.schema["game_time"] == pl.String
        assert pbp.filter(~pl.col("game_time").str.contains(r"^\d+:\d{2}$")).is_empty()

        # Verify legacy games with no third event player retain the nullable schema.
        legacy_pbp = wsba.nhl_scrape_game(2007020043, session=session)
        assert "event_player_3_id" in legacy_pbp.columns

        # Retrieve skater game stats at 5v5
        game_stats = wsba.nhl_calculate_stats(pbp, "skater", "5v5")
        game_stats.write_csv(f"{sample_dir}/sample_game_stats.csv")

        # Find total sample stats and sort by goals
        stats = wsba.nhl_agg_stats(game_stats, ["player_id"], sort={"goals": False}, comparison=False)
        stats.write_csv(f"{sample_dir}/sample_agg_stats.csv")

        # Plot xG shots in the first random game at even strength and then save to file
        game_1 = pbp["game_id"].unique()[0]
        plots_1 = wsba.nhl_plot_events(
            pbp,
            "game",
            game_1,
            events=wsba.FENWICK_EVENTS,
            strengths=["5v5", "4v4", "3v3"],
            season_types=[2, 3],
            legend=True
        )
        plots_1[game_1].savefig(f"{sample_dir}/sample_game_plot.png", bbox_inches="tight")

        # Plot a skater selected from an actual Fenwick event.
        skater_candidates = (
            pbp
            .filter(
                pl.col("event_type").is_in(wsba.FENWICK_EVENTS)
                & pl.col("event_player_1_id").is_not_null()
            )
            .select(["game_id", "event_player_1_id"])
            .with_columns(pl.col("event_player_1_id").cast(pl.Int64, strict=False))
            .drop_nulls()
            .unique(maintain_order=True)
        )
        skater = int(skater_candidates["event_player_1_id"][0])
        skater_name = wsba.nhl_scrape_player_info([skater], session=session)["player_name"][0]
        plots_2 = wsba.nhl_plot_events(
            pbp,
            "skater",
            skater,
            events=wsba.FENWICK_EVENTS,
            season_types=[2, 3],
            display_range="offense",
            rotation=90,
            titles=f"{skater_name} Fenwick Shots",
            legend=True,
        )
        plots_2[skater].savefig(f"{sample_dir}/sample_skater_plot.png", bbox_inches="tight")

        # Standings Scraping
        standings = wsba.nhl_scrape_standings(20222023, session=session)
        standings.write_csv(f"{sample_dir}/sample_standings.csv")

        # Get edge data for David Pastrnak and Morgan Geekie in 2025-26
        edge_data = wsba.nhl_scrape_edge(20252026, "skater", [8477956, 8479987], session=session)
        edge_data.with_columns([
            pl.col(column).map_elements(str, return_dtype=pl.String)
            for column in edge_data.columns
            if isinstance(edge_data[column].dtype, (pl.List, pl.Struct, pl.Array))
        ]).write_csv(f"{sample_dir}/sample_edge_data.csv")

        # Collect animation for all goals in the game we plotted above (game_1):
        events_game_id = 2025021000
        events_pbp = wsba.nhl_scrape_game(events_game_id, session=session)
        event_ids = events_pbp.filter(pl.col("ppt_replay_url").is_not_null()).get_column("ppt_replay_url").str.slice(-8, 3).to_list()

        event_data = wsba.nhl_scrape_event_data(
            {events_game_id: event_ids},
            session=session,
        )
        event_data.write_csv(f"{sample_dir}/sample_event_data.csv")

        # Collect schema of play-by-play dataframe
        pbp_schema = wsba.utility_get_schema(pbp)
        pbp_schema.with_columns([pl.col(column).cast(pl.String) for column in pbp_schema.columns]).write_csv(f"{sample_dir}/sample_schema.csv")

    plt.close("all")

if __name__ == "__main__":
    test_wsba()
