import os
import matplotlib.pyplot as plt
import wsba_hockey as wsba

def test_wsba():
    sample_dir = os.path.join(os.path.dirname(os.path.realpath(__file__)), "samples")

    # Test scrape of random games and exported rosters
    pbp, rosters = wsba.nhl_scrape_game(["random", 3, 2007, 2024], xg=True, export_roster=True)
    pbp.to_csv(f"{sample_dir}/sample_random_pbp.csv", index=False)
    rosters.to_csv(f"{sample_dir}/sample_random_game_rosters.csv", index=False)

    # Retrieve skater game stats at 5v5
    game_stats = wsba.nhl_calculate_stats(pbp, "skater", "5v5")
    game_stats.to_csv(f"{sample_dir}/sample_game_stats.csv", index=False)

    # Find total sample stats and sort by goals
    stats = wsba.nhl_agg_stats(game_stats, ["player_id"], sort={"goals": False}, comparison=False)
    stats.to_csv(f"{sample_dir}/sample_agg_stats.csv", index=False)

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

    # Plot the shots of a random skater in the second random game in the play-by-play data
    game_2 = int(pbp["game_id"].unique()[1])
    skater = int(pbp.loc[pbp["game_id"] == game_2, "home_on_3_id"].iloc[5])
    skater_name = wsba.nhl_scrape_player_info([skater])["player_name"].iloc[0]
    plots_2 = wsba.nhl_plot_events(
        pbp,
        "skater",
        skater,
        season_types=[2, 3],
        display_range="offense",
        rotation=90,
        titles=f"{skater_name} Fenwick Shots",
        legend=True,
    )
    plots_2[skater].savefig(f"{sample_dir}/sample_skater_plot.png", bbox_inches="tight")

    # Standings Scraping
    standings = wsba.nhl_scrape_standings(20222023)
    standings.to_csv(f"{sample_dir}/sample_standings.csv", index=False)

    # Get edge data for David Pastrnak and Morgan Geekie in 2025-26
    edge_data = wsba.nhl_scrape_edge(20252026, "skater", [8477956, 8479987])
    edge_data.to_csv(f"{sample_dir}/sample_edge_data.csv", index=False)

    # Collect animation for all goals in the game we plotted above (game_1):
    events_game_id = 2025021000
    events_pbp = wsba.nhl_scrape_game(events_game_id)
    event_ids = events_pbp.loc[
        (events_pbp["ppt_replay_url"].notna()), "ppt_replay_url"
    ].str[-8:-5].to_list()

    event_data = wsba.nhl_scrape_event_data(
        {events_game_id: event_ids}
    )
    event_data.to_csv(f"{sample_dir}/sample_event_data.csv", index=False)

    # Collect schema of play-by-play dataframe
    pbp_schema = wsba.utility_get_schema(pbp)
    pbp_schema.to_csv(f"{sample_dir}/sample_schema.csv", index=False)

    plt.close("all")

if __name__ == "__main__":
    test_wsba()
