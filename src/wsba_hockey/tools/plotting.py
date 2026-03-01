import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import wsba_hockey.wsba_main as wsba
from matplotlib.colors import Normalize
from hockey_rink import NHLRink
from hockey_rink import CircularImage
from scipy.interpolate import griddata
from scipy.ndimage import gaussian_filter
from scipy.stats import binned_statistic_2d

### PLOTTING FUNCTIONS ###
# Provided in this file are basic plotting functions for the WSBA Hockey Python package. #

def wsba_rink(display_range='offense',rotation = 0):
    rink = NHLRink(center_logo={
        "feature_class": CircularImage,
        "image_path": wsba.IMG_PATH,
        "length": 25, "width": 25,
        "x": 0, "y": 0,
        "radius": 14,    
        "zorder": 11,
        }
        )
    rink.draw(
            display_range=display_range,
            rotation=rotation,
            despine=True
        )

def prep_plot_data(pbp,strengths,season_types=2,marker_dict=wsba.EVENT_MARKERS):
    try: pbp['xG']
    except:
        pbp = wsba.wsba_xG(pbp)
        pbp['xG'] = np.where(pbp['xG'].isna(),0,pbp['xG'])

    #This column is useful for finding player shots
    pbp['wsba_id'] = pbp['event_player_1_id'].astype(str)+pbp['season'].astype(str)+pbp['event_team_abbr']
    
    pbp['event_team_abbr_for'] = pbp['event_team_abbr']
    pbp['event_team_abbr_against'] = np.where(pbp['event_team_venue']=='home',pbp['away_team_abbr'],pbp['home_team_abbr'])
    
    pbp['x_plot'] = np.where(pbp['x_adj']<0,-pbp['y_adj'],pbp['y_adj'])
    pbp['y_plot'] = abs(pbp['x_adj'])

    pbp['home_on_ice'] = pbp['home_on_1'].astype(str) + ";" + pbp['home_on_2'].astype(str) + ";" + pbp['home_on_3'].astype(str) + ";" + pbp['home_on_4'].astype(str) + ";" + pbp['home_on_5'].astype(str) + ";" + pbp['home_on_6'].astype(str)
    pbp['away_on_ice'] = pbp['away_on_1'].astype(str) + ";" + pbp['away_on_2'].astype(str) + ";" + pbp['away_on_3'].astype(str) + ";" + pbp['away_on_4'].astype(str) + ";" + pbp['away_on_5'].astype(str) + ";" + pbp['away_on_6'].astype(str)

    pbp['home_on_ice_id'] = pbp['home_on_1_id'].astype(str) + ";" + pbp['home_on_2_id'].astype(str) + ";" + pbp['home_on_3_id'].astype(str) + ";" + pbp['home_on_4_id'].astype(str) + ";" + pbp['home_on_5_id'].astype(str) + ";" + pbp['home_on_6_id'].astype(str)
    pbp['away_on_ice_id'] = pbp['away_on_1_id'].astype(str) + ";" + pbp['away_on_2_id'].astype(str) + ";" + pbp['away_on_3_id'].astype(str) + ";" + pbp['away_on_4_id'].astype(str) + ";" + pbp['away_on_5_id'].astype(str) + ";" + pbp['away_on_6_id'].astype(str)

    pbp['onice_for_name'] = np.where(pbp['home_team_abbr']==pbp['event_team_abbr'],pbp['home_on_ice'],pbp['away_on_ice'])
    pbp['onice_against_name'] = np.where(pbp['away_team_abbr']==pbp['event_team_abbr'],pbp['home_on_ice'],pbp['away_on_ice'])

    pbp['onice_for_id'] = np.where(pbp['home_team_abbr']==pbp['event_team_abbr'],pbp['home_on_ice_id'],pbp['away_on_ice_id'])
    pbp['onice_against_id'] = np.where(pbp['away_team_abbr']==pbp['event_team_abbr'],pbp['home_on_ice_id'],pbp['away_on_ice_id'])

    pbp['strength_state_for'] = pbp['strength_state']
    pbp['strength_state_against'] = pbp['strength_state'].str[::-1]

    pbp['onice_id'] = pbp['onice_for_id'].astype(str) + ";" + pbp['onice_against_id'].astype(str)
    pbp['onice_name'] = pbp['onice_for_name'].astype(str) + ";" + pbp['onice_against_name'].astype(str)
    pbp['event_team_abbrs'] = pbp['event_team_abbr_for'] + ";" + pbp['event_team_abbr_against']

    pbp['size'] = np.where(pbp['xG']<0.05,20,pbp['xG']*400)
    pbp['marker'] = pbp['event_type'].replace(marker_dict)

    if isinstance(season_types, int):
        season_types = [season_types]

    pbp = pbp.loc[pbp['season_type'].isin(season_types)]

    if strengths != 'all':
        pbp = pbp.loc[(pbp['strength_state_for'].isin(strengths)) | (pbp['strength_state_against'].isin(strengths))]

    pbp['is_shot'] = pbp['event_type'].isin(wsba.METRIC_EVENTS['Shots']).astype(int)
    pbp['is_fenwick'] = pbp['event_type'].isin(wsba.METRIC_EVENTS['Fenwick']).astype(int)
    pbp['is_give'] = (pbp['event_type'] == 'giveaway').astype(int)
    pbp['is_take'] = (pbp['event_type'] == 'takeaway').astype(int)

    return pbp

def heatmap_data(df, sit, toi, metric):
    #Helper function to create the data necessary to plot data

    #Move shots to either the left (against) or the right(for)
    if sit == 'for':
        df['x'] = abs(df['x_adj'])
        df['y'] = np.where(df['x_adj']<0,-df['y_adj'],df['y_adj'])
        df['event_distance'] = abs(df['event_distance'].fillna(0))
        df = df.loc[(df['event_distance']<=89)&(df['x']<=89)]

        x_min = 0
        x_max = 100

    else:
        df['x'] = -abs(df['x_adj'])
        df['y'] = np.where(df['x_adj']>0,-df['y_adj'],df['y_adj'])
        df['event_distance'] = -abs(df['event_distance'].fillna(0))
        df = df.loc[(df['event_distance']>-89)&(df['x']>-89)]

        x_min = -100
        x_max = 0
    
    x_bins = np.linspace(x_min,x_max,(x_max-x_min)+1)
    y_bins = np.linspace(-42.5,42.5,86)

    #Using this binning allows a more seamless swap between different types of metrics
    heatmap, _, _, _ = binned_statistic_2d(
        df['x'],
        df['y'],
        df[wsba.METRIC_LOOKUP[metric]],
        statistic='sum',
        bins=[x_bins, y_bins]
    )

    heatmap = np.where(heatmap<0, 0, heatmap)
    
    #Convert toi to minutes and normalize each coordinate to per-sixty values
    toi /= 60
    heatmap = (heatmap / toi)*60

    #Smooth out the array
    heatmap_smooth = gaussian_filter(heatmap, sigma=3)

    x_centers = (x_bins[:-1] + x_bins[1:]) / 2
    y_centers = (y_bins[:-1] + y_bins[1:]) / 2
    [x,y] = np.meshgrid(x_centers, y_centers)

    return x, y, heatmap_smooth

def gen_heatmap(pbp, player, season, team, strengths, season_types = 2, metric = 'Fenwick', compare = False, strengths_title = None, title = None):
    if isinstance(player, int):
        id_mod = '_id'
    elif player:
        id_mod = '_name'
        player = player.upper()
    else:
        id_mod = None
    
    pbp = pbp.loc[(pbp['season']==season)]
    
    pbp = prep_plot_data(pbp, strengths, season_types)

    events = wsba.METRIC_EVENTS[metric]

    #Quick helper filtering down the pbp df AFTER finding time_on_ice
    def filter_pbp(df):
        df = df.loc[df['event_type'].isin(events)]

        df = df.fillna(0)
        df = df.loc[(df['x_adj'].notna())&(df['y_adj'].notna())&(df['empty_net']==0)]

        return df

    fig, ax = plt.subplots(1, 1, figsize=(10,12), facecolor='w', edgecolor='k')
    wsba_rink(display_range='full')

    #Find TOI for league and player/team
    if compare:
        league_shots_pre = pbp.loc[(pbp['strength_state_for'].isin(strengths) if strengths != 'all' else True)|
                                (pbp['strength_state_against'].isin(strengths) if strengths != 'all' else True)]
        
        #Multiply by two since each team must have their own TOI sum
        league_toi = (league_shots_pre['event_length'].sum())*2

    player_shots_pre = pbp.loc[(pbp[f'onice{id_mod}'].str.contains(str(player)))&
                               (pbp[f'event_team_abbrs'].str.contains(team))] if player else pbp.loc[(pbp[f'event_team_abbrs'].str.contains(team))&
                                                                                                     ((pbp['strength_state_for'].isin(strengths) if strengths != 'all' else True)|
                                                                                                      (pbp['strength_state_against'].isin(strengths) if strengths != 'all' else True))]
    player_toi = player_shots_pre['event_length'].sum()

    for sit in ['for','against']:
        if compare:
            league_shots = filter_pbp(league_shots_pre)
            league_shots = league_shots.loc[(league_shots[f'strength_state_{sit}'].isin(strengths) if strengths != 'all' else True)]
            _, _, league_smooth = heatmap_data(league_shots, sit, league_toi, metric)
        else:
            league_smooth = 0

        player_shots = filter_pbp(player_shots_pre.loc[(player_shots_pre[f'onice_{sit}{id_mod}'].str.contains(str(player)))&
                                            (player_shots_pre[f'event_team_abbr_{sit}']==team)&
                                            (player_shots_pre[f'strength_state_{sit}'].isin(strengths) if strengths != 'all' else True)
                                            ] if player else player_shots_pre.loc[(player_shots_pre[f'event_team_abbr_{sit}']==team)&
                                                                                  (player_shots_pre[f'strength_state_{sit}'].isin(strengths) if strengths != 'all' else True)])

        x, y, player_smooth = heatmap_data(player_shots, sit, player_toi, metric)
                
        difference = player_smooth - league_smooth
        data_min= difference.min()
        data_max= difference.max()
    
        if abs(data_min) > data_max:
            data_max = data_min * -1
        elif data_max > abs(data_min):
            data_min = data_max * -1
        
        cont = ax.contourf(
            x, y, difference.T,
            alpha=0.6,
            cmap='bwr',
            levels=np.linspace(data_min, data_max, 12),
            norm=Normalize(vmin=data_min,vmax=data_max),
            vmin=data_min,
            vmax=data_max,
            zorder=5
        )
    
    ax.text(-50, -50, 'Defense (Against)', ha='center', va='bottom', fontsize=12)
    ax.text(50, -50, 'Offense (For)', ha='center', va='bottom', fontsize=12)
    
    cbar = fig.colorbar(
        cont,
        ax=ax,
        orientation='horizontal',
        shrink=0.75,
        aspect=50,
        fraction=0.05,
        pad=0.05,
        ticks=[data_min, 0 ,data_max]
    )

    cbar.ax.set_xticklabels(['Deficit', 'Even', 'Surplus'])
    cbar.set_label(f'On-Ice {metric}/60, Compared to League Average', fontsize=12)
    
    if not strengths_title:
        strengths_title = ', '.join(strengths) if np.logical_and(isinstance(strengths, list),strengths_title==None) else strengths.title()
    
    fig.text(0.51, 0.175, f'Strength(s)', ha='center', fontsize=10)
    fig.text(0.51, 0.155, f'{strengths_title}', ha='center', fontsize=10)

    plt.title(title)

    return fig

def plot_skater_shots(pbp, player, season, team, strengths, season_types = 2, strengths_title = None, title = None, marker_dict=wsba.EVENT_MARKERS, situation='for', legend=False):
    shots = ['goal','missed-shot','shot-on-goal']
    pbp = prep_plot_data(pbp,strengths,season_types,marker_dict)
    pbp = pbp.loc[(pbp['season']==season)&(pbp['event_type'].isin(shots))&((pbp['away_team_abbr']==team)|(pbp['home_team_abbr']==team))]

    team_data = pd.read_csv(wsba.INFO_PATH)
    team_color = list(team_data.loc[team_data['wsba_id']==f'{team}{season}','primary_color'])[0]
    team_color_2nd = list(team_data.loc[team_data['wsba_id']==f'{team}{season}','secondary_color'])[0]

    if isinstance(player, int):
        id_mod = '_id'
    else:
        id_mod = '_name'
        player = player.upper()

    if situation in ['for','against']:
        skater = pbp.loc[(pbp[f'onice_{situation}{id_mod}'].str.contains(str(player))) ]
        skater['color'] = np.where(skater[f'event_player_1{id_mod}']==player,team_color,team_color_2nd)
    else:
        skater = pbp.loc[pbp[f'event_player_1{id_mod}']==player]
        skater['color'] = team_color

    fig, ax = plt.subplots()
    wsba_rink(rotation=90)

    for event in shots:
        plays = skater.loc[skater['event_type']==event]
        ax.scatter(plays['x_plot'],plays['y_plot'],plays['size'],plays['color'],marker=wsba.EVENT_MARKERS[event],label=event,edgecolors='black' if event=='goal' else 'white',linewidths=0.75,zorder=5)
    
    ax.set_title(title) if title else ''
    ax.legend().set_visible(legend)
    ax.legend().set_zorder(1000)
    
    if not strengths_title:
        strengths_title = ', '.join(strengths) if np.logical_and(isinstance(strengths, list),strengths_title==None) else strengths.title()

    fig.text(0.5, 0.07, f'Strength(s)', ha='center', fontsize=10)
    fig.text(0.5, 0.03, f'{strengths_title}', ha='center', fontsize=10)

    return fig
    
def plot_game_events(pbp,game_id,events,strengths,marker_dict=wsba.EVENT_MARKERS,team_colors={'away':'secondary','home':'primary'},legend=False):
    pbp = prep_plot_data(pbp,strengths,season_types=[2,3],marker_dict=marker_dict)
    pbp = pbp.loc[(pbp['event_type'].isin(events))&(pbp['game_id']==game_id)]
    
    away_abbr = pbp['away_team_abbr'].iloc[0]
    home_abbr = pbp['home_team_abbr'].iloc[0]
    date = pbp['game_date'].iloc[0]
    season = pbp['season'].iloc[0]
    away_xg = pbp.loc[pbp['event_team_venue']=='away','xG'].sum().astype(float).round(2)
    home_xg = pbp.loc[pbp['event_team_venue']=='home','xG'].sum().astype(float).round(2)

    team_data = pd.read_csv(wsba.INFO_PATH)
    team_info ={
        'away_color':'#000000' if list(team_data.loc[team_data['wsba_id']==f'{away_abbr}{season}','secondary_color'])[0]=='#FFFFFF' else list(team_data.loc[team_data['wsba_id']==f'{away_abbr}{season}',f'{team_colors['away']}_color'])[0],
        'home_color': list(team_data.loc[team_data['wsba_id']==f'{home_abbr}{season}',f'{team_colors['home']}_color'])[0],
        'away_logo': f'tools/logos/png/{away_abbr}{season}.png',
        'home_logo': f'tools/logos/png/{home_abbr}{season}.png',
    }

    pbp['color'] = np.where(pbp['event_team_abbr']==away_abbr,team_info['away_color'],team_info['home_color'])

    fig, ax = plt.subplots()
    wsba_rink(display_range='full')

    for event in events:
        plays = pbp.loc[pbp['event_type']==event]
        ax.scatter(plays['x_adj'],plays['y_adj'],plays['size'],plays['color'],marker=wsba.EVENT_MARKERS[event],edgecolors='black' if event=='goal' else 'white',linewidths=0.75,label=event,zorder=5)

    ax.text(-50, -50, f'{away_abbr} xG: {away_xg}', ha='center', va='bottom', fontsize=10)
    ax.text(50, -50, f'{home_abbr} xG: {home_xg}', ha='center', va='bottom', fontsize=10)
    ax.set_title(f'{away_abbr} @ {home_abbr} - {date}')
    ax.legend(handles=wsba.LEGEND_ELEMENTS, bbox_to_anchor =(0.5,-0.35), loc='lower center', ncol=1).set_visible(legend)

    return fig

def plot_game_score(df):
    plots = {}

    teams = df['team_abbr'].drop_duplicates()
    schedule = pd.read_csv(wsba.SCHEDULE_PATH)
    game_date = schedule.loc[schedule['game_id']==df['game_id'].astype(int).iloc[0], 'game_date'].iloc[0]

    for team in teams:
        plot_df = df.loc[df['team_abbr'] == team].copy()
            
        opp = [t for t in teams if t != team]

        plot_df['player_name'] = plot_df['player_name'].str.title()

        comp = [
            'production_score',
            'play_driving_score',
            'even_strength_score',
            'power_play_score',
            'short_handed_score',
            'penalties_score',
            'puck_management_score',
            'faceoffs_score',
            'workload_score',
            'goaltending_score'
        ]

        labels = [
            'Production',
            'Play-Driving',
            'Even Strength',
            'Power Play',
            'Shorthanded',
            'Penalties',
            'Puck Management',
            'Faceoffs',
            'Goaltending Workload',
            'Goaltending Performance'
        ]

        colors = [
            'blue', 'red', 'green', 'orange', 'purple',
            'grey', 'cyan', 'magenta', 'brown', 'pink'
        ]

        fig, ax = plt.subplots(figsize=(10, 8))
        y_pos = np.arange(len(plot_df['player_name']))

        right_offset = np.zeros(len(plot_df))
        
        left_offset = np.zeros(len(plot_df)) 

        for i, col in enumerate(comp):
            values = plot_df[col]
            
            pos_vals = np.where(values > 0, values, 0)
            neg_vals = np.where(values < 0, values, 0)

            ax.barh(y_pos, pos_vals, left=right_offset, height=0.6,
                    color=colors[i], label=labels[i])
            
            right_offset += pos_vals 

            ax.barh(y_pos, neg_vals, left=left_offset, height=0.6,
                    color=colors[i])
            
            left_offset += neg_vals 

        ax.axvline(0, color='black', linewidth=1)
        ax.set_yticks(y_pos)
        ax.set_yticklabels(
            plot_df['player_name'] + ' (' +
            plot_df['game_score'].astype(float).round(2).astype(str) + ')'
        )
        ax.invert_yaxis()
        ax.set_xlabel('Game Score')
        ax.set_ylabel('Player')

        x_min = min(-3, np.min(left_offset))
        x_max = max(3, np.max(right_offset))
        ax.set_xlim(x_min, x_max)

        ax.grid(True, axis='both')
        
        title = f'{team} Game Score - {game_date} vs {opp[0]}'
        ax.set_title(title)
        
        legend = ax.legend(title='Components', bbox_to_anchor=(1.05, 1), loc='upper left')

        logo_img = mpimg.imread(wsba.IMG_PATH)
        fig.canvas.draw()
        bbox = legend.get_window_extent().transformed(fig.transFigure.inverted())
        
        logo_width = bbox.width
        logo_height = 0.3
        logo_bottom = bbox.y0 - 0.4
        logo_ax = fig.add_axes([bbox.x0, logo_bottom, logo_width, logo_height])
        logo_ax.imshow(logo_img)
        logo_ax.axis('off')

        plots[team] = fig

    return plots