import re
from bs4 import BeautifulSoup, SoupStrainer
import hockey_scraper.utils.shared as shared
import hockey_scraper.nhl.pbp.html_pbp as html
import hockey_scraper.nhl.game_scraper as gs
import numpy as np
import pandas as pd
import warnings
import requests as rs
from zipfile import ZipFile
warnings.filterwarnings('ignore')

### SCRAPING FUNCTIONS ###
# Provided in this file are functions vital to the scraping functions in the WSBA Hockey Python package. #

## JSON DATA ##
def retreive_players(json,result = "id"):
    #Given json data from an NHL API call, return dictionary with home and away players and either their id or their position.
    roster = pd.json_normalize(json['rosterSpots'])
    info = pd.json_normalize(json)
    home = info['homeTeam.id'][0]
    away = info['awayTeam.id'][0]

    #Add up to four alternative names for each player in the game
    roster['playerName'] = roster['firstName.default']+" "+roster['lastName.default']
    try: roster['playerName_2'] = roster['firstName.cs']+" "+roster['lastName.default'] 
    except: roster['playerName_2'] = ""
    try: roster['playerName_3'] = roster['firstName.de']+" "+roster['lastName.default']
    except: roster['playerName_3'] = ""
    try: roster['playerName_4'] = roster['firstName.es']+" "+roster['lastName.default']
    except: roster['playerName_4'] = ""

    #For each home/away player their name is included as a key and their id or position is the value
    home_players = {}
    home_id = roster.loc[roster['teamId']==home]
    hid = list(home_id['playerId'])+list(home_id['playerId'])+list(home_id['playerId'])+list(home_id['playerId'])
    hpos = list(home_id['positionCode'])+list(home_id['positionCode'])+list(home_id['positionCode'])+list(home_id['positionCode'])
    hp = list(home_id['playerName'])+list(home_id['playerName_2'])+list(home_id['playerName_3'])+list(home_id['playerName_4'])
    
    for id, pos, player in zip(hid,hpos,hp):
        try: home_players.update({player.upper():
                        {result:id if result == 'id' else pos}})
        except:
            continue

    away_players = {}
    away_id = roster.loc[roster['teamId']==away]
    aid = list(away_id['playerId'])+list(away_id['playerId'])+list(away_id['playerId'])+list(away_id['playerId'])
    apos = list(away_id['positionCode'])+list(away_id['positionCode'])+list(away_id['positionCode'])+list(away_id['positionCode'])
    ap = list(away_id['playerName'])+list(away_id['playerName_2'])+list(away_id['playerName_3'])+list(away_id['playerName_4'])
    
    for id, pos, player in zip(aid,apos,ap):
        try: away_players.update({player.upper():
                        {result:id if result == 'id' else pos}})
        except:
            continue
    
    #Return: Dict of away and home players keyed with id or position as value
    return {
        'home':home_players,
        'away':away_players
    }

def parse_json(json):
    #Given json data from an NHL API call, return play-by-play data.

    events = pd.json_normalize(json['plays']).reset_index(drop=True)
    info = pd.json_normalize(json)
    roster = pd.json_normalize(json['rosterSpots'])

    #Game information
    events['game_id'] = info['id'][0]
    events['season'] = info['season'][0]
    events['season_type'] = info['gameType'][0]
    events['game_date'] = info['gameDate'][0]
    events['start_time'] = info['startTimeUTC'][0]
    events['venue'] = info['venue.default'][0]
    events['venue_location'] = info['venueLocation.default'][0]
    events['away_team_id'] = info['awayTeam.id'][0]
    events['away_team_abbr'] = info['awayTeam.abbrev'][0]
    events['home_team_id'] = info['homeTeam.id'][0]
    events['home_team_abbr'] = info['homeTeam.abbrev'][0]

    teams = {
        info['awayTeam.id'][0]:info['awayTeam.abbrev'][0],
        info['homeTeam.id'][0]:info['homeTeam.abbrev'][0]
    }

    #Create player information dicts used to create event_player columns
    roster['playerName'] = roster['firstName.default']+" "+roster['lastName.default']
    players = {}
    players_pos = {}
    ids = {}
    for id, player in zip(list(roster['playerId']),list(roster['playerName'])):
        players.update({id:player.upper()})
    for id, pos in zip(list(roster['playerId']),list(roster['positionCode'])):
        players_pos.update({id:pos.upper()})
    for id, player in zip(list(roster['playerId']),list(roster['playerName'])):
        ids.update({player.upper():id})

    #Test columns
    cols = ['eventId', 'timeInPeriod', 'timeRemaining', 'situationCode', 'homeTeamDefendingSide', 'typeCode', 'typeDescKey', 'sortOrder', 'periodDescriptor.number', 'periodDescriptor.periodType', 'periodDescriptor.maxRegulationPeriods', 'details.eventOwnerTeamId', 'details.losingPlayerId', 'details.winningPlayerId', 'details.xCoord', 'details.yCoord', 'details.zoneCode', 'pptReplayUrl', 'details.shotType', 'details.scoringPlayerId', 'details.scoringPlayerTotal', 'details.assist1PlayerId', 'details.assist1PlayerTotal', 'details.assist2PlayerId', 'details.assist2PlayerTotal', 'details.goalieInNetId', 'details.awayScore', 'details.homeScore', 'details.highlightClipSharingUrl', 'details.highlightClipSharingUrlFr', 'details.highlightClip', 'details.highlightClipFr', 'details.discreteClip', 'details.discreteClipFr', 'details.shootingPlayerId', 'details.awaySOG', 'details.homeSOG', 'details.playerId', 'details.hittingPlayerId', 'details.hitteePlayerId', 'details.reason', 'details.typeCode', 'details.descKey', 'details.duration', 'details.servedByPlayerId', 'details.secondaryReason', 'details.blockingPlayerId', 'details.committedByPlayerId', 'details.drawnByPlayerId', 'game_id', 'season', 'season_type', 'game_date', 'away_team_id', 'away_team_abbr', 'home_team_id', 'home_team_abbr']

    for col in cols:
        try:events[col]
        except:
            events[col]=""

    #Event_player_columns include players in a given set of events; the higher the number, the greater the importance the event player was to the play
    events['event_player_1_id'] = events['details.winningPlayerId'].combine_first(events['details.scoringPlayerId'])\
                                                                   .combine_first(events['details.shootingPlayerId'])\
                                                                   .combine_first(events['details.playerId'])\
                                                                   .combine_first(events['details.hittingPlayerId'])\
                                                                   .combine_first(events['details.committedByPlayerId'])
        
    events['event_player_2_id'] = events['details.losingPlayerId'].combine_first(events['details.assist1PlayerId'])\
                                                                    .combine_first(events['details.hitteePlayerId'])\
                                                                    .combine_first(events['details.drawnByPlayerId'])\
                                                                    .combine_first(events['details.blockingPlayerId'])

    events['event_player_3_id'] = events['details.assist2PlayerId']

    events['event_team_status'] = np.where(events['home_team_id']==events['details.eventOwnerTeamId'],"home","away")

    #Coordinate adjustments:
    #The WSBA NHL Scraper includes three sets of coordinates per event:
    # x, y - Raw coordinates from JSON pbpp
    # x_fixed, y_fixed - Coordinates fixed to the right side of the ice (x is always greater than 0)
    # x_adj, y_adj - Adjusted coordinates configuring away events with negative x vlaues while home events are always positive
    events['x_fixed'] = abs(events['details.xCoord'])
    events['y_fixed'] = np.where(events['details.xCoord']<0,-events['details.yCoord'],events['details.yCoord'])
    events['x_adj'] = np.where(events['event_team_status']=="home",events['x_fixed'],-events['x_fixed'])
    events['y_adj'] = np.where(events['event_team_status']=="home",events['y_fixed'],-events['y_fixed'])
    events['event_distance'] = np.sqrt(((89 - events['x_fixed'])**2) + (events['y_fixed']**2))
    events['event_angle'] = np.degrees(np.arctan2(abs(events['y_fixed']), abs(89 - events['x_fixed'])))
    
    events['event_team_abbr'] = events['details.eventOwnerTeamId'].replace(teams)

    #Event player information includes ids (included in the JSON events), names (from "rosterSpots"), and positions (also from "rosterSpots")
    events['event_player_1_name'] = events['event_player_1_id'].replace(players)
    events['event_player_2_name'] = events['event_player_2_id'].replace(players)
    events['event_player_3_name'] = events['event_player_3_id'].replace(players)

    events['event_player_1_pos'] = events['event_player_1_id'].replace(players_pos)
    events['event_player_2_pos'] = events['event_player_2_id'].replace(players_pos)
    events['event_player_3_pos'] = events['event_player_3_id'].replace(players_pos)

    events['event_goalie_name'] = events['details.goalieInNetId'].replace(players)

    #Create situations given situation code (this is reconfigured with on ice skaters when provided shifts data)
    events['away_skaters'] = events['situationCode'].astype(str).str.slice(start=1,stop=2)
    events['home_skaters'] = events['situationCode'].astype(str).str.slice(start=2,stop=3)
    events['event_skaters'] = np.where(events['event_team_abbr']==events['home_team_abbr'],events['home_skaters'],events['away_skaters'])
    events['event_skaters_against'] = np.where(events['event_team_abbr']==events['home_team_abbr'],events['away_skaters'],events['home_skaters'])

    events['strength_state'] = events['event_skaters']+"v"+events['event_skaters_against']
    events['strength'] = np.where(events['event_skaters']==events['event_skaters_against'],
                                  "EV",np.where(
                                      events['event_skaters']>events['event_skaters_against'],
                                      "PP","SH"
                                  ))
    
    #Rename columns to follow WSBA naming conventions
    events = events.rename(columns={
        "eventId":"event_id",
        "periodDescriptor.number":"period",
        "periodDescriptor.periodType":"period_type",
        "timeInPeriod":"period_time_elasped",
        "timeRemaining":"period_time_remaining",
        "situationCode":"situation_code",
        "homeTeamDefendingSide":"home_team_defending_side",
        "typeCode":"event_type_code",
        "typeDescKey":"event_type",
        "details.shotType":"shot_type",
        "details.duration":"penalty_duration",
        "details.descKey":"penalty_description",
        "details.reason":"reason",
        "details.zoneCode":"zone_code",
        "details.xCoord":"x",
        "details.yCoord":"y",
        "details.goalieInNetId": "event_goalie_id",
        "details.awaySOG":"away_SOG",
        "details.homeSOG":"home_SOG"
    })

    #Period time adjustments (only 'seconds_elapsed' is included in the resulting data)
    events['period_time_simple'] = events['period_time_elasped'].str.replace(":","",regex=True)
    events['period_seconds_elapsed'] = np.where(events['period_time_simple'].str.len()==3,
                                           ((events['period_time_simple'].str[0].astype(int)*60)+events['period_time_simple'].str[-2:].astype(int)),
                                           ((events['period_time_simple'].str[0:2].astype(int)*60)+events['period_time_simple'].str[-2:].astype(int)))
    events['period_seconds_remaining'] = 1200-events['period_seconds_elapsed']
    events['seconds_elapsed'] = ((events['period']-1)*1200)+events['period_seconds_elapsed']
    
    #The following code is utilized to generate score and fenwick columns for each event
    fenwick_events = ['missed-shot','shot-on-goal','goal']
    ag = 0
    ags = []
    hg = 0
    hgs = []

    af = 0
    afs = []
    hf = 0
    hfs = []
    for event,team in zip(list(events['event_type']),list(events['event_team_status'])):
        if event in fenwick_events:
            if team == "home":
                hf = hf+1
                if event == 'goal':
                    hg = hg+1
            else:
                af = af+1
                if event == 'goal':
                    ag = ag+1
       
        ags.append(ag)
        hgs.append(hg)
        afs.append(af)
        hfs.append(hf)

    events['away_score'] = ags
    events['home_score'] = hgs
    events['away_fenwick'] = afs
    events['home_fenwick'] = hfs
    
    events = events.loc[(events['event_type']!="")&(events['event_type']!="game-end")]

    #Return: dataframe with parsed games in event
    return events



## HTML DATA ##
def get_soup(shifts_html):
    #Parses provided shifts html with BeautifulSoup
    #Utilizes method from Harry Shomer's hockey_scraper package
    parsers = ["lxml", "html.parser", "html5lib"]

    for parser in parsers:
        soup = BeautifulSoup(shifts_html, parser)
        td = soup.findAll(True, {'class': ['playerHeading + border', 'lborder + bborder']})

        if len(td) > 0:
            break
    return td, get_teams(soup)


def get_teams(soup):
    #Collects teams in given shifts html (parsed by Beautiful Soup)
    #Utilizes method from Harry Shomer's hockey_scraper package
    team = soup.find('td', class_='teamHeading + border')  # Team for shifts
    team = team.get_text()

    # Get Home Team
    teams = soup.find_all('td', {'align': 'center', 'style': 'font-size: 10px;font-weight:bold'})
    regex = re.compile(r'>(.*)<br/?>')
    home_team = regex.findall(str(teams[7]))

    return [team, home_team[0]]

#PARSE FUNCTIONS
def analyze_shifts(shift, name, team, home_team, player_ids):
    #Collects teams in given shifts html (parsed by Beautiful Soup)
    #Modified version of Harry Shomer's analyze_shifts function in the hockey_scraper package
    shifts = dict()

    shifts['player_name'] = name.upper()
    shifts['period'] = '4' if shift[1] == 'OT' else '5' if shift[1] == 'SO' else shift[1]
    shifts['team_abbr'] = shared.get_team(team.strip(' '))
    shifts['start'] = shared.convert_to_seconds(shift[2].split('/')[0])
    shifts['duration'] = shared.convert_to_seconds(shift[4].split('/')[0])

    # I've had problems with this one...if there are no digits the time is fucked up
    if re.compile(r'\d+').findall(shift[3].split('/')[0]):
        shifts['end'] = shared.convert_to_seconds(shift[3].split('/')[0])
    else:
        shifts['end'] = shifts['start'] + shifts['duration']

    try:
        if home_team == team:
            shifts['player_id'] = player_ids['home'][name.upper()]['id']
        else:
            shifts['player_id'] = player_ids['away'][name.upper()]['id']
    except KeyError:
        shifts['player_id'] = None

    return shifts

def parse_shifts(html, player_ids, game_id):
    #Two-stage parsing of shifts data for a single team in a provided game
    #Stage one: create dataframe with raw individual shifts
    #Stage two: convert shift events to play-by-play structure created with json_parsing
    #Modified version of Harry Shomer's parse_shifts function in the hockey_scraper package


    all_shifts = []
    columns = ['game_id', 'player_name', 'player_id', 'period', 'team_abbr', 'start', 'end', 'duration']

    td, teams = get_soup(html)

    team = teams[0]
    home_team = teams[1]
    players = dict()

    # Iterates through each player shifts table with the following data:
    # Shift #, Period, Start, End, and Duration.
    for t in td:
        t = t.get_text()
        if ',' in t:     # If a comma exists it is a player
            name = t
            name = name.split(',')
            name = ' '.join([name[1].strip(' '), name[0][2:].strip(' ')])
            #name = shared.fix_name(name)
            #This has been excluded as means to control the differences in names between the JSON and HTML documents
            players[name] = dict()
            players[name]['number'] = name[0][:2].strip()
            players[name]['shifts'] = []
        else:
            players[name]['shifts'].extend([t])

    for key in players.keys():
        # Create lists of shifts-table columns for analysis
        players[key]['shifts'] = [players[key]['shifts'][i:i + 5] for i in range(0, len(players[key]['shifts']), 5)]

        # Parsing
        shifts = [analyze_shifts(shift, key, team, home_team, player_ids) for shift in players[key]['shifts']]
        all_shifts.extend(shifts)

    df = pd.DataFrame(all_shifts)
    df['game_id'] = str(game_id)

    shifts_raw = df[columns]

    shifts_raw = shifts_raw[shifts_raw['duration'] > 0]

    # Second-stage beginds here
    # Identify shift starts for each shift event
    shifts_on = shifts_raw.groupby(['team_abbr', 'period', 'start']).agg(
        num_on=('player_name', 'size'),
        players_on=('player_name', lambda x: ', '.join(x)),
        ids_on=('player_id', lambda x: ', '.join(map(str, x)))
    ).reset_index()

    shifts_on = shifts_on.rename(columns={
        'start':"seconds_elapsed"
    })

    # Identify shift stops for each shift event
    shifts_off = shifts_raw.groupby(['team_abbr', 'period', 'end']).agg(
        num_off=('player_name', 'size'),
        players_off=('player_name', lambda x: ', '.join(x)),
        ids_off=('player_id', lambda x: ', '.join(map(str, x)))
    ).reset_index()

    shifts_off = shifts_off.rename(columns={
        'end':"seconds_elapsed"
    })

    # Merge and sort by time in game
    shifts = pd.merge(shifts_on, shifts_off, on=['team_abbr', 'period', 'seconds_elapsed'], how='outer')
    
    shifts = shifts.sort_values('seconds_elapsed')

    #Modify columns of new total shifts dataframe
    shifts['period'] = shifts['period'].astype(int)
    shifts['event_type'] = 'change'
    shifts['seconds_elapsed'] = shifts['seconds_elapsed'] + (1200 * (shifts['period']-1))
    shifts['game_seconds_remaining'] = 3600 - shifts['seconds_elapsed']

    # Handle missing values at the start and end of periods
    shifts['players_on'] = shifts['players_on'].fillna('None')
    shifts['players_off'] = shifts['players_off'].fillna('None')
    shifts['ids_on'] = shifts['ids_on'].fillna('0')
    shifts['ids_off'] = shifts['ids_off'].fillna('0')
    shifts['num_on'] = shifts['num_on'].fillna(0).astype(int)
    shifts['num_off'] = shifts['num_off'].fillna(0).astype(int)

    #Manual Team Rename
    shifts['team_abbr'] = shifts['team_abbr'].replace({
        "L.A":"LAK",
        "N.J":"NJD",
        "S.J":"SJS",
        "T.B":"TBL"
    })

    #Return: shift events formatted similarly to json pbp: shootout changes are discluded
    return shifts.loc[shifts['period']<5].rename(columns={'team_abbr':'event_team_abbr'})

def construct_skaters_matrix(rosters, shifts, team_abbr, home=True):
    #Given roster info (from the retreive_players function), shifts df, and team, generate on_ice columns for shift events
    #These on-ice columns configure the on-ice players for events in the json play by play as well
    skaters = pd.DataFrame()
    goalies = pd.DataFrame()
    if home:
        team = {key:value for key, value in rosters['home'].items() if value['pos'] != "G"}
    else:
        team = {key:value for key, value in rosters['away'].items() if value['pos'] != "G"}

    names = list(team.keys())
    try: names.remove("")
    except ValueError: ""

    for player in names:
        #For each player in the game, determine when they began and ended shifts.  
        #With player names as columns, 1 represents a shift event a player was on the ice for while 0 represents off the ice
        on_ice = (np.cumsum(
            shifts.loc[(shifts['event_team_abbr'] == team_abbr), 'players_on']
            .apply(str)
            .apply(lambda x: int(bool(re.search(player, x)))) -
            shifts.loc[(shifts['event_team_abbr'] == team_abbr), 'players_off']
            .apply(str)
            .apply(lambda x: int(bool(re.search(player, x))))
        ))
        skaters[player] = on_ice
    
    skaters = skaters.fillna(0).astype(int)


    on_skaters = (skaters == 1).stack().reset_index()
    on_skaters = on_skaters[on_skaters[0]].groupby("level_0")["level_1"].apply(list).reset_index()
    
    max_players = 6
    for i in range(max_players):
        on_skaters[f"{'home' if home else 'away'}_on_{i+1}"] = on_skaters["level_1"].apply(lambda x: x[i] if i < len(x) else " ")
    
    on_skaters = on_skaters.drop(columns=["level_1"]).rename(columns={"level_0": "row"})
    
    #Repeat above process with goaltenders
    if home:
        team = {key:value for key, value in rosters['home'].items() if value['pos'] == "G"}
    else:
        team = {key:value for key, value in rosters['away'].items() if value['pos'] == "G"}
    
    names = list(team.keys())
    try: names.remove("")
    except ValueError: ""

    for player in names:
        on_ice = (np.cumsum(
            shifts.loc[(shifts['event_team_abbr'] == team_abbr), 'players_on']
            .apply(str)
            .apply(lambda x: int(bool(re.search(player, x)))) -
            shifts.loc[(shifts['event_team_abbr'] == team_abbr), 'players_off']
            .apply(str)
            .apply(lambda x: int(bool(re.search(player, x))))
        ))
        goalies[player] = on_ice
    
    goalies = goalies.fillna(0).astype(int)
    
    on_goalies = (goalies == 1).stack().reset_index()
    on_goalies = on_goalies[on_goalies[0]].groupby("level_0")["level_1"].apply(list).reset_index()
    
    max_players = 1
    for i in range(max_players):
        on_goalies[f"{'home' if home else 'away'}_goalie"] = on_goalies["level_1"].apply(lambda x: x[i] if i < len(x) else " ")
    
    on_goalies = on_goalies.drop(columns=["level_1"]).rename(columns={"level_0": "row"})
    
    #combine on-ice skaters and goaltenders for each shift event
    on_players = pd.merge(on_skaters,on_goalies,how='outer',on=['row'])

    shifts['row'] = shifts.index
    
    #Return: shift events with newly added on-ice columns.  NAN values are replaced with string "REMOVE" as means to create proper on-ice columns for json pbp
    return pd.merge(shifts,on_players,how="outer",on=['row']).replace(np.nan,"REMOVE")

def combine_shifts(home_shift,away_shift,json,game_id):
    #Given shifts html documents for home and away team, return shift events complete with both teams' changes in the provided game
    data = retreive_players(json,result="pos")
    data_id = retreive_players(json)

    away = parse_shifts(away_shift,data_id,game_id).sort_values(by=['period','seconds_elapsed'])
    home = parse_shifts(home_shift,data_id,game_id).sort_values(by=['period','seconds_elapsed'])

    away['row'] = away.index
    home['row'] = home.index
    
    away_shifts = construct_skaters_matrix(data,away,pd.json_normalize(json)['awayTeam.abbrev'][0],False).fillna("REMOVE")
    home_shifts = construct_skaters_matrix(data,home,pd.json_normalize(json)['homeTeam.abbrev'][0],True).fillna("REMOVE")

    shifts = pd.concat([away_shifts,home_shifts]).sort_values(by=['period','seconds_elapsed'])
    
    #Return: shifts dataframe with both teams' changes
    return shifts.drop(columns=['row'])

def fix_names(shifts_df,json):
    #Uses alternative names provided in the json to search shifts and ensure both shifts and json dataframes use the same name for each player
    data = pd.json_normalize(json['rosterSpots'])
    data['fullName'] = (data['firstName.default']+" "+data['lastName.default']).str.upper()

    alt_name_col = ['firstName.cs',	'firstName.de',	'firstName.es',	'firstName.fi',	'firstName.sk',	'firstName.sv']
    for i in range(len(alt_name_col)):
        try: data['fullName.'+str(i+1)] = np.where(data[alt_name_col[i]].notna(),(data[alt_name_col[i]].astype(str)+" "+data['lastName.default'].astype(str)).str.upper(),np.nan)
        except: continue

    name_col = ['fullName',	'fullName.1',	'fullName.2',	'fullName.3',	'fullName.4',	'fullName.5', 'fullName.6']

    for name in name_col:
        try: data[name]
        except:
            data[name] = np.nan

    names_dfs = []
    for name in name_col[1:len(name_col)]:
        names_dfs.append(data[[name,'fullName']].rename(columns={name:"alt",
                                            "fullName":'default'}))

    names_df = pd.concat(names_dfs)

    replace = {}
    for default, alt in zip(names_df['default'],names_df['alt']):
        if alt == np.nan or alt == "" or str(alt) == 'nan':
            continue
        else:
            replace.update({alt:default})
    
    return shifts_df.replace(replace,regex=True)

def combine_data(json,html):
    #Given json pbp and html shifts, total game play-by-play data is provided with additional and corrected details
    df = pd.concat([json,html])

    #Fill period_type column and assign shifts a sub-500 event code
    df['period_type'] = np.where(df['period']<4,"REG",np.where(df['period']==4,"OT","SO"))
    df['event_type_code'] = np.where(df['event_type']!='change',df['event_type_code'],499)

    #Create priority columns designed to order events that occur at the same time in a game
    start_pri = ['period-start','game-start']
    even_pri = ['takeaway','giveaway','missed-shot','hit','shot-on-goal','blocked-shot']
    df['priority'] = np.where(df['event_type'].isin(start_pri),0,
                              np.where(df['event_type'].isin(even_pri),1,
                              np.where(df['event_type']=='goal',2,
                              np.where(df['event_type']=='stoppage',3,
                              np.where(df['event_type']=='penalty',4,
                              np.where(df['event_type']=='change',5,
                              np.where(df['event_type']=='period-end',6,
                              np.where(df['event_type']=='game-end',7,
                              np.where(df['event_type']=='faceoff',8,9)))))))))
    
    df = df.sort_values(by=['period','seconds_elapsed','priority']).reset_index()
    #Recreate event_num column to accurately depict the order of all events, including changes
    df['event_num'] = df.index+1
    df['event_team_status'] = np.where(df['event_team_abbr'].isna(),"",np.where(df['home_team_abbr']==df['event_team_abbr'],"home","away"))
    df['event_type_last'] = df['event_type'].shift(1)
    df['event_type_last_2'] = df['event_type_last'].shift(1)
    df['event_type_next'] = df['event_type'].shift(-1)
    lag_events = ['stoppage','goal','period-end']
    lead_events = ['faceoff','period-end']
    period_end_secs = [0,1200,2400,3600,4800,6000,7200,8400,9600,10800]
    #Define shifts by "line-change" or "on-the-fly"
    df['shift_type'] = np.where(df['event_type']=='change',np.where(np.logical_or(np.logical_or(df['event_type_last'].isin(lag_events),df['event_type_last_2'].isin(lag_events),df['event_type_next'].isin(lead_events)),df['seconds_elapsed'].isin(period_end_secs)),"line-change","on-the-fly"),"")

    #Descrpitions:
    #HTML pbp includes descriptions for each event; without the HTML pbp, play descriptions must be generated
    #Different, more originally formatting is employed with these descriptions in comparison to that provided in the HTML pbp
    df['start_end_desc'] = np.where(df['event_type'].isin(['period-start','period-end']),df['away_team_abbr'] + "vs" + df['home_team_abbr'] + ": Period " + df['period'].astype(str) + " " + df['event_type'].str.replace("period-","",regex=True).str.capitalize(),np.nan)
    df['take_give_desc'] = np.where(df['event_type'].isin(['takeaway','giveaway']),df['event_team_abbr'] + " " + df['event_type'].str.upper() + " by " + df['event_player_1_name'],np.nan)
    df['stoppage_desc'] = np.where(df['event_type']=='stoppage',"STOPPAGE: " + df['reason'].str.replace("-"," ",regex=True).str.capitalize(),np.nan)
    df['blocked_desc'] = np.where(df['event_type']=='blocked-shot',df['event_team_abbr'] + " SHOT from " + df['event_player_1_name'] + " BLOCKED by " + df['event_player_2_name'],np.nan)
    df['missed_desc'] = np.where(df['event_type']=='missed-shot',df['event_team_abbr'] + " SHOT by " + df['event_player_1_name'] + " MISSED: " + df['reason'].astype(str).str.replace("-"," ",regex=True),np.nan)
    df['sog_desc'] = np.where(df['event_type']=='shot-on-goal',df['event_team_abbr'] + " SHOT by " + df['event_player_1_name'] + " SAVED by " + df['event_goalie_name'],np.nan)
    df['goal_desc'] = np.where(df['event_type']=='goal',df['event_team_abbr'] + " GOAL SCORED by " + df['event_player_1_name'],np.nan)
    df['assist_desc'] = np.where(np.logical_and(df['event_type']=='goal',df['event_player_2_name'].notna())," ASSISTED by " + df['event_player_2_name'],"")
    df['assist2_desc'] = np.where(np.logical_and(df['event_type']=='goal',df['event_player_3_name'].notna())," and ASSISTED by " + df['event_player_3_name'],"")
    df['goal_desc_complete'] = df['goal_desc'] + df['assist_desc'] + df['assist2_desc']
    df['hit_desc'] = np.where(df['event_type']=='hit',df['event_team_abbr'] + " HIT by " + df['event_player_1_name'] + " on " + df['event_player_2_name'],np.nan)
    df['faceoff_desc'] = np.where(df['event_type']=='faceoff',"FACEOFF WON by " + df['event_player_1_name'] + " AGAINST " + df['event_player_2_name'],np.nan)
    df['penalty_desc'] = np.where(df['event_type']=='penalty',df['event_team_abbr'] + " PENALTY on " + df['event_player_1_name'] + ": " + df['penalty_duration'].astype(str).str.replace(".0","",regex=True) + " minutes for " + df['penalty_description'].astype(str).str.replace("-"," ",regex=True).str.upper(),np.nan)
    
    df['description'] = df['start_end_desc'].combine_first(df['take_give_desc'])\
                                            .combine_first(df['stoppage_desc'])\
                                            .combine_first(df['blocked_desc'])\
                                            .combine_first(df['missed_desc'])\
                                            .combine_first(df['sog_desc'])\
                                            .combine_first(df['goal_desc_complete'])\
                                            .combine_first(df['hit_desc'])\
                                            .combine_first(df['faceoff_desc'])\
                                            .combine_first(df['penalty_desc'])
    ffill_col = ['season','season_type','game_id','game_date',
                 "start_time","venue","venue_location",
        'away_team_abbr','home_team_abbr','home_team_defending_side',
        'away_score','away_fenwick',
        'home_score','home_fenwick',
        'away_goalie','home_goalie']
    away_on = ['away_on_1','away_on_2','away_on_3','away_on_4','away_on_5','away_on_6']
    home_on = ['home_on_1','home_on_2','home_on_3','home_on_4','home_on_5','home_on_6']

    #Forward fill appropriate columns
    for col in ffill_col+away_on+home_on:
        df[col] = df[col].ffill()

    #Now that forward fill is complete, replace "REMOVE" with nan
    df.replace("REMOVE",np.nan,inplace=True)
    
    #Reconfigure strength state and sitution codes
    df['away_skaters'] = df[away_on].replace(r'^\s*$', np.nan, regex=True).notna().sum(axis=1)
    df['home_skaters'] = df[home_on].replace(r'^\s*$', np.nan, regex=True).notna().sum(axis=1)
    df['away_goalie_in'] = np.where(df['away_goalie'].replace(r'^\s*$', np.nan, regex=True).notna(),1,0)
    df['home_goalie_in'] = np.where(df['home_goalie'].replace(r'^\s*$', np.nan, regex=True).notna(),1,0)

    df['event_skaters'] = np.where(df['event_team_abbr']==df['home_team_abbr'],df['home_skaters'],df['away_skaters'])
    df['event_skaters_against'] = np.where(df['event_team_abbr']==df['home_team_abbr'],df['away_skaters'],df['home_skaters'])

    df['strength_state'] = df['event_skaters'].astype(str) + "v" + df['event_skaters_against'].astype(str)
    df['situation_code'] = np.where(df['situation_code'].isna(),df['away_goalie_in'].astype(str) + df['away_skaters'].astype(str) + df['home_skaters'].astype(str) + df['home_goalie_in'].astype(str),df['situation_code'])
    
    col = [
        'season','season_type','game_id','game_date',"start_time","venue","venue_location",
        'away_team_abbr','home_team_abbr','event_num','period','period_type',
        'seconds_elapsed', "situation_code","strength_state","home_team_defending_side","shift_type",
        "event_type_code","event_type","description","reason","penalty_duration","penalty_description",
        "event_team_abbr",'num_on', 'players_on', 'ids_on', 'num_off', 'players_off', 'ids_off',
        "event_team_status","event_player_1_id","event_player_2_id","event_player_3_id",
        "event_player_1_name","event_player_2_name","event_player_3_name","event_player_1_pos","event_player_2_pos",
        "event_player_3_pos","event_goalie_id",
        "event_goalie_name","shot_type","zone_code","x","y","x_fixed","y_fixed","x_adj","y_adj",
        "event_skaters","away_skaters","home_skaters",
        "event_distance","event_angle","away_score","home_score", "away_fenwick", "home_fenwick",
        "away_on_1","away_on_2","away_on_3","away_on_4","away_on_5","away_on_6","away_goalie",
        "home_on_1","home_on_2","home_on_3","home_on_4","home_on_5","home_on_6","home_goalie"
    ]

    #Return: complete play-by-play with all important data for each event in a provided game
    return df[col].replace(r'^\s*$', np.nan, regex=True)