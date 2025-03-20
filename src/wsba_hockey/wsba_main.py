import requests as rs
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from tools.scraping import *

### WSBA HOCKEY ###
## Provided below are all integral functions in the WSBA Hockey Python package. ##


### ALL FEATURES ###
# Note: Features yet included are marked with *

## SCRAPING ##
# NHL Play-by-Play (Of any game frame up to a full season) #
# NHL Season Information #
# NHL Rosters and Player Information #

## DATA ANALYTICS ##
# Expected Goals (WeakSide Breakout and MoneyPuck models)* #
# Goal Impacts and Shot Analysis* #
# Stat Aggregation* #
# Shot Plotting (Plots, Heatmaps, etc.)* #

## REPOSITORY ##
# Past Season Play-by-Play* #
# Team Information #
# Schedule #

### ACKNOWLEDGEMENTS AND CREDITS ###
## Huge thanks to the following: ##
# Harry Shomer - Creator of the hockey_scraper package, which contains select utils functions utilized in this package and otherwise inspires the creation of this package. #
# Dan Morse - Creator of the hockeyR package; another important inspiration and model for developing an NHL scraper. #

## SCRAPE FUNCTIONS ##
def nhl_scrape_game(game_ids,split_shifts = False,remove = ['period-start','period-end','challenge','stoppage']):
    #Given a set of game_ids (NHL API), return complete play-by-play information as requested
    # param 'game_ids' - NHL game ids
    # param 'split_shifts' - boolean which splits pbp and shift events if true
    # param 'remove' - list of events to remove from final dataframe

    pbps = []
    for game_id in game_ids:
        print("Scraping data from game " + str(game_id) + "...")

        game_id = str(game_id)
        season = str(game_id[:4])+str(int(game_id[:4])+1)

        api = "https://api-web.nhle.com/v1/gamecenter/"+game_id+"/play-by-play"
        home_log = "https://www.nhl.com/scores/htmlreports/"+season+"/TH"+str(game_id)[-6:]+".HTM"
        away_log = "https://www.nhl.com/scores/htmlreports/"+season+"/TV"+str(game_id)[-6:]+".HTM"

        #Retrieve raw data
        json = rs.get(api).json()
        home_shift = rs.get(home_log).content
        away_shift = rs.get(away_log).content

        if int(game_id[:4]) < 2010:
            print()
            raise Exception('Games before 2010-2011 are not available yet.')
        else:
            #Parse Json
            pbp = parse_json(json) 
        
        #Create shifts
        shifts = fix_names(combine_shifts(home_shift,away_shift,json,game_id),json)

        #Combine and append data to list
        data = combine_data(pbp,shifts)

        pbps.append(data)

    #Add all pbps together
    df = pd.concat(pbps)

    #Split pbp and shift events if necessary
    #Return: complete play-by-play with data removed or split as necessary
    if split_shifts == True:
        if len(remove) == 0:
            remove = ['change']
        
        #Return: dict with pbp and shifts seperated
        return {"pbp":df.loc[~df['event_type'].isin(remove)].dropna(axis=1,how='all'),
            "shifts":df.loc[df['event_type']=='change'].dropna(axis=1,how='all')
            }
    else:
        #Return: all events that are not set for removal by the provided list
        return df.loc[~df['event_type'].isin(remove)]

def nhl_scrape_schedule(season,start = "09-01", end = "08-01"):
    #Given a season, return schedule data
    # param 'season' - NHL season to scrape
    # param 'start' - Start date in season
    # param 'end' - End date in season

    api = "https://api-web.nhle.com/v1/schedule/"

    #Determine how to approach scraping; if month in season is after the new year the year must be adjusted
    new_year = ["01","02","03","04","05","06"]
    if start[:2] in new_year:
        start = str(int(season[:4])+1)+"-"+start
        end = str(season[:-4])+"-"+end
    else:
        start = str(season[:4])+"-"+start
        end = str(season[:-4])+"-"+end

    form = '%Y-%m-%d'

    #Create datetime values from dates
    start = datetime.strptime(start,form)
    end = datetime.strptime(end,form)

    game = []

    day = (end-start).days+1
    if day < 0:
        #Handles dates which are over a year apart
        day = 365 + day
    for i in range(day):
        #For each day, call NHL api and retreive id, season, season_type (1,2,3), and gamecenter link
        inc = start+timedelta(days=i)
        print("Scraping games on " + str(inc)[:10]+"...")
        
        get = rs.get(api+str(inc)[:10]).json()
        gameWeek = list(pd.json_normalize(get['gameWeek'])['games'])[0]

        for i in range(0,len(gameWeek)):
            game.append(pd.DataFrame({
                "id": [gameWeek[i]['id']],
                "season": [gameWeek[i]['season']],
                "season_type":[gameWeek[i]['gameType']],
                "gamecenter_link":[gameWeek[i]['gameCenterLink']]
                }))
    
    #Concatenate all games
    df = pd.concat(game)
    
    #Return: specificed schedule data (excluding preseason games)
    return df.loc[df['season_type']>1]

def nhl_scrape_season(season,split_shifts = False, remove = ['period-start','period-end','challenge','stoppage','change'], start = "09-01", end = "08-01", local=False, local_path = "schedule/schedule.csv"):
    #Given season, scrape all play-by-play occuring within the season
    # param 'season' - NHL season to scrape
    # param 'split_shifts' - boolean which splits pbp and shift events if true
    # param 'remove' - list of events to remove from final dataframe
    # param 'start' - Start date in season
    # param 'end' - End date in season
    # param 'local' - boolean indicating whether to use local file to scrape game_ids
    # param 'local_path' - path of local file

    #While the default value of local is false, schedule data is provided in the package files; enabling local will automatically find and scrape games in a specified season, saving time otherwise spent scraping a season's schedule
    if local == True:
        load = pd.read_csv(local_path)
        load = load.loc[load['season'].astype(str)==season]
        game_ids = list(load['id'].astype(str))
    else:
        game_ids = list(nhl_scrape_schedule(season,start,end)['id'].astype(str))

    df = []
    df_s = []

    errors = []
    for game_id in game_ids: 
        try:
            if split_shifts == True:
                data = nhl_scrape_game([game_id],split_shifts=True,remove=remove)
                df.append(data['pbp'])
                df_s.append(data['shifts'])
            else:
                data = nhl_scrape_game([game_id],remove=remove)
                df.append(data)

        except: 
            #Errors should be rare; testing of eight full-season scraped produced just one missing game due to erro
            #Games which have not happened yet also print as errors
            print("An error occurred...")
            errors.append(pd.DataFrame({"id":game_id}))
    
    pbp = pd.concat(df)
    if split_shifts == True:
        shifts = pd.concat(df_s)
    else:
        ""
    try: 
        errors = pd.concat(errors)
    except:
        errors = pd.DataFrame()

    #Return: Complete pbp and shifts data for specified season as well as dataframe of game_ids which failed to return data
    if split_shifts == True:
        return {"pbp":pbp,
            'shifts':shifts,
            "errors":errors}
    else:
        return {"pbp":pbp,
            "errors":errors}

def nhl_scrape_seasons_info(seasons = []):
    #Returns info related to NHL seasons (by default, all seasons are included)
    # param 'season' - list of seasons to include

    print("Scraping info for seasons: " + str(seasons))
    api = "https://api.nhle.com/stats/rest/en/season"
    info = "https://api-web.nhle.com/v1/standings-season"
    data = rs.get(api).json()['data']
    data_2 = rs.get(info).json()['seasons']

    df = pd.json_normalize(data)
    df_2 = pd.json_normalize(data_2)

    df = pd.merge(df,df_2,how='outer',on=['id'])
    
    if len(seasons) > 0:
        return df.loc[df['id'].astype(str).isin(seasons)].sort_values(by=['id'])
    else:
        return df.sort_values(by=['id'])

def nhl_scrape_standings(arg = "now"):
    #Returns standings
    # parma 'arg' - by default, this is "now" returning active NHL standings.  May also be a specific date formatted as YYYY-MM-DD
    
    if arg == "now":
        print("Scraping standings as of now...")
    else:
        print("Scraping standings for season: "+arg)
    api = "https://api-web.nhle.com/v1/standings/"+arg
    
    data = rs.get(api).json()['standings']

    return pd.json_normalize(data)

def nhl_scrape_roster(season):
    #Given a nhl season, return rosters for all participating teams
    # param 'season' - NHL season to scrape
    print("Scrpaing rosters for the "+ season + "season...")
    teaminfo = pd.read_csv("teaminfo/nhl_teaminfo.csv")

    rosts = []
    for team in list(teaminfo['Team']):
        try:
            print("Scraping " + team + " roster...")
            api = "https://api-web.nhle.com/v1/roster/"+team+"/"+season
            
            data = rs.get(api).json()
            forwards = pd.json_normalize(data['forwards'])
            forwards['headingPosition'] = "F"
            dmen = pd.json_normalize(data['defensemen'])
            dmen['headingPosition'] = "D"
            goalies = pd.json_normalize(data['goalies'])
            goalies['headingPosition'] = "G"

            roster = pd.concat([forwards,dmen,goalies]).reset_index(drop=True)
            roster['fullName'] = (roster['firstName.default']+" "+roster['lastName.default']).str.upper()
            roster['season'] = str(season)
            roster['team_abbr'] = team

            rosts.append(roster)
        except:
            print("No roster found for " + team + "...")

    return pd.concat(rosts)

def nhl_scrape_player_info(roster):
    #Given compiled roster information from the nhl_scrape_roster function, return a list of all players (seperated into team and season) and associated information
    # param 'roster' - dataframe of roster information from the nhl_scrape_roster function

    data = roster

    print("Creating player info for provided roster data...")

    alt_name_col = ['firstName.cs',	'firstName.de',	'firstName.es',	'firstName.fi',	'firstName.sk',	'firstName.sv']
    for i in range(len(alt_name_col)):
        try: data['fullName.'+str(i+1)] = np.where(data[alt_name_col[i]].notna(),(data[alt_name_col[i]].astype(str)+" "+data['lastName.default'].astype(str)).str.upper(),np.nan)
        except: continue

    name_col = ['fullName',	'fullName.1',	'fullName.2',	'fullName.3',	'fullName.4',	'fullName.5', 'fullName.6']

    for name in name_col:
        try: data[name]
        except:
            data[name] = np.nan

    infos = []
    for name in name_col:
        infos.append(data[[name,"id","season","team_abbr","headshot",
                              "sweaterNumber","headingPosition",
                              "positionCode",'shootsCatches',
                              'heightInInches','weightInPounds',
                              'birthDate','birthCountry']].rename(columns={
                                                              name:'Player',
                                                              'id':"API",
                                                              "season":"Season",
                                                              "team_abbr":"Team",
                                                              'headshot':'Headshot',
                                                              'sweaterNumber':"Number",
                                                              'headingPosition':"Primary Position",
                                                              'positionCode':'Position',
                                                              'shootsCatches':'Handedness',
                                                              'heightInInches':'Height',
                                                              'weightInPounds':'Weight',
                                                              'birthDate':'Birthday',
                                                              'birthCountry':'Nationality'}))
    players = pd.concat(infos)
    players['Season'] = players['Season'].astype(str)
    players['Player'] = players['Player'].replace(r'^\s*$', np.nan, regex=True)

    return players.loc[players['Player'].notna()].sort_values(by=['Player','Season','Team'])