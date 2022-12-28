import asyncio
import json
import aiohttp
from understat import Understat
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import *
import nest_asyncio
nest_asyncio.apply()


def get_match_result(team, season):
    async def main():
        async with aiohttp.ClientSession() as session:
            understat = Understat(session)
            team_stats = await understat.get_team_results(
                team,
                season
            )
            # return json.dumps(team_stats)
            return team_stats


    loop = asyncio.get_event_loop()
    data = loop.run_until_complete(main())
    return data

def get_playershot_data(match_id):
    async def main():
        async with aiohttp.ClientSession() as session:
            understat = Understat(session)
            players = await understat.get_match_shots(match_id)
            return players
    loop = asyncio.get_event_loop()
    players = loop.run_until_complete(main())
    return players

def get_player_data(match_id):
    async def main():
        async with aiohttp.ClientSession() as session:
            understat = Understat(session)
            players = await understat.get_match_players(match_id)
            return players

    loop = asyncio.get_event_loop()
    players = loop.run_until_complete(main())
    return players

def get_goal_rank(league, season, find_team, team = None):
    async def main():
        if find_team:

            async with aiohttp.ClientSession() as session:
                understat = Understat(session)
                players = await understat.get_league_players(
                    league,
                    season,
                    team_title = team
                )

               
        else:
            async with aiohttp.ClientSession() as session:
                understat = Understat(session)
                players = await understat.get_league_players(
                    league,
                    season
                )

        return players

    loop = asyncio.get_event_loop()
    rank = loop.run_until_complete(main())
    return rank


def get_current_rank_data(league, season, start_date, end_date):
    async def main():
        async with aiohttp.ClientSession() as session:
            understat = Understat(session)
            table = await understat.get_league_table(league, season, start_date= start_date, end_date=end_date)
            return table

    loop = asyncio.get_event_loop()
    table = loop.run_until_complete(main())
    return table



    
# def Constrcut_df_from_results(li):
#     '''
#     Parameter: li - List

#     Return DataFrame

#     Id - Game oid
#     oppoenet - team name
#     opponent_id - id of team name
#     home - binary for whether in home(0:home, 1:away)
#     result - win/draw/loss
#     datatime - Game time
#     xG - xG of chelsea
#     xG_opponent - Xg of oppoenent
#     '''

#     col_name = ['id', 'oppenet', 'opponent_id', 'home', 'result', 'datetime', 'xG', 'num_of_score_goals', 'num_of_conced_goals']
#     id, oppenet, opponent_id, home, result, datetime, xG,goal_score, goal_conced = [], [], [], [], [], [], [], [], []
#     for game in li:
#         id.append(game['id'])
#         opo_side = 'a' if game['side'] == 'h' else 'h'
#         oppenet.append(game[opo_side]['title'])
#         opponent_id.append(game[opo_side]['id'])
#         home.append(0 if game['side'] == 'h' else 1)
#         result.append(game['result'])
#         xG.append(game['xG'][game['side']])
#         datetime.append(game['datetime'])
#         goal_score.append(game['goals'][game['side']])
#         goal_conced.append(game['goals'][opo_side])
#     data_list = [id, oppenet, opponent_id, home, result, datetime, xG,goal_score, goal_conced]
#     df = pd.DataFrame({col_name: col for col_name, col in zip(col_name, data_list)})
#     return df


# def Constrcut_df_from_results(li):

#     col_name = ['id', 'datetime', 'result', 'team', 'team_xG', 'team_score', 'team_conced','home', 'opponent', 'oppo_xG']
#     id, datetime, result, team, team_xG, goal_score, goal_conced, home, opponent, oppo_xG = [], [], [], [], [], [], [], [], [], []
#     for game in li:
#         id.append(game['id'])
#         opo_side = 'a' if game['side'] == 'h' else 'h'
#         opponent.append(game[opo_side]['title'])
#         team.append(game[game['side']]['title'])
#         home.append(0 if game['side'] == 'h' else 1)
#         result.append(game['result'])
#         team_xG.append(game['xG'][game['side']])
#         datetime.append(game['datetime'])
#         goal_score.append(game['goals'][game['side']])
#         goal_conced.append(game['goals'][opo_side])
#         oppo_xG.append(game['xG'][opo_side])
#     data_list = [id, datetime, result, team, team_xG, goal_score, goal_conced, home, opponent, oppo_xG]
#     df = pd.DataFrame({col_name: col for col_name, col in zip(col_name, data_list)})
#     df['oppo_score'] = df['team_conced']
#     df['oppo_conced'] = df['team_score']
#     df['oppo_home'] = df['home'].apply(lambda x: 1 if x == 0 else 0)
#     return df


# def num_of_shots(id, home):
#     '''
#     Parmeter:
#     id - Game id
#     home - whether the game is home or away
#     players - player shot data 

#     Return the number of shots and shots of opponent

#     '''
#     status = 'h' if home == 0 else 'a'
#     players = get_playershot_data(id)
#     return len(players[status])

# Utils function when calculate the players stats of a match
def get_stats_dict(league, season, type, n,find_team, team = None):
    data = get_goal_rank(league, season, find_team, team)
    data_dict = {x['player_name']: x[type] for x in data}
    sorted_data = sorted(data_dict.keys(), key= lambda x: int(data_dict[x]), reverse=True)[:n]
    return sorted_data
    
# Get the stats of players in a particular team for a match
def get_player_stats(id_data, season):
    
    col_names = ['id', 'datetime', 'home', 'name', 'goal', 'assit', 'shot', 'rank_goal', 'rank_team_goal', 'rank_assist', 'rank_team_assist']
    res_df = pd.DataFrame(columns=col_names)
    for i in range(id_data.shape[0]):
        print(id_data.loc[i, 'id'])
        status = 'h' if id_data.loc[i, 'home'] == 0 else 'a'
        player_data = get_player_data(id_data.loc[i, 'id'])[status]
        match_data = get_playershot_data(id_data.loc[i, 'id'])[status]
        team = list(map(lambda x: player_data[x]['player'], player_data.keys()))
        df = pd.DataFrame(np.zeros((len(team),len(col_names))), columns=col_names)
        df['id'] = id_data.loc[i, 'id']
        df['status'] = status
        df['name'] = team

        for shot in match_data:
            df.loc[df['name'] == shot['player'], 'shot'] += 1
            if shot['result'] == 'Goal':
                df.loc[df['name'] == shot['player'], 'goal'] += 1
                df.loc[df['name'] == shot['player_assisted'],'assit'] += 1
            else:
                pass
        datetime  =  id_data.loc[i, 'datetime']
        df['datetime'] = datetime
        # start to fill the player columns
        # r
        league_goal_rank = get_stats_dict('epl', season, 'goals', 15, False)
        league_assit_rank = get_stats_dict('epl', season, 'assists', 15, False)
        team_goal_rank = get_stats_dict('epl', season, 'goals', 5, True, team)
        team_assit_rank = get_stats_dict('epl', season, 'goals', 5, True, team)
        
        def player_upgrade(x, list):
            return x+1 if x in list else x

        for col, li in zip(['rank_goal', 'rank_team_goal', 'rank_assist', 'rank_team_assist'], [league_goal_rank, league_assit_rank, 
           team_assit_rank, team_goal_rank]):
            df[col] = df[col].apply(lambda x: player_upgrade(x, li))

        res_df = pd.concat([res_df, df], axis = 0)

    return res_df



    
    



    
