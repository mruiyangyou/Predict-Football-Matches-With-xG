from utils import *
import numpy as np
import pandas as pd
import datetime
import os
import math

# Define some constant
start_date_EPL = {
    2014: '2014-08-16',
    2015: '2015-08-08',
    2016: '2016-08-13',
    2017: '2017-08-11',
    2018: '2018-08-10',
    2019: '2019-08-09',
    2020: '2020-08-12',
    2021: '2021-08-13',
    2022: '2022-08-05'
}


# Simple data constrution from Lists
def Constrcut_df_from_results(li):

    col_name = ['id', 'datetime', 'result', 'team', 'team_xG', 'team_score', 'team_conced','home', 'opponent', 'oppo_xG']
    id, datetime, result, team, team_xG, goal_score, goal_conced, home, opponent, oppo_xG = [], [], [], [], [], [], [], [], [], []
    for game in li:
        id.append(game['id'])
        opo_side = 'a' if game['side'] == 'h' else 'h'
        opponent.append(game[opo_side]['title'])
        team.append(game[game['side']]['title'])
        home.append(0 if game['side'] == 'h' else 1)
        result.append(game['result'])
        team_xG.append(game['xG'][game['side']])
        datetime.append(game['datetime'])
        goal_score.append(game['goals'][game['side']])
        goal_conced.append(game['goals'][opo_side])
        oppo_xG.append(game['xG'][opo_side])
    data_list = [id, datetime, result, team, team_xG, goal_score, goal_conced, home, opponent, oppo_xG]
    df = pd.DataFrame({col_name: col for col_name, col in zip(col_name, data_list)})
    df['oppo_score'] = df['team_conced']
    df['oppo_conced'] = df['team_score']
    df['oppo_home'] = df['home'].apply(lambda x: 1 if x == 0 else 0)
    return df

# Calculate number of shots
def num_of_shots(id, home):
    '''
    Parmeter:
    id - Game id
    home - whether the game is home or away
    players - player shot data 

    Return the number of shots and shots of opponent

    '''
    status = 'h' if home == 0 else 'a'
    players = get_playershot_data(id)
    return len(players[status])

# Get the rank at a paritulcar time(end date)
def get_current_rank(end_date, team, opponent, start_date, year):
    table = get_current_rank_data('EPL', year, start_date= start_date, end_date=end_date)
    data = pd.DataFrame(table[1:], columns=table[0])
    rank = data.loc[data['Team'] == team].index.to_list()[0] + 1
    opponent_rank = data.loc[data['Team'] == opponent].index.to_list()[0] + 1
    return rank, opponent_rank

# Minus one day utils function
def minus_one_day(dt, return_str = True):
    date = datetime.datetime.strptime(dt, '%Y-%m-%d') - datetime.timedelta(days = 1)
    return datetime.datetime.strftime(date,"%Y-%m-%d") if return_str else date

# Calculate the win ratio against a particular team
def get_win_ratio(date, team, opponent, fixture_data, rate_against_new):
    
    fixture_data = fixture_data.loc[fixture_data['Date'] < minus_one_day(str(date).split(' ')[0], return_str=False)]
    select_con = ((fixture_data['HomeTeam'] == team) & (fixture_data['AwayTeam'] == opponent)) |  ((fixture_data['HomeTeam'] == opponent) & (fixture_data['AwayTeam'] == team))
    select_df = fixture_data.loc[select_con]
    
    if select_df.shape[0] >= 1:
        select_df['win'] = select_df.apply(lambda x: 'w' if (x['HomeTeam'] == team and x['Result'] == 'H') or (x['AwayTeam'] == team and x['Result'] == 'A') else 'd' if x['Result'] == 'D' else 'l', axis = 1)
        win = select_df['win'].value_counts().to_frame().loc['w'].values[0] if 'w' in select_df['win'].value_counts().to_frame().index else 0
        draw = select_df['win'].value_counts().to_frame().loc['d'].values[0] if 'd' in select_df['win'].value_counts().to_frame().index else 0
        return win/select_df.shape[0], draw/select_df.shape[0]
    else:
        # no previous give avrage win rate against new team
        return rate_against_new[0], rate_against_new[1]
  

# Get team recent performance
def get_team_recent(interval, prefix, df):

    df[f'{prefix}_recent_goals'] = df[f'{prefix}_score'].rolling(5, closed='left').mean()
    df[f'{prefix}_recent_conced'] = df[f'{prefix}_conced'].rolling(5, closed='left').mean()
    df[f'{prefix}_recent_state'] = df['y'].rolling(3, closed = 'left').mean()
    df[f'{prefix}_recent_xG'] = df[f'{prefix}_xG'].rolling(5, closed='left').mean()
    return df

# Data pipeline for a particulat team
def team_data_pipeline(team, start_year, end_year, fixture_data, rank_table, new_team_record):

    df = pd.DataFrame(columns=['a'] * 30)

    for y in range(start_year, end_year + 1):
        
        data = get_match_result(team, y)
        # basic match information
        df_year = Constrcut_df_from_results(data)
        df_year['team_shot_attempt'] = df_year.apply(lambda x: num_of_shots(x['id'], x['home']), axis = 1)
        df_year['oppo_shot_attempt'] = df_year.apply(lambda x: num_of_shots(x['id'], x['oppo_home']), axis = 1)

        # time feature
        df_year['datetime'] = pd.to_datetime(df_year['datetime'])
        df_year['stage'] = df_year.apply(lambda x: 0 if x['datetime'].year == y else 1, axis = 1)

        # add rank
        df_year['team_rank'], df_year['oppo_rank'] =  zip(*df_year.apply(lambda x: get_current_rank(minus_one_day(str(x['datetime']).split(' ')[0]), x['team'], x['opponent'], start_date_EPL[y], str(y)), axis = 1))

        # adjust the rank by last season, if it is a new team we generate a constant to replace it
        a =  int(rank_table.loc[((rank_table.index == f'{y-1}-{y}') & (rank_table['Team'] == team)), 'Rank'].values[0]) if df_year.loc[0, 'team'] in rank_table.loc[(rank_table.index == f'{y-1}-{y}'), 'Team'].to_list() else 16
        b = int(rank_table.loc[((rank_table.index == f'{y-1}-{y}') & (rank_table['Team'] == df_year.loc[0, 'opponent'])), 'Rank'].values[0]) if df_year.loc[0, 'opponent'] in rank_table.loc[(rank_table.index == f'{y-1}-{y}'), 'Team'].to_list() else 16
        df_year.loc[0, 'team_rank'] = a
        df_year.loc[0, 'oppo_rank'] = b
            

        # add history record
        table_record = new_team_record.loc[new_team_record.index == team].values.reshape(2,).tolist()
        df_year['team_win_rate'], df_year['team_draw_rate'] = zip(*df_year.apply(lambda x: get_win_ratio(x['datetime'], x['team'], x['opponent'], fixture_data, table_record), axis = 1))
        df_year['oppnent_win_rate'] = 1 - df_year['team_win_rate']
        # add interval data
        df_year['y'] = df_year['result'].apply(lambda x: 1 if x == 'w' else 0)
        
        df_year = get_team_recent(5, 'team', df_year)
        df_year = get_team_recent(5, 'oppo', df_year)

        if y == start_year:
            df.columns = df_year.columns
 
        df_year.set_index([[f'{y}-{y+1}'] * df_year.shape[0]], inplace= True)
        df = pd.concat([df, df_year])

    return df

# Get fixture data
def get_all_fixture(path, special_date_list):

    fixtures = pd.DataFrame(columns=['Date', 'HomeTeam', 'AwayTeam', 'Result'])
    start = 2013

    for i in range(10, 0, -1):
    
        df = pd.read_csv(os.path.join(path, f'E0-{i}.csv'))
        df = df[['Date', 'HomeTeam', 'AwayTeam', 'FTR']]
        df.dropna(inplace=True)
        df['Date'] = df['Date'].astype('string')
        df.rename(columns={'FTR':'Result'}, inplace = True)
        if i in special_date_list:
            df['Date'] = df['Date'].apply(lambda x: datetime.datetime.strptime(x, '%d/%m/%y').strftime("%Y-%m-%d"))
        else:
            df['Date'] = df['Date'].apply(lambda x: datetime.datetime.strptime(x, '%d/%m/%Y').strftime("%Y-%m-%d"))
        df.set_index([[f'{start}-{start+1}'] * df.shape[0]], inplace=True)
        fixtures = pd.concat([fixtures, df], axis = 0)
        start += 1

    return fixtures

# Get rank table from 2013 till now
def get_all_rank():
    table_2013 = pd.read_excel('/Users/marceloyou/Desktop/Xg-Prediction/data/table/table2013.xlsx', header=None)
    table_2013.columns = ['Rank', 'Team', 'Gs', 'Weel', 'Pts']
    table_2013 = table_2013[['Team','Rank']]
    table_2013.index = ['2013-2014'] * table_2013.shape[0] 
    table = table_2013

    for y in range(2014, 2023):
        data = get_current_rank_data('epl', y, None, None)
        year_table = pd.DataFrame(data[1:], columns = data[0])
        year_table['Rank'] = year_table.index + 1
        year_table = year_table[["Team", "Rank"]]
        year_table.index = [f'{y}-{y+1}'] * year_table.shape[0]
        table = pd.concat([table, year_table], axis = 0)
    return table

# Get the stats against promote team 
def stats_against_promoted_team(fixture):
    unique_team = fixture['HomeTeam'].unique().tolist()
    record = {}
    new= {}
    for i, team in enumerate(unique_team):
        draw,win, game = 0, 0, 0
        for y in range(2014, 2023):
            last_year = fixture.loc[fixture['Season'] == f'{y-1}-{y}', 'HomeTeam'].unique().tolist()
            this_year = fixture.loc[fixture['Season'] == f'{y}-{y+1}', 'HomeTeam'].unique().tolist()
            new_team = [i for i in this_year if i not in last_year]
            if i == 0:
                new[f'{y}-{y+1}'] = new_team
            fixture_year = fixture.loc[(fixture['Season'] == f'{y}-{y+1}')]
            fixture_year = fixture_year.loc[((fixture_year['HomeTeam'] == team) & (fixture_year['AwayTeam'].isin(new_team))) | ((fixture_year['AwayTeam'] == team) & (fixture_year['HomeTeam'].isin(new_team)))]
            win_con = ((fixture_year['HomeTeam'] == team) & (fixture_year['Result'] == 'H')) | ((fixture_year['AwayTeam'] == team) & (fixture_year['Result'] == 'A'))
            draw_con = (fixture_year['Result'] == 'D')
            win += fixture_year.loc[win_con].shape[0]
            game += fixture_year.shape[0]
            draw += fixture_year.loc[draw_con].shape[0]
        print(team, win, game)
        record[team] = [win/game, draw/game]
        

    win_newteam = pd.DataFrame(record)
    win_newteam.index = ['Win_Percentage_Against_new_team', 'Draw_percentage_Against_new_team']
    win_newteam = win_newteam.transpose()
    return win_newteam

# Construct whole data set 
def final_data_pipeline(fixture, table, newteam_df, start_year, end_year):
    data_path = '/Users/marceloyou/Desktop/Xg-Prediction/data/matchdata'
    final_df = pd.DataFrame(columns=['a'] * 30)
    for y in range(start_year, end_year+1):
        year_team = fixture.loc[fixture['Season'] == f'{y}-{y+1}', 'HomeTeam'].unique().tolist()
        year_df = pd.DataFrame(columns=['a'] * 30)
        for i, team in enumerate(year_team):
            team_df = team_data_pipeline(team, y, y, fixture, table, newteam_df)
            if i == 0:
                year_df.columns = team_df.columns
            year_df = pd.concat([year_df, team_df])
            year_df.drop_duplicates(['id'], inplace = True)
        if y == 2014:
            final_df.columns = year_df.columns
        data_dir = os.path.join(data_path, str(y))
        if not os.path.exists(data_dir):
            os.mkdir(data_dir)
        year_df.to_csv(os.path.join(data_dir, f'{y}_match.csv'))
        final_df = pd.concat([final_df, year_df])
    return final_df

# Select team data
def select_team_data(data, team, season):
    return data.loc[(data['Season'] == f'{season}-{season+1}') &((data['opponent'] == team) | (data['team'] == team))]

#  Cumsum of opponent data
def cumsm_stat(df, team):
    df['team_score'] = df.apply(lambda x: x['team_score'] if x['team'] == team else x['oppo_score'], axis = 1)
    df['team_conced']  = df.apply(lambda x: x['team_conced'] if x['team'] == team else x['oppo_conced'], axis = 1)
    df['team_xG'] = df.apply(lambda x: x['team_xG'] if x['team'] == team else x['oppo_xG'], axis = 1)
    df['y'] = df.apply(lambda x: x['y'] if x['team'] == team else 1 if (x['team'] != team) and (x['y'] == 0) else 0, axis = 1)
    df['team'] = team
    df = df[['Season', 'id', 'datetime', 'team', 'team_score', 'team_conced', 'team_xG', 'y']]
    df = get_team_recent(5, 'team', df)
    return df

# change the opponent statistics 
def change_oppo_stats(row, na_data, promote_team_stats, final_df_sorted):
    if math.isnan(row['team_recent_goals']):
        # deal with na
        season = int(row['Season'].split('-')[0]) - 1
        team_recent_goals = (na_data.loc[(na_data.index == f'{season}-{season+1}') & (na_data['Team'] == row['team']), 'G'] / 38).values[0] if row['team'] not in promote_team_stats.loc[promote_team_stats.index == row['Season'], 'Team'].to_list() else 0.91
        team_recent_conced = (na_data.loc[(na_data.index == f'{season}-{season+1}') & (na_data['Team'] == row['team']), 'GA'] / 38).values[0] if row['team'] not in promote_team_stats.loc[promote_team_stats.index == row['Season'], 'Team'].to_list() else 1.53
        team_recent_xG = (na_data.loc[(na_data.index == f'{season}-{season+1}') & (na_data['Team'] == row['team']), 'xG'] / 38).values[0] if row['team'] not in promote_team_stats.loc[promote_team_stats.index == row['Season'], 'Team'].to_list() else 0.998
        oppo_recent_goals = (na_data.loc[(na_data.index == f'{season}-{season+1}') & (na_data['Team'] == row['opponent']), 'G'] / 38).values[0]  if row['opponent'] not in promote_team_stats.loc[promote_team_stats.index == row['Season'], 'Team'].to_list() else 0.91
        oppo_recent_conced = (na_data.loc[(na_data.index == f'{season}-{season+1}') & (na_data['Team'] == row['opponent']), 'GA'] / 38).values[0] if row['opponent'] not in promote_team_stats.loc[promote_team_stats.index == row['Season'], 'Team'].to_list() else 1.53
        oppo_recent_xG = (na_data.loc[(na_data.index == f'{season}-{season+1}') & (na_data['Team'] == row['opponent']), 'xG'] /38).values[0] if row['opponent'] not in promote_team_stats.loc[promote_team_stats.index == row['Season'], 'Team'].to_list() else 0.998
        if math.isnan(row['team_recent_state']):
            team_recent_state = (na_data.loc[(na_data.index == f'{season}-{season+1}') & (na_data['Team'] == row['team']), 'W'] / 38).values[0]  if row['team'] not in promote_team_stats.loc[promote_team_stats.index == row['Season'], 'Team'].to_list() else 0.232
            oppo_recent_state = (na_data.loc[(na_data.index == f'{season}-{season+1}') & (na_data['Team'] == row['opponent']), 'W'] / 38).values[0] if row['opponent'] not in promote_team_stats.loc[promote_team_stats.index == row['Season'], 'Team'].to_list() else 0.232
        else:
            season = int(row['Season'].split('-')[0]) 
            oppo_data = select_team_data(final_df_sorted, row['opponent'], season)
            oppo_recent_df = cumsm_stat(oppo_data, row['opponent'])
            team_recent_state, oppo_recent_state = row['team_recent_state'],  oppo_recent_df.loc[oppo_recent_df['id'] == row['id'], 'team_recent_state'].values[0]
    else:
        team_recent_goals, team_recent_conced, team_recent_xG, team_recent_state = row['team_recent_goals'], row['team_recent_conced'], row['team_recent_xG'], row['team_recent_state']
        season = int(row['Season'].split('-')[0]) 
        oppo_data = select_team_data(final_df_sorted, row['opponent'], season)
        oppo_recent_df = cumsm_stat(oppo_data, row['opponent'])
        oppo_recent_goals,oppo_recent_conced, oppo_recent_state, oppo_recent_xG = oppo_recent_df.loc[oppo_recent_df['id'] == row['id'], ['team_recent_goals', 'team_recent_conced','team_recent_state', 'team_recent_xG']].values.reshape(4,).tolist()
    return team_recent_goals, team_recent_conced, team_recent_state, team_recent_xG, oppo_recent_goals, oppo_recent_conced, oppo_recent_state, oppo_recent_xG

# Change format for Bradley Terry Model
def format_for_bt_model(data):
    data = data[['datetime', 'y', 'team', 'team_xG', 'home', 
       'team_rank', 'team_win_rate', 'opponent', 'oppo_xG', 'oppo_home', 
       'oppo_rank', 'oppo_win_rate']]
    for row in data.index.tolist():
        if data.loc[row, 'home'] == 0:
            pass
        else:
    
            data.loc[row, 'y'] = 0 if data.loc[row,'y'] == 1 else 0
            data.loc[row, 'team'], data.loc[row, 'opponent'] = data.loc[row, 'opponent'], data.loc[row, 'team']
            data.loc[row, 'home'], data.loc[row, 'oppo_home'] = data.loc[row, 'oppo_home'], data.loc[row, 'home']
            data.loc[row, 'team_rank'], data.loc[row, 'oppo_rank'] = data.loc[row, 'oppo_rank'], data.loc[row, 'team_rank']
            data.loc[row, 'team_xG'], data.loc[row, 'oppo_xG'] = data.loc[row, 'oppo_xG'], data.loc[row, 'team_xG']
            data.loc[row, 'team_win_rate'], data.loc[row, 'oppo_win_rate'] = data.loc[row, 'oppo_win_rate'], data.loc[row, 'team_win_rate']
    return data

# format data into homeaway version
def formatdata_homeaway(data):
    for row in data.index.tolist():
        if data.loc[row, 'home'] == 0:
            pass
        else:
            # need to swap the value
            data.loc[row, 'result'] =  'w' if data.loc[row, 'result'] == 'l' else 'l' if data.loc[row, 'result'] == 'w' else 'd'
            data.loc[row, 'team'], data.loc[row, 'opponent'] = data.loc[row, 'opponent'], data.loc[row, 'team']
            data.loc[row, 'team_rank'], data.loc[row, 'oppo_rank'] = data.loc[row, 'oppo_rank'], data.loc[row, 'team_rank']
            data.loc[row, 'team_xG'], data.loc[row, 'oppo_xG'] = data.loc[row, 'oppo_xG'], data.loc[row, 'team_xG']
            data.loc[row, 'team_win_rate'], data.loc[row, 'oppo_win_rate'] = data.loc[row, 'oppo_win_rate'], data.loc[row, 'team_win_rate']
            data.loc[row, 'team_score'], data.loc[row, 'oppo_score'] = data.loc[row, 'oppo_score'], data.loc[row, 'team_score']
            data.loc[row, 'team_conced'], data.loc[row, 'oppo_conced'] = data.loc[row, 'oppo_conced'], data.loc[row, 'team_conced']
            data.loc[row, 'team_shot_attempt'], data.loc[row, 'oppo_shot_attempt'] = data.loc[row, 'oppo_shot_attempt'], data.loc[row, 'team_shot_attempt']
            data.loc[row, 'team_recent_goals'], data.loc[row, 'oppo_recent_goals'] = data.loc[row, 'oppo_recent_goals'], data.loc[row, 'team_recent_goals']
            data.loc[row, 'team_recent_conced'], data.loc[row, 'oppo_recent_conced'] = data.loc[row, 'oppo_recent_conced'], data.loc[row, 'team_recent_conced']
            data.loc[row, 'team_recent_state'], data.loc[row, 'oppo_recent_state'] = data.loc[row, 'oppo_recent_state'], data.loc[row, 'team_recent_state']
            data.loc[row, 'team_recent_xG'], data.loc[row, 'oppo_recent_xG'] = data.loc[row, 'oppo_recent_xG'], data.loc[row, 'team_recent_xG']
    data['home'], data['oppo_home'] = 0, 1
    data['team_stage'], data['oppo_stage'] = data['stage'], data['stage']
    data.drop(columns = ['y', 'team_draw_rate', 'stage'], inplace = True)
    return data  

# Accumulate points feature utils 
def get_accumalate_points(team, oppo, date, fixture, dict):
   fixture_data = fixture.loc[fixture['Date'] < minus_one_day(str(date).split(' ')[0], return_str=False)]
   match_patten = [' VS '.join([team, oppo]), ' VS '.join([oppo, team])]
   select_df = fixture_data.loc[fixture['MatchName'].isin(match_patten)]

   if select_df.shape[0] != 0:
      result = {team: 0, oppo: 0}
      for idx, x in select_df.iterrows():
         win_team = x.HomeTeam if x.Result == 'H' else x.AwayTeam if x.Result == 'A' else 'Draw' 
         if win_team == 'Draw':
            result[team] += 1
            
            result[oppo] += 1
         elif win_team == team:
            result[team] += 3
         else:
            result[oppo] += 3
      return result[team], result[team] / select_df.shape[0], result[oppo], result[oppo]/select_df.shape[0]

   else:
        return dict[team][0], dict[team][1], dict[oppo][0], dict[oppo][1]

def calcualte_recent_points(df, team, bonus, bonus_list):
    def points(x, team, bonus, bonus_list):
        if team == x['team']:
            if x['result'] == 'l':
                points = 0 
            elif x['result'] == 'd':
                points = 1 + bonus if x['opponent'] in bonus_list else 1
            else:
                points = 3 + bonus if x['opponent'] in bonus_list else 3
        else:
            if x['result'] == 'l':
                points = 3 + bonus if x['team'] in bonus_list else 3 
            elif x['result'] == 'd':
                points = 1 + bonus if x['team'] in bonus_list else 1
            else:
                points = 0
        return points
    df['points'] = df.apply(lambda x: points(x, team,bonus,bonus_list), axis = 1)
    df = df[['Season','id', 'datetime', 'points']]
    df['recent_game_points'] = df['points'].rolling(3, closed = 'left').sum()
    return df
    
def add_recent_points(row,  bonus, bonus_list, final_df_sorted):
    season = int(row['Season'].split('-')[0])
    team_data, oppo_data = select_team_data(final_df_sorted, row['team'], season), select_team_data(final_df_sorted, row['opponent'],season)
    na_list = team_data.iloc[0:3].id.to_list()
    team_recent_df = calcualte_recent_points(team_data, row['team'], bonus, bonus_list)
    oppo_recent_df = calcualte_recent_points(oppo_data, row['opponent'], bonus, bonus_list)
    if row['id'] in na_list:
        # need last season to support
        last_season = season - 1
        team_data, oppo_data = select_team_data(final_df_sorted, row['team'], last_season), select_team_data(final_df_sorted, row['opponent'], last_season)
        # no last team data construct a new data frame with stats
        # if team_data.shape[0] == 0:
        #     team_last_recent_df, oppo_last_recent_df = pd.DataFrame({'team':[row['team']]*3, 'points': [0.964] * 3})
        # elif oppo_data.shape[0] == 0:
        #     oppo_last_recent_df = pd.DataFrame({'team':[row['team']]*3, 'points': [0.964] * 3})
        # else:
        #     team_last_recent_df = calcualte_recent_points(team_data, row['team'], bonus, bonus_list)
        #     oppo_last_recent_df = calcualte_recent_points(oppo_data, row['opponent'], bonus, bonus_list)
        team_last_recent_df =  calcualte_recent_points(team_data, row['team'], bonus, bonus_list) if team_data.shape[0] != 0 else pd.DataFrame({'team':[row['team']]*3, 'points': [0.964] * 3})
        oppo_last_recent_df =  calcualte_recent_points(oppo_data, row['opponent'], bonus, bonus_list) if oppo_data.shape[0] != 0 else pd.DataFrame({'team':[row['team']]*3, 'points': [0.964] * 3})
        if row['id'] == na_list[0]:
            team_recent_points, oppo_recent_points = team_last_recent_df.iloc[-3:].points.sum(), oppo_last_recent_df.iloc[-3:].points.sum()
        elif row['id'] == na_list[1]:
             team_recent_points, oppo_recent_points = team_last_recent_df.iloc[-2:].points.sum() + team_recent_df.iloc[0].points, oppo_last_recent_df.iloc[-2:].points.sum() + oppo_recent_df.iloc[0].points
        else:
             team_recent_points, oppo_recent_points = team_last_recent_df.iloc[-1].points + team_recent_df.iloc[:2].points.sum(), oppo_last_recent_df.iloc[-1].points + oppo_recent_df.iloc[:2].points.sum()   
    else:
        team_recent_points,oppo_recent_points = team_recent_df.loc[team_recent_df['id'] == row['id'], 'points'].values[0], oppo_recent_df.loc[oppo_recent_df['id'] == row['id'], 'points'].values[0]
    return team_recent_points, oppo_recent_points
