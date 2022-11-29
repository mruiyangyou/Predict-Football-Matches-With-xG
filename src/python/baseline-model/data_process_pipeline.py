from utils import *
import numpy as np
import pandas as pd
import datetime
import os

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


# Simple data constrution
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

# data utils function
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


def get_current_rank(end_date, team, opponent, start_date, year):
    table = get_current_rank_data('EPL', year, start_date= start_date, end_date=end_date)
    data = pd.DataFrame(table[1:], columns=table[0])
    rank = data.loc[data['Team'] == team].index.to_list()[0] + 1
    opponent_rank = data.loc[data['Team'] == opponent].index.to_list()[0] + 1
    return rank, opponent_rank

def minus_one_day(dt, return_str = True):
    date = datetime.datetime.strptime(dt, '%Y-%m-%d') - datetime.timedelta(days = 1)
    return datetime.datetime.strftime(date,"%Y-%m-%d") if return_str else date


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
  


def get_team_recent(interval, prefix, df):

    df[f'{prefix}_recent_goals'] = df[f'{prefix}_score'].rolling(5, closed='left').mean()
    df[f'{prefix}_recent_conced'] = df[f'{prefix}_conced'].rolling(5, closed='left').mean()
    df[f'{prefix}_recent_state'] = df['y'].rolling(3, closed = 'left').mean()
    df[f'{prefix}_recent_xG'] = df[f'{prefix}_xG'].rolling(5, closed='left').mean()
    return df

# Data pipeline problem

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

def select_team_data(data, team):
    return data.loc[(data['opponent'] == team) | (data['team'] == team),:]
# CHange dataframe to Bradely Terry model format

'''
def Bt_model_trasformation(final_df):
    fafa
'''
