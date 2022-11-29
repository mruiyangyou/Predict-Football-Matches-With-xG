library(BradleyTerry2)
library(ggplot2)
library(tidyverse)
library(ggrepel)
#
data_path = "/Users/marceloyou/Desktop/Xg-Prediction/data/matchdata/2015-2022matchdata(training).csv"
df = read.csv(data_path)
head(df)
colSums(is.na(df))

# baseline
df$team = as.factor(df$team)
df$opponent = as.factor(df$opponent)
baseline = BTm(y, player1 = team, player2 = opponent, id = "EPL_", data = df)
summary(baseline)

# vislualize the result
table_path = '/Users/marceloyou/Desktop/Xg-Prediction/data/table/2015-2022winaccuracy.csv'
table = read.csv(table_path)
rownames(table) = table$Team
table = table['win_accuracy']
head(table)

visualize_bta = function(model, table) {
    coef_df = data.frame(model$coefficients)
    colnames(coef_df) = "Beta"
    coef_df["EPL_Arsenal",] = 0
    result_df = merge(table, coef_df, by = 0)
    colnames(result_df)[1] = "Team"
    print(result_df)
    result_df %>%
        ggplot(aes(x = win_accuracy, y = Beta)) + 
            geom_point() + geom_text_repel(aes(label = Team), max.overlaps = 20) + 
            labs(x = 'Win Accuracy', y = 'Beta', 
                title = 'Bradley-Terry beta vs. Win %')
}
coef_df = data.frame(baseline$coefficients)
colnames(coef_df) = 'beta'
coef_df['EPL_Arsenal',] = 0
dim(coef_df)

result_df = merge(table, coef_df, by = 0)
colnames(result_df)[1] = 'Team'

visualize_bta(baseline, table)

result_df %>%
    ggplot(aes(x = win_accuracy, y = beta)) + 
        geom_point() + geom_text_repel(aes(label = Team), max.overlaps = 20) + 
        labs(x = 'Win Accuracy', y = 'Beta', 
            title = 'Bradley-Terry beta vs. Win %')

# add rank
data("flatlizards", package = "BradleyTerry2")
rank_df = subset(df, select = c(team,opponent, y))
colnames(rank_df) = c('home.team','away.team', 'y')
rank_df$home.team = as.factor(rank_df$home.team)
rank_df$away.team = as.factor(rank_df$away.team)
add_rank = BTm(y, player1 = home.team, player2 = away.team, id = 'team', data = rank_df)
rank_df$home.team = data.frame(team = rank_df$home.team, rank = df$team_rank)
rank_df$away.team = data.frame(team = rank_df$away.team, rank = df$oppo_rank)
head(rank_df)
add_rank2 = update(add_rank, formula = ~ team + log(rank))
summary(add_rank2)

# add history record
rank_df$home.team = data.frame(rank_df$home.team, history_record = df$team_win_rate)
rank_df$away.team = data.frame(rank_df$away.team, history_record = df$oppo_win_rate)
head(rank_df)
model_rankrecord = update(add_rank2, formula = ~team + log(rank) + history_record)
summary(model_rankrecord)

# add xG Prediction
rank_df$home.team = data.frame(rank_df$home.team, xG = df$team_xG)
rank_df$away.team = data.frame(rank_df$away.team, xG = df$oppo_xG)
head(rank_df)
model_rankrecordxg = update(model_rankrecord, formula = ~team + log(rank) + history_record + xG)
summary(model_rankrecordxg)

# add recent score
rank_df$home.team = data.frame(rank_df$home.team, recent_goal = df$team_recent_goals)
rank_df$away.team = data.frame(rank_df$away.team, recent_goal = df$oppo_recent_goals)
head(rank_df)
model5 = update(model_rankrecord, formula = ~team + log(rank) + history_record + xG + recent_goal) # update the 5 goals
summary(model5)


# add recent conced
rank_df$home.team = data.frame(rank_df$home.team, recent_conced = df$team_recent_conced)
rank_df$away.team = data.frame(rank_df$away.team, recent_conced = df$oppo_recent_conced)
head(rank_df)
model6 = update(model5, formula = ~team + log(rank) + history_record + xG + recent_conced) # update the 5 goals
summary(model6)

# add recent state
rank_df$home.team = data.frame(rank_df$home.team, recent_state = df$team_recent_state)
rank_df$away.team = data.frame(rank_df$away.team, recent_state = df$oppo_recent_state)
head(rank_df)
model7 = update(model6, formula = ~team + log(rank) + history_record + xG + recent_conced + recent_state) # update the 5 goals
summary(model7)



