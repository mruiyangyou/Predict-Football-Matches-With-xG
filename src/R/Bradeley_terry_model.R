# load package
library(ggplot2)
library(BradleyTerry2)

# Read data set
data_path = "/Users/marceloyou/Desktop/Xg-Prediction/data/matchdata/2014-2016match.csv"
df = read.csv(data_path)
head(df)
dim(df)
colSums(is.na(df))

# select useful data
data = subset(df, select = c(Season, datetime, y, team, team_xG, home, 
       team_rank, team_win_rate, opponent, oppo_xG, oppo_home, 
       oppo_rank, oppnent_win_rate))
head(data)
colSums(is.na(data))
data = data[order(as.Date(data$datetime, format = '%Y-%m-%d %H:%M:%S')),]

# take arsenal
arsenal = subset(df, team == "Arsenal" | opponent == 'Arsenal')
head(arsenal)
dim(arsenal)
arsenal = arsenal[order(as.Date(arsenal$datetime, format = '%Y-%m-%d %H:%M:%S')),]
arsenal = subset(arsenal, select = c(Season, datetime, y, team, team_xG, home, 
       team_rank, team_win_rate, opponent, oppo_xG, oppo_home, 
       oppo_rank, oppnent_win_rate))
dim(arsenal)

# base line no   
bt_data = read.csv('/Users/marceloyou/Desktop/Xg-Prediction/data/matchdata/B-T-Moedl-data/2014_2016bt_data.csv')
head(bt_data)
bt_data$team = as.factor(bt_data$team)
bt_data$opponent = as.factor(bt_data$opponent)
pl_model = BTm(y, player1 = team, player2 = opponent, data = bt_data)
length(pl_model$coefficients)
summary(pl_model)

coef_df = data.frame(pl_model$coefficients)
coef_df$name = rownames(coef_df)
coef_df[order(coef_df$beta, decreasing = TRUE), ]
