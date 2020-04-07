# Predicting Soccer Player Transfer Fees

For this project I used linear regression to try to predict transfer fees for soccer players in the top 5 European leagues. My predictions were based on each player's in-game stats from the season prior to his transfer.

My regression model ended up not performing very well at all, predicting fees with a RMSE of over £7 million.

There were many reasons I identified that contributed to the poor model performance. Among them were severe overfitting and a lack of player passing stats. Passing data would have been helpful because soccer is a relatively low-scoring sport, so scoring data is not always a good metric of player performance/value.

For a full write-up of this project please visit my [blog](https://www.noah-halpern.com/predicting-soccer-player-transfer-fees/).

## Special Thanks

Shoutout to Github user [ewanme](https://github.com/ewenme/transfers) who compiled the player transfer data that I used.
