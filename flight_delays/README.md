# Thanksgiving Flight Delays

In this project I used air travel data from the Department of Transportation and weather data from Dark Sky to predict whether flights out of O'Hare Airport in Chicago would be delayed on arrival. I specifically focused on the month of November in order to hopefully be able to alleviate some of the stress of Thanksgiving travel.

I tried two approaches for this analysis. One approach used a single Random Forest model to predict on all of the data. The second approach used two different Random Forests, one to predict on flights that departed on time, and another on flights that departed late.

I achieved my best result with the single Random Forest model, with a positive prediction threshold of 0.45.

For a full write-up of this project, please visit my [blog](https://www.noah-halpern.com/de-stressing-thanksgiving/).
