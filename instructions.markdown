Below are the round multipliers for each available barrier to the game. An example of using the below coefficients and intercept is as follows.

Predicted Rounds = intercept X multiplier_1 X multiplier_2

So if Stefan was playing with no revives and with juggernaunt banned:

Predicted Rounds = intercept X stefan_playing X jugg_removed X no_revives

## Types of Models (See dropdown)
- Least Error
  - A conservative model that aims to obtain the best cross-validated R-squared. Currently the model is heavily regularized.
- Raw
  - Simply a normal linear regression, until there is more data this model will be very reactive to individual rows/games. 