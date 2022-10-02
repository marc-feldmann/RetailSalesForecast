# RetailSalesForecast
This project uses gradient boosted trees to predict Rossmann retail store sales in Germany six weeks ahead. It uses data containing daily sales of 1115 stores between 2013 and 2015.

For each Rossmann store, and each day six weeks ahead, 85% mean forecasting accuracy is achieved by the trained model. This is very close to model performance on holdout data:

![XGBoost_ModelEvaluation](https://user-images.githubusercontent.com/97337456/193450433-5da6d3ee-3a32-4fe4-81e8-a7c0ef5cc172.png)

The final tree (after last boost):

![LastTree](https://user-images.githubusercontent.com/97337456/193451074-ff79ea74-8103-44a8-ab29-44de05880ad9.png)

