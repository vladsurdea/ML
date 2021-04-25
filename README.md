# Electricity Demand Forecasting (EDF) using an LSTM-based RNN

The following project predicts the hourly electricity demand in a given market by using a LSTM-based RNN architecture. The prediction is based on historical data of the hourly electricity load available between 2006 and 2019, in the absence of covariates describing weather events during this period. The results of the model are compared with the baseline ARIMA model, traditionally used in the transport and distribution industry to forecast demand in the day-ahead and spot markets. The ARIMA model is a standard, econometric model that does not employ ML algorithms. A combination of human and automatic evaluations is performed. The preliminary results show that the LSTM-based RNN outperforms the traditional model in multiple areas, especially when it comes to predict sudden spikes in electricity load consumption.


<img width="448" alt="Readme" src="https://user-images.githubusercontent.com/60359645/115990308-230e0e80-a5cb-11eb-8f5c-793ebf590c31.png">
