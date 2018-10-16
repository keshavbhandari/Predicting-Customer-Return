setwd("C:/Users/kbhandari/Desktop/Chandan/Ad Hoc")

feature_engineering <- function(dep_timeframe, n_days_back){
  require(data.table)
  require(lubridate)
  require(dplyr)
  
  transactions <- fread("BCF_forEN1.csv", data.table = F)
  transactions$ITEMID <- NULL
  transactions$LIKELYBUYER <- NULL
  transactions <- transactions[transactions$AMOUNT > 0, ]
  transactions$TXDATE <-as.Date(as.character(transactions$TXDATE), "%d%B%Y")
  
  dep_timeframe <- as.Date(dep_timeframe) - n_days_back
  one_week_ago <- dep_timeframe - 7
  one_month_ago <- dep_timeframe - 31
  one_quarter_ago <- dep_timeframe - 90
  six_months_ago <- dep_timeframe - 180
  one_year_ago <- dep_timeframe - 365
  two_years_ago <- dep_timeframe - 731
  same_month_last_year <- c(dep_timeframe-396,dep_timeframe-365)
  
  #Reducing to Transaction Level
  print("Reducing to Transaction Level")
  transactions <- transactions %>%
    group_by(CUSTID, TXDATE) %>%
    summarise(QUANTITY = sum(QUANTITY),
              AMOUNT = sum(AMOUNT)) %>%
    as.data.frame()
  
  #Creating Data
  print("Creating Data")
  data <- transactions %>%
    group_by(CUSTID) %>%
    summarise(
              #Dependent Variable
              actual = if(max(TXDATE)>dep_timeframe) 1 else 0,
              exclude = if(min(TXDATE)>dep_timeframe) 1 else 0,
              #Days since 1st and last visit
              d_first_visit = as.numeric(dep_timeframe - min(TXDATE[which(TXDATE <= dep_timeframe)])),
              d_last_visit = as.numeric(dep_timeframe - max(TXDATE[which(TXDATE <= dep_timeframe)]))
              ) %>%
    as.data.frame()
  
  print(table(is.na(data$d_first_visit)))
  print(table(is.na(data$d_last_visit)))
  
  #Average monthly visits
  print("Creating average monthly visits")
  sum_monthly_visits <- transactions %>%
    filter(TXDATE <= dep_timeframe) %>%
    mutate(month = format(TXDATE, "%m")) %>%
    group_by(CUSTID, month) %>%
    summarise(sum_monthly_visits = n_distinct(TXDATE)) %>%
    as.data.frame()
  
  avg_monthly_visits <- sum_monthly_visits %>%
    group_by(CUSTID) %>%
    summarise(avg_monthly_visits = mean(sum_monthly_visits),
              stdev_monthly_visits = sd(sum_monthly_visits)) %>%
    as.data.frame()
  rm(sum_monthly_visits)
  
  data <- merge(data, avg_monthly_visits, by.x = "CUSTID", by.y ="CUSTID", all.x=TRUE)
  print(table(is.na(data$avg_monthly_visits)))
  rm(avg_monthly_visits)
  
  #RFM Variables
  print("Creating RFM variables")
  rfm_features <- transactions %>%
    group_by(CUSTID) %>%
    filter(TXDATE <= dep_timeframe) %>%
    arrange(TXDATE) %>%
    summarise(
      avg_time_between_txns = as.numeric(mean(diff(TXDATE))),
      sd_time_between_txns = as.numeric(sd(diff(TXDATE))),
      overall_txn_cnt = n_distinct(TXDATE),
      txn_cnt_one_week = n_distinct(TXDATE[which(TXDATE > one_week_ago)]),
      txn_cnt_one_month = n_distinct(TXDATE[which(TXDATE > one_month_ago)]),
      txn_cnt_one_quarter = n_distinct(TXDATE[which(TXDATE > one_quarter_ago)]),
      txn_cnt_six_months = n_distinct(TXDATE[which(TXDATE > six_months_ago)]),
      txn_cnt_one_year = n_distinct(TXDATE[which(TXDATE > one_year_ago)]),
      txn_cnt_two_year = n_distinct(TXDATE[which(TXDATE > two_years_ago)]),
      txn_cnt_sm_mth_lst_yr = n_distinct(TXDATE[which(TXDATE > same_month_last_year[1] & TXDATE < same_month_last_year[2])]),
      overall_qty_cnt = sum(QUANTITY),
      qty_avg = mean(QUANTITY),
      qty_cnt_one_week = sum(QUANTITY[which(TXDATE > one_week_ago)]),
      qty_avg_one_week = mean(QUANTITY[which(TXDATE > one_week_ago)]),
      qty_cnt_one_month = sum(QUANTITY[which(TXDATE > one_month_ago)]),
      qty_avg_one_month = mean(QUANTITY[which(TXDATE > one_month_ago)]),
      qty_cnt_one_quarter = sum(QUANTITY[which(TXDATE > one_quarter_ago)]),
      qty_avg_one_quarter = mean(QUANTITY[which(TXDATE > one_quarter_ago)]),
      qty_cnt_six_months = sum(QUANTITY[which(TXDATE > six_months_ago)]),
      qty_avg_six_months = mean(QUANTITY[which(TXDATE > six_months_ago)]),
      qty_cnt_one_year = sum(QUANTITY[which(TXDATE > one_year_ago)]),
      qty_avg_one_year = mean(QUANTITY[which(TXDATE > one_year_ago)]),
      
      qty_cnt_two_year = sum(QUANTITY[which(TXDATE > two_years_ago)]),
      qty_avg_two_year = mean(QUANTITY[which(TXDATE > two_years_ago)]),
      qty_cnt_sm_mth_lst_yr = sum(QUANTITY[which(TXDATE > same_month_last_year[1] & TXDATE < same_month_last_year[2])]),
      overall_amt = sum(AMOUNT),
      avg_amt = mean(AMOUNT),
      amt_one_week = sum(AMOUNT[which(TXDATE > one_week_ago)]),
      avg_amt_one_week = mean(AMOUNT[which(TXDATE > one_week_ago)]),
      amt_one_month = sum(AMOUNT[which(TXDATE > one_month_ago)]),
      avg_amt_one_month = mean(AMOUNT[which(TXDATE > one_month_ago)]),
      amt_one_quarter = sum(AMOUNT[which(TXDATE > one_quarter_ago)]),
      avg_amt_one_quarter = mean(AMOUNT[which(TXDATE > one_quarter_ago)]),
      amt_six_months = sum(AMOUNT[which(TXDATE > six_months_ago)]),
      avg_amt_six_months = mean(AMOUNT[which(TXDATE > six_months_ago)]),
      amt_one_year = sum(AMOUNT[which(TXDATE > one_year_ago)]),
      avg_amt_one_year = mean(AMOUNT[which(TXDATE > one_year_ago)]),
      amt_two_year = sum(AMOUNT[which(TXDATE > two_years_ago)]),
      avg_amt_two_year = mean(AMOUNT[which(TXDATE > two_years_ago)]),
      amt_sm_mth_lst_yr = sum(AMOUNT[which(TXDATE > same_month_last_year[1] & TXDATE < same_month_last_year[2])])
    ) %>%
    as.data.frame()
  
  data <- merge(data, rfm_features, by.x = "CUSTID", by.y ="CUSTID", all.x=TRUE)
  rm(rfm_features)
  
  #Seasonality
  print("Creating seasonality variables")
  transactions$WEEKDAY <- weekdays(transactions$TXDATE, abbr = TRUE)
  transactions$Wk_End_Transaction_Cnt = ifelse(transactions$WEEKDAY %in% c("Fri","Sat","Sun") & transactions$TXDATE<=dep_timeframe,1,0)
  transactions$WEEKDAY <- NULL
  transactions$DAY <- wday(ymd(transactions$TXDATE),week_start = getOption("lubridate.week.start", 1))
  transactions$START <- as.numeric(dep_timeframe+n_days_back-transactions$TXDATE)
  transactions$N_WEEK <- floor(as.numeric(transactions$START/7))
  transactions$REMAINDER <- as.numeric(transactions$START%%7)
  transactions$START <- NULL
  transactions$N_Wk_End_Start <- transactions$N_WEEK + ifelse(transactions$DAY==1 & transactions$REMAINDER>4,1,
                                                              ifelse(transactions$DAY==2 & transactions$REMAINDER>3,1,
                                                                     ifelse(transactions$DAY==3 & transactions$REMAINDER>2,1,
                                                                            ifelse(transactions$DAY==4 & transactions$REMAINDER>1,1,
                                                                                   ifelse(transactions$DAY==5 & transactions$REMAINDER>=0,1,
                                                                                          ifelse(transactions$DAY==6 & transactions$REMAINDER>=0,1,
                                                                                                 ifelse(transactions$DAY==7 & transactions$REMAINDER>=0,1,0)))))))
  
  transactions$DAY <- NULL
  transactions$N_WEEK <- NULL
  transactions$REMAINDER <- NULL
  
  seasonality <- transactions %>%
    group_by(CUSTID) %>%
    filter(TXDATE<=dep_timeframe) %>%
    summarise(Wk_End_Transaction_Cnt = sum(Wk_End_Transaction_Cnt),
              N_wknd_Frst_Tns = max(N_Wk_End_Start),
              N_wknd_Lst_Tns = min(N_Wk_End_Start)) %>%
    as.data.frame()
  
  data <- merge(data, seasonality, by.x = "CUSTID", by.y ="CUSTID", all.x=TRUE)
  print(table(is.na(data$Wk_End_Transaction_Cnt)))
  rm(seasonality)
  
  #Handling Missing Values
  data[is.na(data)] <- 0
  
  #Excluding single transactions after the holdout period
  data <- data[data$exclude!=1,]
  data$exclude <- NULL
  head(data)
  print(table(data$actual))
  
  return(data)

}

data <- feature_engineering(dep_timeframe = "2017-08-04", n_days_back = 31)
apply(data, 2, function(x) any(is.na(x) | is.infinite(x)))
save(data, file = "training_data.rda")

set.seed(123)
sample <- sample.int(n = nrow(data), size = floor(.8*nrow(data)), replace = F)
validation  <- data[-sample,]
train <- data[sample,]
rm(sample)
invisible(gc())

library(car)
library(MASS)
log_model <- step(glm(actual ~ ., data=train[,-which(names(train) %in% c("CUSTID"))], family=binomial()), direction = "backward")
log_model <- glm(actual ~ ., data=train[,-which(names(train) %in% c("CUSTID"))], family=binomial())
summary(log_model)
vif(log_model)

# Predicted Probabilities
prediction <- predict(log_model,newdata = validation,type="response")
library(pROC)
validation$actual <- as.factor(validation$actual)
rocCurve   <- roc(response = validation$actual, predictor = prediction, levels = rev(levels(validation$actual)))

#Metrics - Fit Statistics
predclass <-ifelse(prediction>coords(rocCurve,"best")[1],1,0)
Confusion <- table(Predicted = predclass,Actual = validation$actual)
AccuracyRate <- sum(diag(Confusion))/sum(Confusion)
Gini <-2*auc(rocCurve)-1
AUCmetric <- data.frame(c(coords(rocCurve,"best"),AUC=auc(rocCurve),AccuracyRate=AccuracyRate,Gini=Gini))
AUCmetric <- data.frame(rownames(AUCmetric),AUCmetric)
rownames(AUCmetric) <-NULL
names(AUCmetric) <- c("Metric","Values")
AUCmetric

Confusion 
plot(rocCurve)

#Removing variables and processing data again
rm(data, train, validation, AccuracyRate, Confusion, Gini, predclass, prediction, rocCurve, AUCmetric)

save(log_model, file = "log_model.rda")

data <- feature_engineering(dep_timeframe = "2017-08-04", n_days_back = 0)

#Predicting on Test
data$pred = predict(log_model,newdata = data,type="response")
df <- data[,c("CUSTID","pred")]
write.csv(df,"logit_scores.csv")
