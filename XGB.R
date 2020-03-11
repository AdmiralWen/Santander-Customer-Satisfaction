library(data.table); library(ggplot2); library(dplyr); library(caret); library(xgboost);
Project_Dir <- paste(Sys.getenv('home'), "/Kaggle/Santander Customer Satisfaction", sep = "")

###################################
# Run Feature Engineering First!! #
###################################

# Create response vector and training/testing matrices:
train_target <- San_Train[, "TARGET", with = FALSE][[1]]
train_matrix <- as.matrix(San_Train[, -c("ID", "TARGET"), with = FALSE])
ID <- San_Test[, "ID", with = FALSE][[1]]; ID <- format(ID, scientific = FALSE);
test_matrix <- as.matrix(San_Test[, -"ID", with = FALSE])

##########################
## GBM Cross-Validation ##

xgb_cv.params <- list("objective" = "binary:logistic",
                      "eval_metric" = "auc",
                      "eta" = 0.02,
                      "max.depth" = 6,
                      "subsample" = 0.7,
                      "colsample_bytree" = 0.7,
                      "min_child_weight" = 1,
                      "gamma" = 0,
                      "nthread" = 8
)

# Tune max.depth:
for(i in 4:10) {
  xgb_cv.params["max.depth"] <- i
  set.seed(1)
  xgb_cv1 <- xgb.cv(param = xgb_cv.params, data = train_matrix, label = train_target, nfold = 6, nrounds = 2000,
                    early.stop.round = 100, verbose = 0)
  print(paste("Depth=", i, " : ", max(xgb_cv1$test.auc.mean), sep = ""))
}
xgb_cv.params["max.depth"] <- 5

# Tune subsample and colsample_bytree:
for(i in seq(0.6, 0.8, 0.2)) {
  xgb_cv.params["subsample"] <- i
  for(j in seq(0.25, 0.35, 0.05)) {
    xgb_cv.params["colsample_bytree"] <- j
    set.seed(1)
    xgb_cv1 <- xgb.cv(param = xgb_cv.params, data = train_matrix, label = train_target, nfold = 6, nrounds = 2000,
                      early.stop.round = 100, verbose = 0)
    print(paste("Subsample=", i, " Colsample=", j, " : ", max(xgb_cv1$test.auc.mean), sep = ""))
  }
}
xgb_cv.params["subsample"] <- 0.8
xgb_cv.params["colsample_bytree"] <- 0.4

# Tune min_child_weight and gamma:
for(i in 1:3) {
  xgb_cv.params["min_child_weight"] <- i
  for(j in seq(0.0, 0.3, 0.1)) {
    xgb_cv.params["gamma"] <- j
    set.seed(1)
    xgb_cv1 <- xgb.cv(param = xgb_cv.params, data = train_matrix, label = train_target, nfold = 6, nrounds = 2000,
                      early.stop.round = 100, verbose = 0)
    print(paste("Child_Wgt=", i, " Gamma=", j, " : ", max(xgb_cv1$test.auc.mean), sep = ""))
  }
}
xgb_cv.params["min_child_weight"] <- 1
xgb_cv.params["gamma"] <- 0

##################################
## GBM Cross-Validation Round 2 ##

xgb_cv.params2 <- list("objective" = "binary:logistic",
                       "eval_metric" = "auc",
                       "eta" = 0.01,
                       "max.depth" = 5,
                       "subsample" = 0.8,
                       "colsample_bytree" = 0.4,
                       "min_child_weight" = 1,
                       "gamma" = 0
)

set.seed(1990)
xgb_cv2 <- xgb.cv(param = xgb_cv.params2, data = train_matrix, label = train_target, nfold = 8, nrounds = 3000,
                  early.stop.round = 150)

optimal_ntrees <- match(max(xgb_cv2$test.auc.mean), xgb_cv2$test.auc.mean) - 1
ggplot(data = xgb_cv2, aes(x = 1:length(xgb_cv2$test.auc.mean))) +
  geom_line(aes(y = xgb_cv2$train.auc.mean), size = 1) +
  geom_line(aes(y = xgb_cv2$test.auc.mean), color = 'red', size = 1) +
  geom_vline(xintercept = optimal_ntrees, color = 'blue')

#####################
## Fit Final Model ##

xgb.params <- list("objective" = "binary:logistic",
                   "eval_metric" = "auc",
                   "eta" = 0.01,
                   "max.depth" = 5,
                   "subsample" = 0.8,
                   "colsample_bytree" = 0.4,
                   "min_child_weight" = 1,
                   "gamma" = 0
)

set.seed(1990)
xgb_fit1 <- xgboost(param = xgb.params, data = train_matrix, label = train_target, nrounds = 1055)
summary(xgb_fit1)

names <- dimnames(train_matrix)[[2]]
importance_matrix <- xgb.importance(names, model = xgb_fit1)
xgb.plot.importance(importance_matrix[1:25, ])

# Run a few more times for XGB Enselmble:
set.seed(1378)
xgb_fit2 <- xgboost(param = xgb.params, data = train_matrix, label = train_target, nrounds = 1055)

set.seed(2849)
xgb_fit3 <- xgboost(param = xgb.params, data = train_matrix, label = train_target, nrounds = 1055)

set.seed(54)
xgb_fit4 <- xgboost(param = xgb.params, data = train_matrix, label = train_target, nrounds = 1055)

set.seed(916)
xgb_fit5 <- xgboost(param = xgb.params, data = train_matrix, label = train_target, nrounds = 1055)

######################
## Test Predictions ##

TARGET1 <- predict(xgb_fit1, test_matrix)
TARGET2 <- predict(xgb_fit2, test_matrix)
TARGET3 <- predict(xgb_fit3, test_matrix)
TARGET4 <- predict(xgb_fit4, test_matrix)
TARGET5 <- predict(xgb_fit5, test_matrix)
TARGET <- rowMeans(cbind(TARGET1, TARGET2, TARGET3, TARGET4, TARGET5))

pred_data <- cbind(ID, TARGET)
write.csv(pred_data,
          file = paste(Project_Dir, "/Submissions/4. XGB_Enselmble.csv", sep = ""),
          row.names=FALSE,
          quote = FALSE
)