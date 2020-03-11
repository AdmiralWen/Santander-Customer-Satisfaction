library(data.table); library(ggplot2); library(dplyr); library(caret); library(randomForest);
library(doParallel); library(foreach);

Project_Dir <- paste(Sys.getenv('home'), "/Kaggle/Santander Customer Satisfaction", sep = "")

###################################
# Run Feature Engineering First!! #
###################################

#############################
## Fit Random Forest Model ##

RF_train <- San_Train[, -c("TARGET", "ID"), with = FALSE]
RF_target <- as.factor(San_Train[, "TARGET", with = FALSE][[1]])
RF_test <- San_Test[, -"ID", with = FALSE]
ID <- San_Test[, "ID", with = FALSE][[1]]; ID <- format(ID, scientific = FALSE);

clus <- makeCluster(4)
registerDoParallel(clus)
fit1 <- foreach(ntrees = rep(200, 4), .combine = combine, .multicombine = TRUE, .packages = "randomForest") %dopar% {
  randomForest(x = RF_train, y = RF_target, mtry = 50, ntree = ntrees, nodesize = 2)
}
stopCluster(clus)
summary(fit1)

# OOB error rate:
test <- predict(fit1)
confusionMatrix(test, RF_target)

# Predictions:
TARGET <- predict(object = fit1, newdata = RF_test, type = "prob")[, 2]
pred_data <- cbind(ID, TARGET)
write.csv(pred_data,
          file = paste(Project_Dir, "/Submissions/6. RandomForest.csv", sep = ""),
          row.names=FALSE,
          quote = FALSE
)