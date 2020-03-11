options(java.parameters = "-Xmx8g")
library(data.table); library(ggplot2); library(dplyr); library(caret); library(extraTrees); library(AUC);
Project_Dir <- paste(Sys.getenv('home'), "/Kaggle/Santander Customer Satisfaction", sep = "")

###################################
# Run Feature Engineering First!! #
###################################

# Removal of some possibly useless variables:
var_removals <- c("delta_imp_amort_var18_1y3", "delta_imp_amort_var34_1y3", "delta_imp_reemb_var13_1y3",
                  "delta_imp_reemb_var17_1y3", "delta_imp_reemb_var33_1y3", "delta_imp_trasp_var17_in_1y3",
                  "delta_imp_trasp_var17_out_1y3", "delta_imp_trasp_var33_in_1y3", "delta_imp_trasp_var33_out_1y3",
                  "delta_num_reemb_var13_1y3", "delta_num_reemb_var17_1y3", "delta_num_reemb_var33_1y3",
                  "delta_num_trasp_var17_in_1y3", "delta_num_trasp_var17_out_1y3", "delta_num_trasp_var33_in_1y3",
                  "delta_num_trasp_var33_out_1y3")

San_Train <- San_Train[, -c(var_removals), with = FALSE]
San_Test <- San_Test[, -c(var_removals), with = FALSE]

################################
## Cross-Validate Extra Trees ##

nfolds <- 4
folds <- round(runif(nrow(San_Train), 1, nfolds))

San_CV <- San_Train[, -"ID", with = FALSE]

# Start Cross Validation:
AUC <- c()
for(f in 1:nfolds) {
  print(paste('fold', f))
  CV_train <- subset(San_CV[, -"TARGET", with = FALSE], folds != f)
  CV_target <- as.factor(subset(San_CV[, "TARGET", with = FALSE], folds != f)[[1]])
  CV_test <- subset(San_CV[, -"TARGET", with = FALSE], folds == f)
  CV_test_target <- as.factor(subset(San_CV[, "TARGET", with = FALSE], folds == f)[[1]])
  
  # Model fit:
  set.seed(1990)
  fit0 <- extraTrees(x = CV_train,
                     y = CV_target,
                     ntree = 500,
                     mtry = 150,
                     nodesize = 2,
                     numRandomCuts = 3,
                     numThreads = 8
  )
  
  pred1 <- predict(object = fit0, newdata = CV_test, probability = TRUE)[, 2]
  pred1 <- ifelse(pred1 < 0.00001, 0.00001, pred1)
  pred1 <- ifelse(pred1 > 0.99999, 0.99999, pred1)
  
  # Compute AUC:
  roc_val <- roc(predictions = pred1, labels = CV_test_target)
  AUC <- c(AUC, auc(roc_val))
  gc()
}
cat(unlist(AUC), ":", mean(unlist(AUC)))

##############################
## Fit Full Model & Predict ##

ET_train <- San_Train[, -c("target", "ID"), with = FALSE]
ET_target <- as.factor(San_Train[, "target", with = FALSE][[1]])
ET_test <- San_Test[, -"ID", with = FALSE]
ID <- San_Test[, "ID", with = FALSE][[1]]; ID <- format(ID, scientific = FALSE);

gc()
fit1 <- extraTrees(x = ET_train, y = ET_target, ntree = 850, mtry = 120, nodesize = 2,
                   numRandomCuts = 3, numThreads = 8)

TARGET <- predict(object = fit1, newdata = ET_test, probability = TRUE)[, 2]
pred_data <- cbind(ID, TARGET)
write.csv(pred_data,
          file = paste(Project_Dir, "/Submissions/ExtraTrees.csv", sep = ""),
          row.names=FALSE,
          quote = FALSE
)