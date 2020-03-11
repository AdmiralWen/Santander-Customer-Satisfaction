library(data.table); library(ggplot2); library(dplyr); library(bit64);
Project_Dir <- paste(Sys.getenv('home'), "/Kaggle/Santander Customer Satisfaction", sep = "")

##################################
## Data Input and Preprocessing ##

# Data Input (Need to convert Int64 to standard Int):
San_Train <- fread(paste(Project_Dir, "/Data/train.csv", sep = ""), stringsAsFactors = TRUE, data.table = TRUE)
San_Test <- fread(paste(Project_Dir, "/Data/test.csv", sep = ""), stringsAsFactors = TRUE, data.table = TRUE)
San_Train[San_Train == 9999999999] <- 99999999
San_Test[San_Test == 9999999999] <- 99999999
San_Train[San_Train == -999999] <- -997
San_Test[San_Test == -999999] <- -997

temp_train <- sapply(San_Train, is.integer64)
temp_train <- names(temp_train[temp_train == TRUE])
for(i in temp_train) {
	San_Train[[i]] <- as.integer(San_Train[[i]])
}
temp_test <- sapply(San_Test, is.integer64)
temp_test <- names(temp_test[temp_test == TRUE])
for(i in temp_test) {
	San_Test[[i]] <- as.integer(San_Test[[i]])
}

test_id <- San_Test$ID
San_All_Orig <- rbind(San_Train, San_Test, fill = TRUE)

# Remove some unnecessary variables:
zero_variance <- c("ind_var2_0", "ind_var2", "ind_var27_0", "ind_var28_0", "ind_var28", "ind_var27", "ind_var41",
                   "ind_var46_0", "ind_var46", "num_var27_0", "num_var28_0", "num_var28", "num_var27", "num_var41",
                   "num_var46_0", "num_var46", "saldo_var28", "saldo_var27", "saldo_var41", "saldo_var46",
                   "imp_amort_var18_hace3", "imp_amort_var34_hace3", "imp_reemb_var13_hace3", "imp_reemb_var33_hace3",
                   "imp_trasp_var17_out_hace3", "imp_trasp_var33_out_hace3", "num_var2_0_ult1", "num_var2_ult1",
                   "num_reemb_var13_hace3", "num_reemb_var33_hace3", "num_trasp_var17_out_hace3",
                   "num_trasp_var33_out_hace3", "saldo_var2_ult1", "saldo_medio_var13_medio_hace3")

San_All <- San_All_Orig[, -zero_variance, with = FALSE]
San_All <- data.table(unique(as.matrix(San_All), MARGIN = 2))

########################
## Clean up workspace ##

rm(list = ls(pattern = "All_|_new|_level|na_|dummies|temp_"))
rm(i, zero_variance)