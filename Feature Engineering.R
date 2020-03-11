library(data.table); library(ggplot2); library(dplyr); library(caret); library(bit64);
Project_Dir <- paste(Sys.getenv('home'), "/Kaggle/Santander Customer Satisfaction", sep = "")

####################################
## Run Data Preprocessing First!! ##

# 0-counts per observation:
count0s <- function(x) {return(sum(x == 0))}
countNAs <- function(x) {return(sum(x == -997))}
countInfs <- function(x) {return(sum(x == 99999999))}

San_All$num_0 <- apply(San_All[, -c("TARGET", "ID"), with = FALSE], 1, FUN = count0s)
San_All$num_NA <- apply(San_All[, -c("TARGET", "ID"), with = FALSE], 1, FUN = countNAs)
San_All$num_NA <- apply(San_All[, -c("TARGET", "ID"), with = FALSE], 1, FUN = countInfs)

# Replace var3 (customer nationality) unknowns with most common value:
San_All$var3 <- ifelse(San_All$var3 == -997, 2, San_All$var3)

# Segment out num_var4 (number of bank products) where it is 0:
San_All$num_var4_0 <- ifelse(San_All$num_var4 == 0, 1, 0)

# Split var38 (some measurement of value?) into multiple variables, log the rest:
San_All$var38_mc <- ifelse(round(San_All$var38) == 117311, 1, 0)
San_All$log_var38 <- ifelse(San_All$var38_mc == 0, log(San_All$var38), 0)

# Make use of var15 (age of customer) distribution:
San_All$var15_young <- ifelse(San_All$var15 < 23, 1, 0)
San_All$var15_old <- ifelse(San_All$var15 > 89, 1, 0)
San_All$var15_mid <- ifelse(San_All$var15 > 25 & San_All$var15 < 50, 1, 0)

# Split saldo_var30 into multiple variables:
San_All$saldo_var30_0 <- ifelse(San_All$saldo_var30 < 1, 1, 0)
San_All$log_saldo_var30 <- ifelse(San_All$saldo_var30_0 == 0, log(San_All$saldo_var30), -1)

# Segment var36:
San_All$var36_99 <- ifelse(San_All$var36 == 99, 1, 0)

################################
## Re-separate Train and Test ##

San_Train <- San_All[!(ID %in% test_id)]
San_Test <- San_All[ID %in% test_id][, -"TARGET", with = FALSE]

# Limit the values of the test dataset to those of the train dataset (might help):
for(f in colnames(San_Train)) {
  lim <- min(San_Train[, f])
  San_Test[San_Test[, f] < lim, f] <- lim

  lim <- max(San_Train[, f])
  San_Test[San_Test[, f] > lim, f] <- lim
}

# Clean up workspace:
rm(f, lim)