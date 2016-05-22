## *******************************************************************************************
## Applied Machine Learning Project - Analysis Code File
## 
## Jack G. Riddle
## 
## *******************************************************************************************
# Clear Workspace
remove(list = ls())

# Load necessary Libraries
library(caret)
library(randomForest)
library(e1071)

# --- Create data frames containing the training and testing data. 
#  Note: these instructions require the data be in the same directory this file
base_train <- read.csv(file = "pml-training.csv", na.string = c("NA", ""))
base_test <- read.csv(file = "pml-testing.csv", na.string = c("NA", ""))

# --- Inspect the data set
#str(base_trn)
#str(tst)

# Separate "bad" features from pertinate ones 
ncnt <- apply(base_train, 2, function(var) sum(is.na(var)))
features_bad <- names(ncnt)[ncnt > 0]
features_good <- names(ncnt)[ncnt == 0]
base_train_tmp <- base_train[, features_good]

# Very useful caret package function that diagnoses predictors that have one unique value 
#  (i.e. are zero variance predictors) or predictors that are have both of the following 
#  characteristics: they have very few unique values relative to the number of samples and 
#  the ratio of the frequency of the most common value to the frequency of the second most 
#  common value is large
#nearZeroVar(base_train_tmp, saveMetrics = TRUE)  
base_train_red <- base_train_tmp[, -c(1 : 7)]

# --- Create new training and validation sets
# Set random seed
set.seed(29544)

# Divide training set into new training (90%) and validation (10%) sets 
inTrain <- createDataPartition(y = base_train_red$classe, p = .90, list = FALSE)
training <- base_train_red[ inTrain, ]
validation  <- base_train_red[-inTrain, ]

# --- Choose classifier based on accuracy agains cross validation set
# Candidates: random forest, stochastic gradient boosting and support vector machine
cf <- vector("numeric", length = 3)

# random forest result using validation set
modFit_rf <- randomForest(classe ~ ., data = training)
pred_rf <- predict(modFit_rf, validation)
cf_rf <- confusionMatrix(validation$classe, pred_rf)
cf[1] <- cf_rf$overall[[1]]

# stochastic gradient boosting result using validation set
modFit_gbm <- train(classe ~ ., method = "gbm", data = training, verbose = FALSE)
pred_gbm <- predict(modFit_gbm, validation)
cf_gbm <- confusionMatrix(validation$classe, pred_gbm)
cf[2] <- cf_gbm$overall[[1]]

# SVM result using validation set
modFit_svm <- svm(classe ~ ., data = training)
pred_svm <- predict(modFit_svm, validation)
cf_svm <- confusionMatrix(validation$classe, pred_svm)
cf[3] <- cf_svm$overall[[1]]
cf

# Out-of-sample error
err <- 1.0 - cf  
err

# --- Evaluate against test set
pred <- predict(modFit_rf, base_test)
pred
