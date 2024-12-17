# Load libraries
library(caret)
library(randomForest)
library(xgboost)
library(ggplot2)

if (!require(pROC)) install.packages("pROC")
library(pROC)

# Step 1: Load Dataset (using iris as an example)
data(iris)
data <- iris
data$Species <- as.factor(ifelse(data$Species == "setosa", 1, 0)) # Binary target

# Step 2: Train-Test Split
set.seed(123)
trainIndex <- createDataPartition(data$Species, p = 0.8, list = FALSE)
trainData <- data[trainIndex,]
testData <- data[-trainIndex,]

# Step 3: Train Random Forest Model
model_rf <- randomForest(Species ~ ., data = trainData, ntree = 100)
pred_rf <- predict(model_rf, testData)
confusionMatrix(pred_rf, testData$Species)

# Step 4: Train XGBoost Model
train_matrix <- xgb.DMatrix(data = as.matrix(trainData[, -5]), label = as.numeric(trainData$Species) - 1)
test_matrix <- xgb.DMatrix(data = as.matrix(testData[, -5]), label = as.numeric(testData$Species) - 1)

params <- list(objective = "binary:logistic", eval_metric = "auc")
model_xgb <- xgb.train(params, train_matrix, nrounds = 50)

# Step 5: Model Evaluation (XGBoost)
pred_xgb <- predict(model_xgb, test_matrix)
print(paste("AUC:", roc(testData$Species, pred_xgb)$auc))

# Step 6: Feature Importance Visualization
importance <- varImp(model_rf)
plot(importance)

