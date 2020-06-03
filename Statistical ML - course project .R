
## Machine learning project
# CHENG Fangbei 20641288


library(corrplot)
library(caret)
library(tidyr)
library(MASS)
library(e1071)
library(rpart)
library(rpart.plot)
library(randomForest)
library(gbm)
library(tibble)
library(dplyr)
library(cowplot)
library(xgboost)
library(ROCR)
library(plotROC)


## Data Preprocessing

# Load data

data <- read.csv("/Users/chengfangbei/Downloads/ML project/train.csv")
songs <- read.csv("/Users/chengfangbei/Downloads/ML project/songs.csv")
members <- read.csv("/Users/chengfangbei/Downloads/ML project/members.csv")

# Merge dataset

data <- merge(data, songs, "song_id", all.x = TRUE)
data <- merge(data, members, "msno", all.x = TRUE)

# Omit NA rows
data <- na.omit(data)

# Check dataset balance
sum(data$target==1)/nrow(data)
sum(data$target==0)/nrow(data)

# Lable encode for categorial data

data$source_system_tab <- as.numeric(factor(data$source_system_tab))

data$source_screen_name <- as.numeric(factor(data$source_screen_name))

data$source_type <- as.numeric(factor(data$source_type))

data$genre_ids <- as.numeric(factor(data$genre_ids))

data$artist_name <- as.numeric(factor(data$artist_name))

data$composer <- as.numeric(factor(data$composer))

data$lyricist <- as.numeric(factor(data$lyricist))

data$gender <- as.numeric(factor(data$gender))

data$language <- as.numeric(factor(data$language))

data$city <- as.numeric(factor(data$city))

data$registered_via <- as.numeric(factor(data$registered_via))

data$bd <- as.numeric(factor(data$bd))



## Feature Engineering

# Remove insignificant features - "msno","song_id"

data$msno <- NULL
data$song_id <- NULL

# Correlation matrix

variable <- c("source_system_tab","source_screen_name","source_type","song_length","genre_ids","artist_name","composer","lyricist","gender","language", "city","registered_via","bd","registration_init_time","expiration_date")

variable <- data[variable]

res <- cor(variable)

corrplot(res, method = "shade",
         type = "full",
         diag = F,
         order = "hclust",
         tl.srt = 45, 
         tl.cex = 0.6, 
         tl.col = "black",
         cl.cex = 0.6)

# Remove high correlated feature - bd
data$bd <- NULL




## Model training

# Split train and test set

set.seed(2222)

train <- createDataPartition(y = data$target,
                             times = 1,
                             p = 0.7,
                             list = F)

trainset <- data[train,]
testset <- data[-train,]

trainset$target <- factor(trainset$target)
testset$target <- factor(testset$target)

newtrain <- trainset[sample(nrow(trainset),70000), ]
newtest <- testset[sample(nrow(testset),30000), ]


## Logistic regression

fit1 <- glm(target ~ ., data = newtrain, family = "binomial")
summary(fit1)
fit2 <- glm(target ~ .- source_system_tab - composer - city - language - registered_via, data = newtrain, family = "binomial" )
summary(fit2)

## LDA & QDA

fit3 <- lda(target ~ ., data=newtrain)
fit4 <- qda(target ~ ., data=newtrain)

# accuracy

prob1 <- predict(fit1, newtest, type = "response")
pred1 <- ifelse(prob1 > 0.5, "1", "0")

prob2 <- predict(fit2, newtest, type = "response")
pred2 <- ifelse(prob2 > 0.5, "1", "0")

pred3 <- predict(fit3, newtest, type="response")
pred4 <- predict(fit4, newtest, type="response")

# compare logistic regression, LDA and QDA

accuracy1 <- mean(pred1==newtest$target)
accuracy2 <- mean(pred2==newtest$target)
accuracy3 <- mean(pred3$class==newtest$target)
accuracy4 <- mean(pred4$class==newtest$target)

comp <- rbind(accuracy1,accuracy2,accuracy3,accuracy4)
rownames(comp) <- c("fit1","fit2","fit3","fit4")
colnames(comp) <- c("accuracy")
round(comp,4)

## Decision tree

dtree_fit <- rpart(target ~ ., data = newtrain, cp = 0.001)
y_predict <- predict(dtree_fit, newdata = newtest, type = "class")
dtree_accuracy <-confusionMatrix(newtest$target, y_predict)$overall['Accuracy']

rpart.plot(dtree_fit)

## Random forest

rf_fit <- randomForest(target~. , newtrain, importance=TRUE)
rf_predict <- predict(rf_fit, newdata = newtest)
rf_accuracy1 <- confusionMatrix(newtest$target, rf_predict)$overall['Accuracy']

# Variable importance 

variable_importance_plot <- function(model){
  
  imp <- varImp(model)
  imp["1"] <- NULL
  colnames(imp) <- c("Overall")
  imp <- rownames_to_column(imp,var="rowname")
  imp <- arrange(imp,-Overall)
  
  theme_set(theme_bw())
  
  ggplot <- ggplot(imp) +
    aes(x = reorder(rowname, Overall), weight = Overall, fill = -Overall) +
    geom_bar() +
    coord_flip() +
    xlab("Variables") +
    ylab("Importance") +
    theme(legend.position = "none")
  
  return(plot_grid(ggplot))
}

variable_importance_plot(rf_fit)

## SVM

svm_fit <- svm(target~ ., data = newtrain, kernel="linear")
svm_prob <- predict(svm_fit, type="prob", newdata = newtest)
svm_predict <- predict(svm_fit, newdata = newtest)
svm_accuracy <-confusionMatrix(newtest$target, svm_predict)$overall['Accuracy']

## Adaboost

ada_fit <- gbm(target~. , newtrain,
               distribution = "multinomial",
               n.trees = 500) 
ada_predict <- predict.gbm(ada_fit, newdata = newtest,
                           n.trees = 500,
                           type = "response")
labels = colnames(ada_predict)[apply(ada_predict, 1, which.max)]
ada_accuracy <-confusionMatrix(newtest$target, factor(labels))$overall['Accuracy']

## XGboost
train_x <- as.matrix(select(newtrain,-target))
test_x <- as.matrix(select(newtest,-target))
train_y <- as.matrix(mutate(select(newtrain,target),target = as.numeric(as.character(target))))
test_y <- as.matrix(mutate(select(newtest,target),target = as.numeric(as.character(target))))

train_xgb <- xgb.DMatrix(train_x, label = train_y)
test_xgb <- xgb.DMatrix(test_x, label = test_y)

params_xgb = list(
  nthreads = 10,
  booster = "gbtree",
  eval_metric = "aucpr",
  learning_rate = 0.015,
  max_depth = 4, 
  subsample = 1, 
  colsample_bytree = 0.6,
  min_child_weight = 3
)

xgb_fit <- xgb.train(params = params_xgb,
                     train_xgb,
                     watchlist = list(train = train_xgb,test = test_xgb),
                     nrounds=400)

xgb_predict <- predict(xgb_fit,
                       newdata = test_x,
                       ntreelimit = xgb_fit$best_iteration)

xgb_accuracy <- mean(as.numeric(xgb_predict > 0.5) == as.numeric(test_y))



## AUC score, ROC & overall accuracy compare

# AUC score

#logistic regression
predlog <- prediction(prob1, newtest$target)
auclog <- performance(predlog,"auc")@y.values[[1]]

# LDA
predlda <- prediction(pred3$posterior[,2], newtest$target)
auclda <- performance(predlda,"auc")@y.values[[1]]

# QDA
predqda <- prediction(pred4$posterior[,2], newtest$target)
aucqda<- performance(predqda,"auc")@y.values[[1]]

# Decision tree
probdtree <- predict(dtree_fit, newdata = newtest, type = "prob")[,2]
preddtree <- prediction(probdtree, newtest$target)
aucdtree <- performance(preddtree, measure = "auc")@y.values[[1]]

# Random forest

probrf <- predict(rf_fit, type="prob",newdata = newtest)[,2]
predrf <- prediction(probrf, newtest$target)
aucrf <- performance(predrf, measure = "auc")@y.values[[1]]

# XGboost

predxgb <- prediction(xgb_predict, newtest$target)
aucxgb <- performance(predxgb,"auc")@y.values[[1]]

# Compare

auc <- rbind(auclog,auclda,aucqda,aucdtree,aucrf,aucxgb)
rownames(auc) <- c("logistic regression",
                   "LDA",
                   "QDA",
                   "Decision tree",
                   "Random forest",
                   "XGBoost")
auc[order(-auc[,1]),]


## ROC Curve
score_train = data.frame("logistic" = prob1,
                         "LDA" = pred3$posterior[,2],
                         "QDA" = pred4$posterior[,2],
                         "Decision tree" = probdtree,
                         "Random forest" = probrf,
                         "XGboost" = xgb_predict,
                         "obs" = as.numeric(newtest$target) - 1)

ggplot(gather(score_train,key = "Method", value = "score", -obs)) +
  aes(d = obs,
      m = score,
      color = Method) +
  geom_roc(labels = F, pointsize = 0, size = 0.6) +
  xlab("Specificity") +
  ylab("Sensitivity") +
  ggtitle("ROC Curve", subtitle = "Test dataset")

## Overall accuracy

compa <- rbind(accuracy1, accuracy2, accuracy3, accuracy4,
               dtree_accuracy, rf_accuracy1, svm_accuracy, ada_accuracy, xgb_accuracy)
rownames(compa) <- c("logistic regression(full)",
                     "logistic regression(selected)",
                     "LDA",
                     "QDA",
                     "Decision tree",
                     "Random forest",
                     "SVM",
                     "AdaBoost",
                     "XGBoost")
compa[order(-compa[,1]),]

