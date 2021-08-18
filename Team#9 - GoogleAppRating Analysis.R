library(dplyr)
library(corrplot)
library(stringr)
library(mlbench)
library(caret)
library(randomForest)
library("PerformanceAnalytics")
library(kernlab)
library(neuralnet)
library(C50)
library(gmodels)
library(vcd)

################# Read in the data ###################################################
data <- read.csv('googleplaystore.csv', stringsAsFactors= FALSE)
head(data)
str(data)
dim(data)

################# Data cleaning ######################################################
#Remove duplicate values
data <- data %>% distinct()
dim(data)
summary(data)

#Maxinum value at "Rating" is unsual, 19 instead of 5.Review the row that contains that value
data[data$Rating == "19",]
#It seems that all values in this row does not match with the column name => remove this row from data
data = data[-9991,]
dim(data)

#Replace NA values of "Rating" by mean 
data <- data %>% mutate(Rating = replace(Rating, is.na(Rating), mean(Rating, na.rm = TRUE)))
summary(data$Rating)

#Transform "Reviews" from character to numeric
data$Reviews <- as.numeric(data$Reviews)
summary(data$Reviews) 

#Transform "Size" to numeric
#"size" has two type of size given in MB and KB => make it uniform
data$Size <- str_replace_all(data$Size, "M", "")
data$Size <- ifelse(grepl( "k", data$Size), as.numeric(data$Size)/1000, data$Size)
#Replace "Varies with device" value by NA
data$Size <- na_if(data$Size, "Varies with device")
data$Size <- as.numeric(data$Size)
summary(data$Size)
#Replace NA value by mean of size 
data <- data %>%mutate(Size = replace(Size, is.na(Size), mean(Size, na.rm = TRUE)))
summary(data$Size)

#Transform "Installs" to numeric
data$Installs <- str_replace_all(data$Installs,"\\+","")
data$Installs <- str_replace_all(data$Installs,"\\,","")
data$Installs <- as.integer(data$Installs)
summary(data$Installs)

#Replace NA value in "Type"
data$Type = as.factor(data$Type)
summary(data$Type)
levels(data$Type) <-c("Free", "Free", "Paid")
summary(data$Type)

#Remove $ symbol from Price and convert it to numeric
data$Price = as.numeric(gsub("\\$", "", data$Price))
summary(data$Price)

####################### Data Analysis ###############################################
#Which app category has the highest number of reviews?
ggplot(data, aes(x= Category, y= Reviews, fill = Type)) +
  geom_bar(position='dodge',stat='identity') +
  coord_flip() +
  ggtitle("Number Of App Reviews Based On Category and Type")  

#Which category has the highest rating per Free and Paid apps?
ggplot(data, aes(x=Rating, y=Category)) +
  geom_segment(aes(yend=Category), xend=0, colour="grey50") +
  geom_point(size=1, aes(colour=Type)) +
  scale_colour_brewer(palette="Set1", limits=c("Free", "Paid"), guide=FALSE) +
  theme_bw() +
  theme(panel.grid.major.y = element_blank()) +
  facet_grid(Type ~ ., scales="free_y", space="free_y") +
  ggtitle("Rating per Category between Free and Paid Apps")

#Which category has the most installation?
ggplot(data, aes(x= Category, y= Installs, fill = Type)) +
  geom_bar(position='dodge',stat='identity') +
  coord_flip() +
  ggtitle("Number Of App Installs Based On Category and Type")

#Is there any correleation between app Rating and other attribute?
corrplot(cor(my_data))
#Observation: since there is no correlation between rating and other features
# => we cannot use Regresion Model to predict app Rating
# => Instead, we will try classification models to predit app rating.

#########Basic Decision Tree modeling
summary(data$Rating)
newdata = data
newdata$Rating <- ifelse(newdata$Rating <=4,"Low", "High")

#Remove unnecessary columns
newdata <- newdata[-c(1,10,11,12,13)]
str(newdata)

#Transform character variables to factor variables
newdata$Category <- as.factor(newdata$Category)
newdata$Rating <- as.factor(newdata$Rating)
newdata$Content.Rating <- as.factor(newdata$Content.Rating)
str(newdata)

#Randomly divide the data into a 70/30 split 
set.seed(111)
index <- createDataPartition(newdata$Rating, p = .70, 
                             list = FALSE, 
                             times = 1)

newdata_train <- newdata[index,]
dim(newdata_train)
newdata_test <- newdata[-index,]
dim(newdata_test)

#Apply Decision Tree model
m_C50 <- C5.0(newdata_train[-2], newdata_train$Rating)  
m_C50
summary(m_C50)
#Observation: the model predicts 80.4% correctly on training set

rating_DT_pred <- predict(m_C50, newdata_test)
CrossTable(newdata_test$Rating, rating_DT_pred,
           prop.chisq = FALSE, prop.c = FALSE, prop.r = FALSE,
           dnn = c('actual default', 'predicted default'))
#Observation: the model predicts 75.7% correctly on test set

############ MODEL IMPROVEMENT

########### Tuning model

#Customize the parameter tuning process
#10 cross validation, model selection function is "OneSE"
#evaluation matrix is "Kappa"
#trials is 1,3,5; winnow = TRUE
RNGversion("3.5.2")
set.seed(300)
ctrl <-trainControl(method = "cv", number = 10, selectionFunction = "oneSE")
grid <- expand.grid(model = "tree", trials = c(1,3,5), winnow = TRUE)
m_C50_tuning <- train(Rating~., data = newdata_train, method = "C5.0",
                      metric = "Kappa", trControl = ctrl, tuneGrid = grid)
m_C50_tuning
#Observation: the best Decision tree model with 3 trials

##Test the best model's performance on the test dataset
RNGversion("3.5.2")
set.seed(300)
ctrl <-trainControl(method = "cv", number = 10, selectionFunction = "oneSE")
grid <- expand.grid(model = "tree", trials = 3, winnow = TRUE)
m_C50_trials3 <- train(Rating~., data = newdata_train, method = "C5.0",
                       metric = "Kappa", trControl = ctrl, tuneGrid = grid) 
prediction_DT_trials3 <- predict(m_C50_trials3, newdata_test)
prop.table(table(prediction_DT_trials3, newdata_test$Rating))
#Observation: the accuracy of the model keeps the same at 75.7%

############### Bagging
#Apply Bagging algorithm on the training set

library(ipred)
RNGversion("3.5.2")
set.seed(300)
mybag <- bagging(Rating ~.,data = newdata_train, nbagg = 25)

##Customize the parameter tuning process
#10 cross validation, model selection function is "OneSE"
#evaluation matrix is "Kappa"
RNGversion("3.5.2")
set.seed(300)
ctrl <- trainControl(method = "cv", number = 10, selectionFunction = "oneSE")
mybag2 <- train(Rating ~., data = newdata_train, method = "treebag", trControl = ctrl,
                metric = "Kappa")
mybag2

#Test the best model's performance on the test dataset
p <- predict(mybag2, newdata_test)
round(prop.table(table(p, newdata_test$Rating)),3)
## No improvement

################ Boosting

#Apply booting algorithm on the training set.  

library(adabag)
RNGversion("3.5.2")
set.seed(300)
m_adaboost <- boosting(Rating~., data = newdata_train)

#Test the best model's performance on the training set
p_adaboost <- predict(m_adaboost, newdata_train)
round(prop.table(p_adaboost$confusion),2)
#Observation: there is likely an overfitting on the training dataset

#Another evalutaion method
RNGversion("3.5.2")
set.seed(300)
adaboost_cv <- boosting.cv(Rating ~., data = newdata_train)
round(prop.table(adaboost_cv$confusion),2)

library(vcd)
Kappa(adaboost_cv$confusion) 

#Test the best model's performance on the test dataset
p <- boosting.cv(Rating~., data = newdata_test)
round(prop.table(p$confusion),3)
##Observation: the model predicts 76% correctly on the test set.

############# Random Forest

#Apply random forest algorithm on the training set
RNGversion("3.5.2")
set.seed(300)
rf <- randomForest(Rating ~., data = newdata_train)
rf
Kappa(rf$confusion[1:2,1:2])

#Evaluate model performance
ctrl <- trainControl(method = "repeatedcv", number = 10, repeats = 10,
                     selectionFunction = "best", savePredictions = TRUE,
                     classProbs = TRUE, summaryFunction = twoClassSummary)
grid_rf <- expand.grid(mtry = c(2,3,4))
RNGversion('3.5.2')
set.seed(300)
m_rf <- train(Rating~., data = newdata_train, method = "rf", metric = "ROC",
              trControl = ctrl, tuneGrid = grid_rf)
m_rf
#Observation: using mtry = 4 for the final model

#Test the best model's performance on the test dataset
m_rf_final <- train(Rating~., data = newdata_train, method = "rf", metric = "ROC",
                    trControl = ctrl, tuneGrid = expand.grid(mtry = 4))
p <- predict(m_rf_final, newdata_test)
round(prop.table(table(p, newdata_test$Rating)),3)
##Observation: the model predicts 76.1% correctly on the test set.
##Conlusion: the best model is boosting since it takes shorter processing time than Random Forest model.
