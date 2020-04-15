## CASE STUDY BGL GROUP: COMPARETHEMARKET.COM
## AUTHOR: IOANNIS MESIONIS
## DATE: 17/12/2018
## DATA PROVIDED BY BGL
## THE PURPOSE OF THE CASE STUDY IS TO EXPLORE HOW USER DATA CAN INFORM TARGETED PRODUCT AND CONTENT RECOMMENDATION

## REQUIRED PACKAGES
library(ggplot2)
library(dplyr)
library(ROCR)
library(caret)
library(randomForest)
library(glmnet)
library(gridExtra)

##########################################################################################
############################### EXPLANATORY ANALYSIS #####################################
##########################################################################################

## READ THE DATA
data <- read.csv(file.choose())

## EXPLORE HOW THE DATA LOOKS
dim(data)  ## 850.009   12
head(data)
tail(data)

## EXPLORE THE ATTRIBUTES OF THE DATA
colnames(data)

## CHECK FOR DUPLICATED ENTRIES
sum(duplicated(data))    ## 0 DUPLICATED
n_distinct(data$UserID)  ## 100.000 UNIQUE CUSTOMER IDS

## CHECK IF VARIABLES ARE IN THE CORRECT FORM 
str(data)

## TAKE A GENERAL INSIGHT OF THE DATA
summary(data)
levels(data$UserSegment)   ##   "A"        "B"        "C"      "NULL"
levels(data$Recency)       ## "Active"   "Dormant"  "Inactive" "NULL"

## DISTRIBUTION OF THE INTEREST OF THE CUSTOMER FOR A PRODUCT
table(data$PriorEvent)   ## PRIOR: NO INTEREST(0): 90.822      INTEREST(1): 9.178
table(data$Event)        ## POSTERIOR: NO INTEREST(0): 70.000      INTEREST(1): 30.000

## CHECK THE SUMMARY OF THE AGE
summary(data$Age)

## DISTRIBUTION OF SEGMENTS
Segment <- as.data.frame(table(data$UserSegment))
Segment <- Segment[Segment$Var1 != "NULL",]       ## REMOVE THE NULL OBSERVATIONS
ggplot(data = Segment, aes(x = Var1, y = Freq)) +
  geom_bar(stat = "identity", fill = "#F8766D") +
  geom_text(y = Segment[,2] - 2000,
            aes(label = paste0(round((Segment[,2]/sum(Segment[,2]))*100, digits = 1), "%" ))
  ) +
  labs(x = "Segments", y = "Frequency") +
  ggtitle("User Segments Distribution")

## DISTRIBUTION OF RECENCY
RecencyDis <- as.data.frame(table(data$Recency))
RecencyDis <- RecencyDis[RecencyDis$Var1 != "NULL", ]    ## REMOVE THE NULL OBSERVATIONS
ggplot(data = RecencyDis, aes(x = Var1, y = Freq)) +
  geom_bar(stat = "identity", position = "dodge", fill = "#F8766D") +
  geom_text(y = RecencyDis[,2] - 2000,
            aes(label = paste0(round((RecencyDis[,2]/sum(RecencyDis[,2]))*100, digits = 1), "%" ))
            ) +
  labs(x = "Recency", y = "Frequency") +
  ggtitle("Recency Distribution")

## DISTRIBUTION OF THE AGES ACCORDING TO RECENCY
## VISUALISE THE CORRESPONDING DATA
AgeActivity <- as.data.frame(data %>%
                               group_by(Recency, Age)
                             %>% summarise(counts = n())
                             %>% filter(Recency != "NULL")
                             %>% arrange(desc(counts))
)

p1 <- ggplot(data = AgeActivity, aes(x = factor(AgeActivity[,2]), y = AgeActivity[,3])) +
  geom_bar(stat = "identity", aes(fill = AgeActivity[,1])) +
  labs(x="Ages", y="Frequency", fill="Recency") +
  ggtitle("Distribution of Ages by Recency")

p2 <- ggplot(data = AgeActivity, aes(x = AgeActivity[,2], y = AgeActivity[,3], fill = AgeActivity[,1])) +
  geom_bar(stat = "identity") +
  facet_grid(AgeActivity[,1]) +
  labs(x="Ages", y="Frequency", fill="Recency")

grid.arrange(p1, p2, ncol = 1, widths = 2)

## DISTRIBUTION OF THE RECENCY ACCORDING TO USER SEGMENT
## VISUALISE THE CORRESPONDING DATA
ActivitySegment <- as.data.frame(data %>% 
                                   group_by(Recency, UserSegment) 
                                 %>% summarise(counts = n())
                                 %>% filter(Recency != "NULL")
                                 %>% arrange(desc(counts))
                                 %>% mutate(percent = round((counts/sum(counts))*100, digits = 2))
)

p1.2 <- ggplot(data = ActivitySegment, aes(x = ActivitySegment[,2], y = ActivitySegment[,3], fill = ActivitySegment[,1])) +
  geom_bar(stat = "identity", position = "dodge") +
  geom_text(size = 3, stat = "identity", position = position_dodge(.9), aes(y = ActivitySegment[,4]+ 800, label = paste0(ActivitySegment[,4], "%") )) +
  facet_grid(ActivitySegment[,1]) +
  labs(x = "Segment", y = "Frequency", fill = "Recency") +
  ggtitle("Segments by Recency")

## DISTRIBUTION OF THE USER SEGMENT ACCORDING TO RECENCY
ActivitySegment2 <- as.data.frame(data %>% 
                                    group_by(UserSegment, Recency) 
                                  %>% summarise(counts = n())
                                  %>% filter(Recency != "NULL")
                                  %>% arrange(desc(counts))
                                  %>% mutate(percent = round((counts/sum(counts))*100, digits = 2))
)

p2.2 <- ggplot(data = ActivitySegment2, aes(x = ActivitySegment2[,2], y = ActivitySegment2[,3], fill = ActivitySegment2[,1])) +
  geom_bar(stat = "identity", position = "dodge") +
  geom_text(size = 3, stat = "identity", position = position_dodge(.9), aes(y = ActivitySegment2[,4] + 800, label = paste0(ActivitySegment2[,4], "%") )) +
  facet_grid(ActivitySegment2[,1]) +
  labs(x = "Segment", y = "Frequency", fill = "Segments") +
  ggtitle("Recency by Segments")

grid.arrange(p1.2, p2.2, nrow = 2)

# ## DISTRIBUTION OF THE AGES IN THE DATASET
# ## VISUALISE THE COUNTS OF THE AGES
# AgeCount <- as.data.frame(data %>%
#                 group_by(Age)
#                 %>% summarise(counts = n())
#                 %>% arrange(desc(counts))
# )  
# 
# ggplot(data = AgeCount, aes(factor(AgeCount[,1]), AgeCount[,2])) +
#   geom_bar(stat = "identity", fill = "#F8766D") +
#   labs(x = "Ages", y = "Frequency") +
#   ggtitle("Distribution of the Ages")

## REMOVE UNNECESSARY DATA
rm(AgeActivity, AgeCount, ActivitySegment, ActivitySegment2, RecencyDis, Segment, p1, p2, p1.2, p2.2)

##########################################################################################
#################################### DATA MODELLING ######################################
##########################################################################################

## SET NUMBER OF REPETITIONS
n <- 10

## EMPTY CONFUSION MATRIX TO BE STORED FROM VARIOUS REPETITIONS 
confusion <- data.frame(r1 = rep(0,n), r2 = rep(0,n), r3 = rep(0,n), r4 = rep(0,n))

## AUC VALUE EMPTY VECTOR TO STORE AUC VALUES FROM VARIOUS REPETITIONS
auc.total <- vector(mode = "numeric", length = n)

## RUN THE MODEL 
for(i in 1:n){
## CREATE BALANCED DATASET BY UNDERSAMPLING THE MAJORITY CLASS
event0 <- data[data$Event == 0,]     
event0 <- sample_n(event0, 30000, replace = FALSE)  ## FROM MAJORITY CLASS, TAKE A 30.000 SAMPLE 

event1 <- data[data$Event == 1,]    ## FROM MINORITY CLASS, TAKE ALL THE 30.000 INSTANCES
balancedData <- rbind(event0, event1)   ## CREATE THE BALANCED DATA

## TAKE THE DEISGN MATRIX
designMatrix <- sparse.model.matrix(Event ~ .  , data = balancedData[,-c(1)])[,-1]  ## REMOVE THE CUSTOMER ID AND THE INTERCEPT
response <- balancedData$Event  ## STORE THE RESPONSE VARIABLE

## SPLIT IN TRAIN AND TEST SET
ind <- sample(2, nrow(designMatrix), replace = TRUE, prob = c(0.8, 0.2))

trainData <- designMatrix[ind == 1,] 
responseTrain <- response[ind == 1]
dim(trainData) 

testData <- designMatrix[ind == 2,]
responseTest <- response[ind == 2]
dim(testData)

## CHECK FOR PROPORTION
# FOR DESIGN TRAIN
table(responseTrain)

# FOR DESIGN TEST
table(responseTest)

## LASSO CROSS VALIDATION TO FIND THE BEST LAMBDA
cv <- cv.glmnet(trainData, responseTrain, 
                nfolds = 10, 
                type.measure = "class",
                family = "binomial", 
                alpha = 1)
plot(cv)

## PREDICT ON TEST.SET (UNSEEN DATA) USING THE BEST LAMBDA FROM CROSS VALIDATION
predictions <- predict(cv, newx = testData, type = "response", s = "lambda.min")
pred.roc <- predictions     ## STORE THE PROBABILTIES FOR THE ROC CURVE TO BE USED LATER
cut_off <- 0.45   ## SELECT THE CUT-OFF TO CLASSIFY THE RESULT 
predictions <- if_else(predictions >= cut_off, 1, 0)

## STORE AUC VALUE
predictions.bal <- prediction(pred.roc, responseTest)
auc <- performance(predictions.bal, "auc")
auc.total[i] <- unlist(slot(auc, "y.values"))

## STORE THE CONFUSION MATRIX FOR EVERY REPETITION
confusion[i,] <- as.vector(table(predictions, responseTest))
confusionMatrix(as.factor(predictions),as.factor(responseTest))
}

## STORE THE LASSO LOGISTIC REGRESSION COEFFICIENTS
tmp_coef <- coef(cv, s = "lambda.min")   # COEF OF LAMBDA MIN
coef <- data.frame(name = tmp_coef@Dimnames[[1]][tmp_coef@i + 1], coefficient = tmp_coef@x)
b_coef <- coef[order(coef[,2]),]

## VISUALISE THE COEFFICIENTS
ggplot(b_coef, aes(y = b_coef[,2], x = b_coef[,1], color = if_else(sign(b_coef[,2]) > 0, "Positive", "Negative"))) +
  geom_point(stat = "identity") +
  geom_segment(aes(y = 0, x = b_coef[,1], yend = b_coef[,2], xend = b_coef[,1])) +
  coord_flip() +
  theme(axis.title.x = element_blank(), axis.title.y = element_blank(), legend.title = element_blank()) +
  ggtitle("Coefficients Predictive Contribution")

## CREATE A FUNCTION TO CALCULATE THE AVERAGE ACCURACY, SPECIFICITY, SENSITIVITY OF THE VARIOUS REPETITIONS
res <- function(x) {
  x <- matrix(as.numeric(x), ncol = 2)
  accur <- (x[1,1]+x[2,2])/sum(x)
  sens <- x[2,2]/sum(x[,2])
  spec <- x[1,1]/sum(x[,1])
  out <- list(Accuracy = accur, Specificity = spec, Sensitivity = sens)
  out
}
total <-  apply(confusion, 1, FUN = res)

## OBTAIN THE AVERAGE ACCURACY, SPECIFICITY, SENSITIVITY AND STANDARD DEVIATION OF THESE
avg.spec <-0
avg.acc <-0
avg.sens <- 0

spec.total <- vector(mode = "numeric", length = n)
sens.total <- vector(mode = "numeric", length = n)
acc.total <- vector(mode = "numeric", length = n)
for(i in 1:n){
  
  avg.acc = as.numeric(total[[i]][1]) + avg.acc
  acc.total[i] <- as.numeric(total[[i]][1])
  
  avg.sens = as.numeric(total[[i]][3]) + avg.sens
  sens.total[i] <- as.numeric(total[[i]][3])
  
  avg.spec = as.numeric(total[[i]][2]) + avg.spec  
  spec.total[i] <- as.numeric(total[[i]][2])
  
}

## MODEL VALIDATION RESULTS
## AVERAGE AVERAGE ACCURACY, SPECIFICITY, SENSITIVITY
print(c("The accuracy of the model for", n, "repetitions is:", round(avg.acc/n, digits = 2)))
print(c("The sensitivity of the model for", n, "repetitions is:", round(avg.sens/n, digits = 2)))
print(c("The specificity of the model for", n, "repetitions is:", round(avg.spec/n, digits = 2)))

## STANDARD DEVIATIONS OF ACCURACY, SPECIFICITY, SENSITIVITY
print(c("The standard deviation of the accuracy for", n, "repetitions is:", round(sd(acc.total),3)))
print(c("The standard deviation of the accuracy for", n, "repetitions is:", round(sd(sens.total),3)))
print(c("The standard deviation of the accuracy for", n, "repetitions is:", round(sd(spec.total),3)))

## ROC CURVE
predictions.bal <- prediction(pred.roc, responseTest)
roc.lasso <- performance(predictions.bal, measure = "tpr", x.measure = "fpr")

## AUC
mean.auc <- round(mean(auc.total),3)   ## AVEGRAGE AUC
sd.auc <- round(sd(auc.total),3)       ## STANDARD DEVIATION AUC

## ACCURACY CUT-OFF
perf.bal.ac <- performance(predictions.bal, "acc")
plot(perf.bal.ac)

## PLOT ROC CURVE AND AUC VALUE TO FURTHER VALIDATE THE MODEL
plot(roc.lasso, colorize = TRUE, ylab = "Sensitivity", xlab = "1 - Specificity", main = "ROC Curve - Logistic Regression with L1")
lines(c(0,1), c(0,1), col = "black", lty = 2)
legend(.6, .35, mean.auc, title = "AUC", cex = .8)
legend(.8, .35, sd.auc, title = "+- SD", cex = .8)

##########################################################################################
########################################## EXTRAS ########################################
##########################################################################################

## SET NUMBER OF REPETITIONS
n <- 10
## EMPTY CONFUSION MATRIX TO BE STORED FROM VARIOUS REPETITIONS 
confusionRF <- data.frame(r1 = rep(0,n), r2 = rep(0,n), r3 = rep(0,n), r4 = rep(0,n))

## AUC VALUE EMPTY VECTOR TO STORE AUC VALUES FROM VARIOUS REPETITIONS
auc.totalRF <- vector(mode = "numeric", length = n)
OOB <- vector(mode = "numeric", length = n)

## RUN THE MODEL 
for(i in 1:n){
  ## CREATE BALANCED DATASET BY UNDERSAMPLING THE MAJORITY CLASS
  event0 <- data[data$Event == 0,]     
  event0 <- sample_n(event0, 30000, replace = FALSE)  ## FROM MAJORITY CLASS, TAKE A 30.000 SAMPLE 
  
  event1 <- data[data$Event == 1,]    ## FROM MINORITY CLASS, TAKE ALL THE 30.000 INSTANCES
  balancedDataRF <- rbind(event0, event1)   ## CREATE THE BALANCED DATA
  
  ## TAKE THE DEISGN MATRIX
  designMatrixRF <- sparse.model.matrix(Event ~ .  , data = balancedDataRF[,-c(1)])[,-1]  ## REMOVE THE CUSTOMER ID AND THE INTERCEPT
  responseRF <- balancedDataRF$Event  ## STORE THE RESPONSE VARIABLE
  
  ## SPLIT IN TRAIN AND TEST SET
  ind <- sample(2, nrow(designMatrixRF), replace = TRUE, prob = c(0.8, 0.2))
  
  trainDataRF <- designMatrixRF[ind == 1,] 
  responseTrainRF <- responseRF[ind == 1]
  dim(trainDataRF) 
  
  testDataRF <- designMatrixRF[ind == 2,]
  responseTestRF <- responseRF[ind == 2]
  dim(testDataRF)
  
  ## CHECK FOR PROPORTION
  # FOR DESIGN TRAIN
  table(responseTrainRF)
  
  # FOR DESIGN TEST
  table(responseTestRF)
  
  ## RANDOM FORREST
  rf <- randomForest(x = as.matrix(trainDataRF), y = as.factor(responseTrainRF))
  rf
  
  OOB.error <- rf$err.rate[,1]
  OOB[i] <- OOB.error[length(OOB.error)]
  
  # tune(randomForest, train.x = as.matrix(train.data.rf), train.y = as.factor(response.x.rf))
  
  ## AUC VALUE
  prediction.rf <- as.vector(rf$votes[,2])
  pred.rf <- prediction(prediction.rf, ifelse(responseTrainRF == "1", 1, 0))
  auc.rf <- performance(pred.rf, "auc")
  auc.rf <- auc.rf@y.values[[1]]
  auc.totalRF[i] <- auc.rf
  
  # PREDICT ON UNSEEN DATA
  rf_pred <- predict(rf, testDataRF, cutoff = c(0.55, 0.45))
  
  # RESULTS
  confusionRF[i,] <- as.vector(table(as.factor(rf_pred), as.factor(responseTestRF)))
  
}

# VISUALIZATIONS
# plot(rforest)
varImpPlot(rf, sort = TRUE, main = "Variable Importance")

## CREATE A FUNCTION TO CALCULATE THE AVERAGE ACCURACY, SPECIFICITY, SENSITIVITY OF THE VARIOUS REPETITIONS
res <- function(x) {
  x <- matrix(as.numeric(x), ncol = 2)
  accur <- (x[1,1]+x[2,2])/sum(x)
  sens <- x[2,2]/sum(x[,2])
  spec <- x[1,1]/sum(x[,1])
  out <- list(Accuracy = accur, Specificity = spec, Sensitivity = sens)
  out
}

## OBTAIN RESULTS
total.bal <-  apply(confusionRF, 1, FUN = res)

avg.spec.bal <-0
avg.acc.bal <-0
avg.sens.bal <- 0

acc.rf.total <- vector(mode = "numeric", length = n)
sens.rf.total <- vector(mode = "numeric", length = n)
spec.rf.total <- vector(mode = "numeric", length = n)

for(i in 1:n){
  avg.spec.bal = as.numeric(total.bal[[i]][2]) + avg.spec.bal
  spec.rf.total[i] <- as.numeric(total.bal[[i]][2])
  
  avg.acc.bal = as.numeric(total.bal[[i]][1]) + avg.acc.bal
  acc.rf.total[i] <- as.numeric(total.bal[[i]][1])
  
  avg.sens.bal = as.numeric(total.bal[[i]][3]) + avg.sens.bal
  sens.rf.total[i] <- as.numeric(total.bal[[i]][3])
}
## AVERAGE OOB ERROR
cat("The average Out Of Bag error for random forest is:", mean(OOB))

## AVERAGES
cat("The average accuracy of", n, "repetitions is:", round(avg.acc.bal/n, digits = 3))
cat("The average sensitivity of", n, "repetitions is:", round(avg.sens.bal/n, digits = 3))
cat("The average specificity of", n, "repetitions is:", round(avg.spec.bal/n, digits = 3))

## STANDARD DEVIATIONS
round(sd(acc.rf.total),3)
round(sd(sens.rf.total),3)
round(sd(spec.rf.total),3)

## ROC CURVE
prediction.rf <- as.vector(rf$votes[,2])
pred.rf <- prediction(prediction.rf, ifelse(responseTrainRF == "1", 1, 0)) 
roc.rf <- performance(pred.rf, "tpr", "fpr")

## AUC
auc.rf.avg <- round(mean(auc.totalRF),3)
auc.rf.sd <- round(sd(auc.totalRF),3)

plot(roc.rf, colorize = TRUE, ylab = "Sensitivity", xlab = "1 - Specificity", main = "ROC Curve - Random Forest")
lines(c(0,1), c(0,1), col = "black", lty = 2)
legend(.6, .35, auc.rf.avg, title = "AUC", cex = .8)
legend(.8, .35, auc.rf.sd, title = "+- SD", cex = .8)
