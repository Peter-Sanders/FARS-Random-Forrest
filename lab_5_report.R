d<-read.csv(file='file:///C:/Users/Pete/OneDrive/Documents/Fall 2017/IE 322/HW&LABS/Lab 5/FARS.csv')
#d<-read.csv(file='file:///S:/windows/FARS.csv')
d1=d
d1<-subset(d1, age!="999")
d1<-subset(d1, airbag != "99")
d1<-subset(d1, sex !="9")
d1<-subset(d1, restraint != "99")
d1<-subset(d1, modelyr != "9999")
d1<-subset(d1, airbagAvail != "NA-code")
d1<-subset(d1, D_airbagAvail != "NA-code")
d1<-subset(d1, D_Restraint != "NA-code")
d1<-subset(d1, inimpact!="99")
d1<-subset(d1, airbagDeploy != "NA-code")
d1<-subset(d1, Restraint != "NA-code")
d1<-subset(d1, D_airbagDeploy != "NA-code")
d1<-subset(d1, D_Restraint != "NA-code")
d1<-subset(d1, restraint != "97")
d1<-subset(d1, age != "998")
d1<-subset(d1, inimpact!="98")
m<- names(d1) %in% c("caseid","state","X")
d1<-d1[!m]
D<-d1
cols1<- c(2:6,11)
D[cols1]<-lapply(D[cols1],factor)
D$airbagAvail<-as.factor(as.numeric(D$airbagAvail))
D$airbagDeploy<-as.factor(as.numeric(D$airbagDeploy))
D$Restraint<-as.factor(as.numeric(D$Restraint))
D$D_airbagAvail<-as.factor(as.numeric(D$D_airbagAvail))
D$D_airbagDeploy<-as.factor(as.numeric(D$D_airbagDeploy))
D$D_Restraint<-as.factor(as.numeric(D$D_Restraint))
D<-subset(D, injury!="5")
D<-subset(D, D_injury !="5")
levels(D$injury) <- c("1","1","2","2","4","5")
levels(D$D_injury) <- c("1","1","2","2","4","5")
D$injury<-factor(D$injury)
D$D_injury<-factor(D$D_injury)


library(GoodmanKruskal)
W<-GKtauDataframe(D)
plot(W, diagSize = .6)


#######################################


library(randomForest)
library(caret)
library(mlbench)
library(psych)
library(foreach)
library(doParallel)
library(e1071)
library(class)
library(rpart)
library(C50)
library(ggplot2)
##########################33
nb_cores<-3
cl<-makeCluster(nb_cores)
registerDoParallel(cl)
ptm<-proc.time()

testIndex = createDataPartition(D$injury, p = 0.6, list = FALSE)
training <- D[testIndex, ]
testing <- D[-testIndex, ]

testIndex1 = createDataPartition(D$D_injury, p = 0.6, list = FALSE)
training1 <- D[testIndex1, ]
testing1 <- D[-testIndex1, ]

lowVariance <- describe(D)
plot(lowVariance$sd, ylab = "variance")



# Create model with default paramters
control <- trainControl(method="cv", number=3, search = "grid")
seed <- 555
metric <- "Accuracy"
set.seed(seed)
tunegrid <- expand.grid(.mtry=c(1:15))

#################################################

rf_gridsearch <- train(injury~., data=training, method="rf", metric=metric, tuneGrid=tunegrid, trControl=control)
print(rf_gridsearch)
prdrfgs<-predict(rf_gridsearch, testing[,-3])
P<-as.matrix(table(testing[,3],prdrfgs))
P


rf_gridsearch1 <- train(D_injury~., data=training1, method="rf", metric=metric, tuneGrid=tunegrid, trControl=control)
print(rf_gridsearch1)
prdrfgs1<-predict(rf_gridsearch1, testing1[,-11])
P1<-as.matrix(table(testing1[,11],prdrfgs1))
P1

###################################


rf<-randomForest(injury~., data = training)
prd<-predict(rf, testing[,-3], type = "class")
P2<-as.matrix(table(testing[,3],prd))
P2

rf1<-randomForest(D_injury~.,data = training1)
prd1<-predict(rf1, testing1[,-11], type = "class")
P3<-as.matrix(table(testing1[,11],prd1))
P3

stopCluster(cl)
#####################################################

model<-naiveBayes(injury~.,data = training)
prdnb<-predict(model, testing[,-3],type = "class")
P4<-as.matrix(table(testing[,3],prdnb))
P4

model1<-naiveBayes(D_injury~.,data = training1)
prdnb1<-predict(model1, testing1[,-11],type = "class")
P5<-as.matrix(table(testing1[,11],prdnb1))
P5
######################################################
training_target<-training[,3]
testing_target<-testing[,3]
training_target1<-training1[,11]
testing_target1<-testing1[,11]

knn<-knn(train = training, test = testing, cl = training_target, k =260)
P6<-as.matrix(table(testing_target,knn))
knn1<-knn(train = training1, test = testing1, cl = training_target1, k =260)
P7<-as.matrix(table(testing_target1,knn1))

############################################
knncv <- train(injury~., data=training, method="knn", metric=metric,trControl=control)
prdknncv<-predict(knncv, testing[,-3])
P8<-as.matrix(table(testing[,3],prdknncv))
P8
knncv1 <- train(D_injury~., data=training1, method="knn", metric=metric,trControl=control)
prdknncv1<-predict(knncv1, testing[,-11])
P9<-as.matrix(table(testing[,11],prdknncv1))
P9
##############################################
tree<-rpart(injury~., data = training, method = "class")
prdrpart<-predict(tree, testing[,-3], type = "class")
P10<-as.matrix(table(testing[,3],prdrpart))

tree1<-rpart(D_injury~., data = training1, method = "class")
prdrpart1<-predict(tree1, testing1[,-11], type = "class")
P11<-as.matrix(table(testing1[,11],prdrpart1))
##################################################
ctree<-C5.0(injury~., data = training)
prdctree<-predict(ctree,testing[,-3],type = "class")
P12<-as.matrix(table(testing[,3],prdctree))

ctree1<-C5.0(D_injury~., data = training1)
prdctree1<-predict(ctree1,testing1[,-11],type = "class")
P13<-as.matrix(table(testing1[,11],prdctree1))
#####################################################
#P

np = sum(P) # number of instances
diagp = diag(P) # number of correctly classified instances per class 
accuracyp = sum(diagp) / np 
accuracyp 

#P1

np1 = sum(P1) # number of instances
diagp1 = diag(P1) # number of correctly classified instances per class 
accuracyp1 = sum(diagp1) / np1 
accuracyp1 

#P2

np2 = sum(P2) # number of instances
diagp2 = diag(P2) # number of correctly classified instances per class 
accuracyp2 = sum(diagp2) / np2 
accuracyp2

#P3

np3 = sum(P3) # number of instances
diagp3 = diag(P3) # number of correctly classified instances per class 
accuracyp3 = sum(diagp3) / np3 
accuracyp3 

#P4

np4 = sum(P4) # number of instances
diagp4 = diag(P4) # number of correctly classified instances per class 
accuracyp4 = sum(diagp4) / np4 
accuracyp4 

#P5

np5 = sum(P5) # number of instances
diagp5 = diag(P5) # number of correctly classified instances per class 
accuracyp5 = sum(diagp5) / np5 
accuracyp5 

#P6

np6 = sum(P6) # number of instances
diagp6 = diag(P6) # number of correctly classified instances per class 
accuracyp6 = sum(diagp6) / np6 
accuracyp6 

#P7

np7 = sum(P7) # number of instances
diagp7 = diag(P7) # number of correctly classified instances per class 
accuracyp7 = sum(diagp7) / np7 
accuracyp7

#P8

np8 = sum(P8) # number of instances
diagp8 = diag(P8) # number of correctly classified instances per class 
accuracyp8 = sum(diagp8) / np8 
accuracyp8

#P9

np9 = sum(P9) # number of instances
diagp9 = diag(P9) # number of correctly classified instances per class 
accuracyp9 = sum(diagp9) / np9 
accuracyp9

#P10

np10 = sum(P10) # number of instances
diagp10 = diag(P10) # number of correctly classified instances per class 
accuracyp10 = sum(diagp10) / np10 
accuracyp10

#P11

np11 = sum(P11) # number of instances
diagp11 = diag(P11) # number of correctly classified instances per class 
accuracyp11 = sum(diagp11) / np11
accuracyp11

#P12

np12 = sum(P12) # number of instances
diagp12 = diag(P12) # number of correctly classified instances per class 
accuracyp12 = sum(diagp12) / np12 
accuracyp12

#P13

np13 = sum(P13) # number of instances
diagp13 = diag(P13) # number of correctly classified instances per class 
accuracyp13 = sum(diagp13) / np13 
accuracyp13

ACCrfcv<- cbind(accuracyp,accuracyp1)
ACCrf<- cbind(accuracyp2,accuracyp3)
ACCnb<- cbind(accuracyp4,accuracyp5)
ACCknn<- cbind(accuracyp6,accuracyp7)
ACCknncv<-cbind(accuracyp8,accuracyp9)
ACCrpart<-cbind(accuracyp10,accuracyp11)
ACCc50<-cbind(accuracyp12,accuracyp13)


ACC<- rbind(ACCrfcv,ACCrf,ACCnb,ACCknn,ACCknncv,ACCrpart,ACCc50)
ACC<-as.data.frame(ACC)
names(ACC)<- c("injury accuracy", "D_injury accuracy")
ACC$method<-c("rfcv","rf","nb","knn","knncv","rpart","c5.0")
ACC
####################################################

#Graphing

qplot(injury, data =D, bins = 10, main = "Disbursment of Injuries for Passengers", fill = factor(injury))
qplot(D_injury, data =D, bins = 10, main = "Disbursment of Injuries for Drivers", fill = factor(D_injury))
qplot(age, data = D, bins = 100, main = "Range of ages and sex", fill = factor(sex))
qplot(age, data = D, bins = 100, main = "Range of ages and passenger injury level", fill = factor(injury))
qplot(age, data = D, bins = 100, main = "Range of ages and driver injury level", fill = factor(D_injury))
