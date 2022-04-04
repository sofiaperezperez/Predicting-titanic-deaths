load("titanic_train.Rdata")
data=titanic.train
names(data)

library("ggplot2")
library("ggthemes")
library("factoextra")
library("caTools")
library("fastDummies")
library("class")
library("rpart")
library("rpart.plot")
library("randomForest")
library("e1071")
library("MASS")
library("tictoc")
library("iterators")
library("foreach")
library("parallel")
library("doParallel")


#exploratory data analysis

#choosing variables (plotting them), always comparing them with survived since it is the class we are going to study
#Pclass, we consider it is important
ggplot(data)+aes(x=Survived,fill=Pclass)+geom_bar(position = "fill")
#Sex,it is important
ggplot(data)+aes(x=Survived,fill=Sex)+geom_bar(position = "fill")
#age,it is important, plus it has outliers that could be interesting
ggplot(data)+aes(x=Survived,y=Age)+geom_boxplot()
#SibSp,it is important,as we will show in a ggplot
ggplot(data)+aes(x=Survived,y=SibSp)+geom_boxplot()
#Parch,it is important, as we will show in a ggplot
ggplot(data)+aes(x=Survived,y=Parch)+geom_boxplot()
#Fare,it is not relevant
ggplot(data)+aes(x=Survived,y=Fare)+geom_boxplot()
#we are not going to take into account cabin since there are observations left.
#Embarked, not relevant,your survival does not depend on where you started the trip
ggplot(data)+aes(x=Survived,fill=Embarked)+geom_bar(position = "fill")



#create a family size variable for each passenger.

data$FamilySize = data$SibSp + data$Parch + 1 #We add 1 cause we have to add the passenger itself

ggplot(data[1:668,], aes(x = FamilySize, fill = factor(Survived))) +
  geom_bar(stat='count', position='dodge') +
  scale_x_continuous(breaks=c(1:11)) +
  labs(x = 'Family Size') +
  theme_few()

#prepare the data

data=dummy_cols(data,select_columns = "Sex")

data=data[,-3]
data=data[,-12]
data=data[,-(4:9)]
survive=data[,1]


#to do cross validation, we have to split the data randomly into five groups
set.seed(300)
n=nrow(data)
nfolds=5
folds=cut(1:n,breaks=nfolds,labels=FALSE)
folds=sample(folds,length(folds),replace=FALSE)

#now we start with the differents classifiers

#smv
set.seed(300)
parametersvm=seq(1,100,0.5) #this classifier depends on just one parameter, gamma
result=matrix(0,length(parametersvm),nfolds)

for(i in 1:nfolds){
  tst_set=data[which(folds==i),]
  tst_set1=tst_set[,2:5]
  
  tr_set=data[which(folds!=i),]
  tr_set1=tr_set[,2:5]
  
  tr_class=tr_set[,1]
  tst_class=tst_set[,1]
  
  for (j in 1:length(parametersvm)) {
    model=svm(tr_class~.,data=tr_set1,kernel="radial",gamma=parametersvm[j])
    prediction=predict(model,tst_set1)
    t=table(prediction,tst_class)
    accuracy=sum(diag(t))/sum(t)
    result[j,i]=accuracy
    
  }
}

result
average=apply(result,1,mean)
for (i in 1:length(average)){
  if (average[i]>0.95){  #to avoid the overfitting
    average[i]=0
  }
}

maximum=which.max(average)
finalgamma=parametersvm[maximum]
plot(gamma,average,type="b")

#now we do svm with the correct gamma
set.seed(300)
finalmodel=svm(tr_class~.,data=tr_set,kernel="radial",gamma=finalgamma)
prediction=predict(model,tst_set)
t=table(prediction,tst_class)
accuracy=sum(diag(t))/sum(t)  

#in our exeriment, our gamma=1, and we obtain 74,62% of accuracy
set.seed(300)
finalmodel=svm(tr_class~.,data=tr_set,kernel="radial",gamma=1)
prediction=predict(model,tst_set)
t=table(prediction,tst_class)
accuracy=sum(diag(t))/sum(t) 




#decision trees
set.seed(300)
parameterminsplit=seq(1,1000,5)
parameterminbucket=seq(1,1000,5)  #with decision trees we depend on two parameters, minsplit and mindbucket
result=matrix(0,length(parameterminsplit),2)
resultaccuracy=matrix(0,length(parameterminbucket),1)
tic()
foreach(w=1:length(parameterminsplit), .combine =c)%dopar%{
  foreach(j=1:length(parameterminbucket), .combine =c)%dopar%{
    partialmean=0
    result[j,1]=w
    result[j,2]=j
    foreach(i=1:nfolds, .combine =c)%dopar%{
      tst_set=data[which(folds==i),]
      tst_set1=tst_set[,2:5]
      
      tr_set=data[which(folds!=i),]
      tr_set1=tr_set[,2:5]
      
      tr_class=tr_set[,1]
      tst_class=tst_set[,1]
      
      model=rpart(formula = tr_class~.,data=tr_set1,method = "class" ,control=rpart.control(minsplit=w,minbucket = j)) 
      prediction=predict(model,newdata=tst_set1,type = "class")
      t=table(prediction,tst_class,dnn = c("class","predicted"))
      accuracy=sum(diag(t)/sum(t))
      partialmean=partialmean+accuracy
      resultaccuracy[j,1]=partialmean/nfolds
    }
  }
}


toc()
result
resultaccuracy
maximum=which.max(resultaccuracy)
finalparameterminsplit=result[maximum,1]
finalparameterminbucket=result[maximum,2]
finalparameterminbucket
finalparameterminsplit


#now we do decision trees with the correct parameters
set.seed(300)
model=rpart(formula = tr_set[,1]~.,data=tr_set[,2:5],method = "class" ,control=rpart.control(minsplit=finalparameterminsplit,minbucket = finalparameterminbucket)) 
prediction=predict(model,newdata=tst_set,type = "class")
t=table(prediction,tst_class,dnn = c("class","predicted"))
accuracy=sum(diag(t)/sum(t))
prp(model,type=1,extra=104,shadow.col = "blue",nn=TRUE)

#in our case, we have obtain the best accuracy, 85,82% with minsplit=1 and minbucket=5
set.seed(300)
model=rpart(formula = tr_set[,1]~.,data=tr_set[,2:5],method = "class" ,control=rpart.control(minsplit=1,minbucket =5)) 
prediction=predict(model,newdata=tst_set,type = "class")
t=table(prediction,tst_class,dnn = c("class","predicted"))
accuracy=sum(diag(t)/sum(t))
prp(model,type=1,extra=104,shadow.col = "blue",nn=TRUE)

#knn

set.seed(300)

accuracy = rep(0, 10)
k = 1:10

for(i in 1:nfolds){
  tst_set=data[which(folds==i),]
  tst_set1=tst_set[,2:5]
  
  tr_set=data[which(folds!=i),]
  tr_set1=tr_set[,2:5]
  
  tr_class=tr_set[,1]
  tst_class=tst_set[,1]
  for (x in k) {
    model=knn(train = tr_set1,test=tst_set1, cl=tr_class, k=x)
    accuracy[x] = mean(model == tst_class)
  }
}

plot(k, accuracy, type = 'b')

#Finally, we select the best k for our knn, and we plot it
set.seed(300)
finalModel=knn(train=tr_set1,test = tst_set1, cl=tr_class, k=3)
t=table(finalModel,tst_class)
finalAccuracy=sum(diag(t))/sum(t)
 


#Random Forest

set.seed(300)
rfmodel = randomForest(x=tr_set1, y=tr_class)
rfmodel 

#We plot a graph to check if the default number of trees (500) are good enough for optimal classification.
oob.error.data = data.frame(
  Trees = rep(1:nrow(rfmodel$err.rate), times=3),
  Type=rep(c("OOB","Not survived","Survived"), each=nrow(rfmodel$err.rate)),
  Error=c(rfmodel$err.rate[,1],
          rfmodel$err.rate[,2],
          rfmodel$err.rate[,3]))
ggplot(data=oob.error.data,aes(x=Trees,y=Error))+geom_line(aes(color=Type))

#If we added more trees, would the error rate go down further?

set.seed(300)
rfmodel_modified = randomForest(x=tr_set1, y=tr_class, ntree = 1000)
rfmodel_modified 

oob.error.data2 = data.frame(
  Trees = rep(1:nrow(rfmodel_modified$err.rate), times=3),
  Type=rep(c("OOB","Not survived","Survived"), each=nrow(rfmodel_modified$err.rate)),
  Error=c(rfmodel_modified$err.rate[,1],
          rfmodel_modified$err.rate[,2],
          rfmodel_modified$err.rate[,3]))
ggplot(data=oob.error.data2,aes(x=Trees,y=Error))+geom_line(aes(color=Type))

#Considering the optimal number of variables

set.seed(300)
oob.values = vector(length=5)
for (i in 1:5){
  temporary.model = randomForest(x=tr_set1, y=tr_class, mtry = i, ntree = 1000)
  oob.values[i] = temporary.model$err.rate[nrow(temporary.model$err.rate),1]
}
oob.values

#Now that we have chosen our parameters we can create a model for our random forest.

set.seed(300)
for(i in 1:nfolds){
  tst_set=data[which(folds==i),]
  tst_set1=tst_set[,2:5]
  
  tr_set=data[which(folds!=i),]
  tr_set1=tr_set[,2:5]
  
  tr_class=tr_set[,1]
  tst_class=tst_set[,1]
  
  rfmodel_final = randomForest(x=tr_set1, y=tr_class, ntree = 1000)
  predValid = predict(rfmodel_final,tst_set1)
  conf_matrix=table(predValid,tst_class) 
  meanrf=mean(predValid == tst_class)
  
}

meanrf
conf_matrix


