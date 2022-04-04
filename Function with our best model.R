#now to test our classifier, random forest, with your secret data, use this function

my_model=function(test){
  load("my_fitted_model.RData")
  library("randomForest")
  library("fastDummies")
  yourdata=test
  yourdata$FamilySize = yourdata$SibSp + yourdata$Parch + 1 #We add 1 cause we have to add the passenger itself
  yourdata=dummy_cols(yourdata,select_columns = "Sex")
  yourdata=yourdata[,-3]
  yourdata=yourdata[,-12]
  yourdata=yourdata[,-(4:9)]
  yoursurvive=yourdata[,1]
  predValid = predict(mymodel_final,yourdata[,2:5])
  predValid
  passengers = c(1:nrow(yourdata))
  passengers
  solution = data.frame(Passenger = passengers, Survived = predValid)
  return(solution)
}

test = #Enter your test data
print_prediction = my_model(test)
print_prediction

