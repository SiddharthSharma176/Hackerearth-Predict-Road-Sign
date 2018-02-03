import pandas as pd
from sklearn.ensemble import RandomForestClassifier

train = pd.read_csv("train.csv")
test = pd.read_csv("test.csv")

#encode as integer
mapping = {'Front':0, 'Right':1, 'Left':2, 'Rear':3}
train = train.replace({'DetectedCamera':mapping})
test = test.replace({'DetectedCamera':mapping})

#renaming column
train.rename(columns = {'SignFacing (Target)': 'Target'}, inplace=True)

#encode Target Variable based on sample submission file
mapping = {'Front':0, 'Left':1, 'Rear':2, 'Right':3}
train = train.replace({'Target':mapping})

#target variable
y_train = train['Target']
test_id = test['Id']

#drop columns
train.drop(['Target','Id', 'SignWidth', 'SignHeight'], inplace=True, axis=1)
test.drop(['Id', 'SignWidth', 'SignHeight'],inplace=True,axis=1)

#train model
clf = RandomForestClassifier(n_estimators=500,max_features=3,min_samples_split=5,oob_score=True)
clf.fit(train, y_train)

#predict on test data
pred = clf.predict_proba(test)

#write submission file and submit
columns = ['Front','Left','Rear','Right']
sub = pd.DataFrame(data=pred, columns=columns)
sub['Id'] = test_id
sub = sub[['Id','Front','Left','Rear','Right']]
sub.to_csv("sub_rf2.csv", index=False) #99.74868
