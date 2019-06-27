import pandas as pd
import numpy as np
import sklearn
import matplotlib as plt
from sklearn import ensemble
from sklearn.metrics import roc_auc_score
test_data = r"C:\Users\wtclark\Desktop\DataScience\Kaggle\Titanic Survival Data\test.csv"
test = pd.read_csv(test_data)
train_data = r"C:\Users\wtclark\Desktop\DataScience\Kaggle\Titanic Survival Data\train.csv"
train = pd.read_csv(train_data)

train['Sex'].loc[train['Sex']=='male'] = 0
train['Sex'].loc[train['Sex']=='female'] = 1
test['Sex'].loc[test['Sex']=='male'] = 0
test['Sex'].loc[test['Sex']=='female'] = 1

test['Embarked'] = test['Embarked'].fillna(test['Embarked']=='S')
train['Embarked'] = train['Embarked'].fillna(train['Embarked']=='S')
train['Embarked'].loc[train['Embarked']=='S'] = 0
train['Embarked'].loc[train['Embarked']=='C'] = 1
train['Embarked'].loc[train['Embarked']=='Q'] = 2
test['Embarked'].loc[test['Embarked']=='S'] = 0
test['Embarked'].loc[test['Embarked']=='C'] = 1
test['Embarked'].loc[test['Embarked']=='Q'] = 2

index = np.where(train['Fare'] == max(train['Fare']))
print(index)
test['Fare'] = test['Fare'].fillna(test['Fare'].median())

missing = np.where(train['Age'].isnull() == True)
#print(missing)
print("Number of Passengers with Missing Age:")
print(len(missing[0]))

#This section thanks to Analytics Vidhya
#Training Data
def name_extract(word):
    return word.split(',')[1].split('.')[0].strip()

df1 = pd.DataFrame({'Salutation':train['Name'].apply(name_extract)})
train = pd.merge(train, df1, left_index = True, right_index = True)
temp1 = train.groupby('Salutation').PassengerId.count()

def group_salutation(old_salutation):
    if old_salutation in ('Mr', 'Capt', 'Col', 'Don', 'Dr', 'Major', 'Rev', 
                          'Sir', 'Jonkheer'):
        return('Mr')
    elif old_salutation in ('Mrs', 'Mme', 'the Countess'):
        return('Mrs')
    elif old_salutation =='Master':
        return('Master')
    elif old_salutation in ('Miss', 'Mlle', 'Lady'):
        return('Miss')

df2 = pd.DataFrame({'New_Salutation': train['Salutation'].apply
(group_salutation)})

train = pd.merge(train, df2, left_index = True, right_index = True)
temp2 = df2.groupby('New_Salutation').count()

train.boxplot(column = 'Age', by = 'New_Salutation')

table1 = train.pivot_table(values='Age', index=['New_Salutation'], 
                           columns=['Pclass', 'Sex'], aggfunc=np.median)
# Define function to return value of this pivot_table
def fage(x):
    return table1[x['Pclass']][x['Sex']][x['New_Salutation']]

# Replace missing values
train['Age'].fillna(train[train['Age'].isnull()].apply(fage, axis=1), 
                    inplace=True)

train.hist(column = 'Age', figsize = (9, 6), bins = 20)

#Test Data
def name_extract2(word):
    return word.split(',')[1].split('.')[0].strip()

df3 = pd.DataFrame({'Salutation':test['Name'].apply(name_extract2)})
test = pd.merge(test, df3, left_index = True, right_index = True)
temp3 = test.groupby('Salutation').PassengerId.count()

def group_salutation2(old_salutation):
    if old_salutation in ('Mr', 'Col', 'Dr', 'Major', 'Rev'):
        return('Mr')
    elif old_salutation in ('Mrs', 'Dona'):
        return('Mrs')
    elif old_salutation =='Master':
        return('Master')
    elif old_salutation in ('Miss', 'Ms'):
        return('Miss')

df4 = pd.DataFrame({'New_Salutation': test['Salutation'].apply
(group_salutation2)})

test = pd.merge(test, df4, left_index = True, right_index = True)
temp4 = df4.groupby('New_Salutation').count()
#test.boxplot(column = 'Age', by = 'New_Salutation')

table2 = test.pivot_table(values='Age', index=['New_Salutation'],
columns=['Pclass', 'Sex'], aggfunc=np.median)

#Define function to return value of this pivot_table
def fage_2(x):
    return table2[x['Pclass']][x['Sex']][x['New_Salutation']]
#Replace missing values
test['Age'].fillna(test[test['Age'].isnull()].apply(fage_2, axis=1), 
                   inplace=True)

test['Age'] = test['Age'].fillna(test['Age'].median())

#Random Forest #3
target = train['Survived'].values
features_forest = train[['Pclass', 'Age', 'Sex', 'Fare','SibSp', 'Parch'
]].values
forest = sklearn.ensemble.RandomForestClassifier(max_depth = 20,
min_samples_split = 2, n_estimators = 100, random_state = 1, oob_score = True)
my_forest = forest.fit(features_forest, target)

test_features = test[['Pclass','Age', 'Sex', 'Fare', 'SibSp', 'Parch']].values
pred_forest = my_forest.predict(test_features)

print(len(pred_forest))
print(my_forest.score(features_forest, target))
#print ("AUC - ROC: ", roc_auc_score(target, forest.oob_prediction))

df4 = pd.DataFrame(pred_forest, columns = ['Survived'])
df5 = pd.DataFrame(test['PassengerId'])
random_forest4 = pd.concat([df5, df4], axis = 1)

random_forest4.to_csv(r"C:\Users\wtclark\Desktop\random_forest4.csv", 
                      index = False)
