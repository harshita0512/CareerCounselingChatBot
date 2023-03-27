from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
import numpy as np
import pandas as pd
import pickle

data = pd.read_csv("career_pred.csv")

X = data[['Programming', 'ComputerNetworks', 'logical_quotient', 'coding_skills', 'public_speaking', 'Self_learn', 'Certifications', 'Workshops', 'Memory', 'InterestedSubjects', 'InterestedCareerarea', 'TypeofCompany', 'jobRole', 'work_teams']]

y = data['SuggestedJobRole']

enc = OneHotEncoder(handle_unknown='ignore')
enc.fit(X)
pickle.dump(enc, open('enc.pkl', 'wb'))
X = enc.fit_transform(X).toarray()


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

mlp = MLPClassifier(hidden_layer_sizes=(20,), max_iter=1000, alpha=1e-4,
                    solver='sgd', verbose=10, tol=1e-4, random_state=1,
                    learning_rate_init=.1)


mlp.fit(X_train, y_train)
print("Training set score: %f" % mlp.score(X_train, y_train))

input_features = pd.DataFrame(
    {'Programming': ['78'], 'ComputerNetworks': ['60'], 'logical_quotient': ['7'], 'coding_skills': ['9'],
         'public_speaking': ['6'], 'Self_learn': ['no'],'Certifications': ['r programming'], 'Workshops': ['cloud computing']
         ,'Memory': ['excellent'] , 'InterestedSubjects': ['parallel computing'],'InterestedCareerarea':['developer'],'TypeofCompany': ['BPA']
         ,'jobRole': ['Technical'],'work_teams': ['yes']})


input_features = enc.transform(input_features).toarray()

predicted_output = mlp.predict(input_features)

print("Predicted output:", predicted_output)

pickle.dump(mlp, open('model.pkl', 'wb'))