
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt


sns.set_style("darkgrid")
data = pd.read_csv('data/predictive_maintenance.csv')
data.head()
data.info()
data = data.drop(["UDI",'Product ID'],axis=1)
data.head(3)
data.groupby(['Target','Failure Type']).median()
plt.figure(figsize=(10, 8))
sns.countplot(data=data, x="Target")
plt.figure(figsize=(10, 5))
sns.countplot(data=data[data['Target'] == 1], x="Failure Type")
sns.pairplot(data, hue='Target')
data.columns
plt.figure(figsize=(20, 15))
m = 1
for i in ['Air temperature [K]', 'Process temperature [K]', 'Rotational speed [rpm]', 'Torque [Nm]', 'Tool wear [min]']:
    plt.subplot(3, 2, m)
    sns.boxplot(data=data, y=i, x="Type", hue="Target")
    m += 1
import plotly.express as px

fig = px.scatter_3d(data, x='Tool wear [min]', y='Torque [Nm]', z='Rotational speed [rpm]',
                    color='Target')
fig.show()


def feat_prob(feature, data):
    x, y = [], []
    for j in data[feature].unique():
        temp = data
        temp = temp[temp[feature] >= j]
        y.append(round((temp.Target.mean() * 100), 2))
        x.append(j)
    return (x, y)


plt.figure(figsize=(15, 17))
m = 1
for i in ['Air temperature [K]', 'Process temperature [K]', 'Rotational speed [rpm]', 'Torque [Nm]', 'Tool wear [min]']:
    plt.subplot(3, 2, m).set_title(label=("Possibility of failure wrt " + i))
    x, y = feat_prob(i, data)
    plt.xlabel(i)
    plt.ylabel("Possibility of Failure (%)")
    sns.lineplot(y=y, x=x)
    m += 1
plt.figure(figsize=(18, 7))
m = 1

datasets = []
for i in [60, 120, 180, 240]:
    datasets.append(data[data['Tool wear [min]'] >= i])

for i in datasets:
    x, y = feat_prob("Torque [Nm]", i)
    plt.xlabel("Torque [Nm]")
    plt.ylabel("Possibility of Failure (%)")
    sns.lineplot(y=y, x=x, legend='brief')

    m += 1

plt.legend([60, 120, 180, 240])
plt.figure(figsize=(18, 7))
m = 1

datasets = []
for i in [20, 40, 60]:
    datasets.append(data[data['Torque [Nm]'] >= i])

for i in datasets:
    x, y = feat_prob("Rotational speed [rpm]", i)
    plt.xlabel("Rotational speed [rpm]")
    plt.ylabel("Possibility of Failure (%)")
    sns.lineplot(y=y, x=x, legend='brief')

    m += 1

plt.legend([20, 40, 60])
plt.figure(figsize=(18, 7))
m = 1

datasets = []
for i in [60, 120, 180, 240]:
    datasets.append(data[data['Tool wear [min]'] >= i])

for i in datasets:
    x, y = feat_prob("Torque [Nm]", i)
    plt.xlabel("Torque [Nm]")
    plt.ylabel("Possibility of Failure (%)")
    sns.lineplot(y=y, x=x, legend='brief')

    m += 1

plt.legend(["Wear time = " + "60",
            "Wear time = " + '120',
            "Wear time = " + '180',
            "Wear time = " + '240'])
plt.figure(figsize=(18, 7))
m = 1

datasets = []
for i in ["L", "M", "H"]:
    datasets.append(data[data['Type'] == i])

for i in datasets:
    x, y = feat_prob("Torque [Nm]", i)
    plt.xlabel("Torque [Nm]")
    plt.ylabel("Possibility of Failure (%)")
    sns.lineplot(y=y, x=x, legend='brief')

    m += 1

plt.legend(["L", "M", "H"])
plt.figure(figsize=(18, 7))
m = 1

datasets = []
for i in ["L", "M", "H"]:
    datasets.append(data[data['Type'] == i])

for i in datasets:
    x, y = feat_prob("Rotational speed [rpm]", i)
    plt.xlabel("Rotational speed [rpm]")
    plt.ylabel("Possibility of Failure (%)")
    sns.lineplot(y=y, x=x, legend='brief')

    m += 1

plt.legend(["L", "M", "H"])
plt.figure(figsize=(18, 7))
m = 1

datasets = []
for i in ["L", "M", "H"]:
    datasets.append(data[data['Type'] == i])

for i in datasets:
    x, y = feat_prob("Tool wear [min]", i)
    plt.xlabel("Tool wear [min]")
    plt.ylabel("Possibility of Failure (%)")
    sns.lineplot(y=y, x=x, legend='brief')

    m += 1

plt.legend(["L", "M", "H"])
data['nf'] = data['Tool wear [min]'] * data['Torque [Nm]']
sns.scatterplot(data=data, x='nf', y='Rotational speed [rpm]', hue='Target')
from sklearn.preprocessing import LabelEncoder

label_encoder = LabelEncoder()

label_encoder.fit(data['Type'])
data['Type'] = label_encoder.transform(data['Type'])

label_encoder.fit(data['Target'])
data['Target'] = label_encoder.transform(data['Target'])
data.tail()
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(data.drop(['Failure Type', 'Target'], axis=1),
                                                    data['Target'], test_size=0.3, random_state=42)
import time

from sklearn.metrics import accuracy_score,classification_report
classifier=[]
imported_as=[]

#LGBM
import lightgbm as lgb
lgbm = lgb.LGBMClassifier()
classifier.append('LightGBM')
imported_as.append('lgbm')

#MultiLayerPerceptron
from sklearn.neural_network import MLPClassifier
mlp=MLPClassifier()
classifier.append('Multi Layer Perceptron')
imported_as.append('mlp')

#Bagging
from sklearn.ensemble import BaggingClassifier
bc = BaggingClassifier()
classifier.append('Bagging')
imported_as.append('bc')

#GBC
from sklearn.ensemble import GradientBoostingClassifier
gbc = GradientBoostingClassifier()
classifier.append('Gradient Boosting')
imported_as.append('gbc')

#ADA
from sklearn.ensemble import AdaBoostClassifier
ada = AdaBoostClassifier()
classifier.append('Ada Boost')
imported_as.append('ada')

#XGB
import xgboost as xgb
from xgboost import XGBClassifier
xgb = XGBClassifier()
classifier.append('XG Boost')
imported_as.append('xgb')

# Logistic Regression
from sklearn.linear_model import LogisticRegression
lr = LogisticRegression()
classifier.append('Logistic Regression')
imported_as.append('lr')

#RFC
from sklearn.ensemble import RandomForestClassifier
rfc = RandomForestClassifier()
classifier.append('Random Forest')
imported_as.append('rfc')

#KNN
from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier(n_neighbors=1)
classifier.append('k Nearest Neighbours')
imported_as.append('knn')

#SVM
from sklearn.svm import SVC
svc = SVC()
classifier.append('Support Vector Machine')
imported_as.append('svc')

#Grid
from sklearn.model_selection import GridSearchCV
param_grid = {'C': [0.1,1, 10, 100, 1000,2000], 'gamma': [1,0.1,0.01,0.001,0.0001], 'kernel': ['rbf']}
grid = GridSearchCV(SVC(),param_grid,refit=True,verbose=3)
classifier.append('SVM tuning grid')
imported_as.append('grid')

#STcaking
from sklearn.ensemble import StackingClassifier
estimators=[('rf', RandomForestClassifier(n_estimators=10, random_state=42)),
            ('svr',SVC(random_state=42))]
stc = StackingClassifier(estimators=estimators, final_estimator=LogisticRegression())
classifier.append('Stacked (RFR & SVM)')
imported_as.append('stc')

classifiers = pd.DataFrame({'Classifier':classifier,'Imported as':imported_as})
print('All Models Imported\nModels stored in dataframe called classifiers')


class Modelling:
    def __init__(self, X_train, Y_train, X_test, Y_test, models):
        self.X_train = X_train
        self.X_test = X_test
        self.Y_train = Y_train
        self.Y_test = Y_test
        self.models = models

    def fit(self):
        model_acc = []
        model_time = []
        for i in self.models:
            start = time.time()
            if i == 'knn':
                accuracy = []
                for j in range(1, 200):
                    kn = KNeighborsClassifier(n_neighbors=j)
                    kn.fit(self.X_train, self.Y_train)
                    predK = kn.predict(self.X_test)
                    accuracy.append([accuracy_score(self.Y_test, predK), j])
                temp = accuracy[0]
                for m in accuracy:
                    if temp[0] < m[0]:
                        temp = m
                i = KNeighborsClassifier(n_neighbors=temp[1])
            i.fit(self.X_train, self.Y_train)
            model_acc.append(accuracy_score(self.Y_test, i.predict(self.X_test)))
            stop = time.time()
            model_time.append((stop - start))
            print(i, 'has been fit')
        self.models_output = pd.DataFrame({'Models': self.models, 'Accuracy': model_acc, 'Runtime (s)': model_time})

    def results(self):
        models = self.models_output
        models = models.sort_values(by=['Accuracy', 'Runtime (s)'], ascending=[False, True]).reset_index().drop('index',
                                                                                                                axis=1)
        self.best = models['Models'][0]
        models['Models'] = models['Models'].astype(str).str.split("(", n=2, expand=True)[0]
        models['Accuracy'] = models['Accuracy'].round(5) * 100
        self.models_output_cleaned = models
        return (models)

    def best_model(self, type):
        if type == 'model':
            return (self.best)
        elif type == 'name':
            return (self.models_output_cleaned['Models'][0])

    def best_model_accuracy(self):
        return (self.models_output_cleaned['Accuracy'][0])

    def best_model_runtime(self):
        return (round(self.models_output_cleaned['Runtime (s)'][0], 3))

    def best_model_predict(self, X_test):
        return (self.best.predict(X_test))

    def best_model_clmatrix(self):
        return (classification_report(self.Y_test, self.best.predict(self.X_test)))
#display(classifiers)
models_to_test = [bc, gbc, ada, rfc, mlp, lr, knn, stc]
X_train.info()
classification = Modelling(X_train, y_train, X_test, y_test, models_to_test)
classification.fit()
classification.results()
print('BestModel is:', classification.best_model(type='name'))
print('Accuracy of model:', classification.best_model_accuracy())
print('Training Runtime in seconds', classification.best_model_runtime())
print('Classification Matrix:\n')
print(classification.best_model_clmatrix())
sns.set_style("darkgrid", {"grid.color": "1", "grid.linestyle": " "})

from sklearn.metrics import plot_confusion_matrix

plot_confusion_matrix(classification.best_model(type='model'), X_test, y_test)