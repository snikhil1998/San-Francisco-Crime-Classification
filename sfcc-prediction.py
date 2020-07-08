from datetime import datetime
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import log_loss
import xgboost as xgb
import s2sphere
import pickle

class Models:
	def __init__(self, train_path, test_path):
		self.train = pd.read_csv(train_path)
		self.test = pd.read_csv(test_path)

	def dummy_df(self, data, columns):
		for col in columns:
			dummies = pd.get_dummies(data[col],prefix=col, dummy_na=False)
			data = data.drop(col, 1)
			data = pd.concat([data, dummies], axis=1)
		return data

	def preProcessing(self):
		data = pd.concat(objs=[self.train, self.test], axis=0).reset_index(drop=True)

		data.insert(loc=0, column='Year', value=data['Dates'].map(lambda x: datetime.strptime(x, '%Y-%m-%d %H:%M:%S').year))
		data.insert(loc=1, column='Month', value=data['Dates'].map(lambda x: datetime.strptime(x, '%Y-%m-%d %H:%M:%S').month))
		data.insert(loc=2, column='Hour', value=data['Dates'].map(lambda x: datetime.strptime(x, '%Y-%m-%d %H:%M:%S').hour))

		data.insert(loc=8, column='Intersection', value=data['Address'].map(lambda x : 1 if '/' in x else 0))
		data.insert(loc=9, column='Block', value=data['Address'].map(lambda x : 1 if 'Block' in x else 0))
		data.insert(loc=10, column='StreetSuffix', value=data['Address'].map(lambda x: x.strip().split()[-1] if len(x.strip().split()[-1])==2 else "0"))

		data['SimulCrime'] = data[['Dates', 'Address']].apply(lambda x: ' '.join(x), axis=1)
		simultaneous_crimes = data['SimulCrime'].value_counts().to_dict()
		data['SimulCrime'] = data['SimulCrime'].map(lambda x: simultaneous_crimes[x])

		data.drop('Dates', axis='columns', inplace=True)
		data.drop('Category', axis='columns', inplace=True)
		data.drop('Descript', axis='columns', inplace=True)
		data.drop('Id', axis='columns', inplace=True)
		data.drop('Address', axis='columns', inplace=True)
		data.drop('Resolution', axis='columns', inplace=True)

		data['X'] = data['X'].map(lambda x: x if x<-122.3 else -122.3)
		data['Y'] = data['Y'].map(lambda x: x if x<37.82 else 37.82)
		data['Coordinate'] = data[['X', 'Y']].apply(lambda x: s2sphere.CellId.from_lat_lng(s2sphere.LatLng.from_degrees(x[0], x[1])).level(), axis=1)

		data.drop('X', axis='columns', inplace=True)
		data.drop('Y', axis='columns', inplace=True)

		data = self.dummy_df(data, data.columns)

		return data

	def split_data(self):
		data = self.preProcessing()
		return [data[:len(self.train)], LabelEncoder().fit_transform(self.train.Category.values), data[len(self.train):]]

	def LogisticRegressionModel(self):
		datasets = self.split_data()
		lr = LogisticRegression().fit(datasets[0], datasets[1])

		pkl_filename = 'sfcc-prediction.pkl'
		pkl_file = open('sfcc-prediction.pkl', "wb")
		pickle.dump(lr, pkl_file)
		pkl_file.close()

		test_y = lr.predict_proba(datasets[2])

		categories = pd.get_dummies(self.train['Category']).columns

		submit = pd.DataFrame(test_y, columns=categories)
		submit['Id'] = self.test['Id']
		submit.to_csv("sfcc_pred.csv", index=None)

	def RandomForestModel(self):
		datasets = self.split_data()
		rf = RandomForestClassifier(n_estimators=300, max_depth=20, min_samples_leaf = 1, n_jobs=10, random_state=10).fit(datasets[0], datasets[1])

		pkl_filename = 'sfcc-prediction.pkl'
		pkl_file = open('sfcc-prediction.pkl', "wb")
		pickle.dump(rf, pkl_file)
		pkl_file.close()

		test_y = rf.predict_proba(datasets[2])

		categories = pd.get_dummies(self.train['Category']).columns

		submit = pd.DataFrame(test_y, columns=categories)
		submit['Id'] = self.test['Id']
		submit.to_csv("sfcc_pred.csv", index=None)

	def XGBoostModel(self):
		datasets = self.split_data()
		param = {'max_depth':5, 'silent':1, 'eta':0.4, 'objective':'multi:softprob', 'sub_sample':0.9, 'num_class':36, 'eval_metric':'mlogloss'}
		xgbtrain = xgb.DMatrix(datasets[0], datasets[1])
		xgbtest = xgb.DMatrix(datasets[2])
		bst = xgb.train(param, xgbtrain, 50)

		pkl_filename = 'sfcc-prediction.pkl'
		pkl_file = open('sfcc-prediction.pkl', "wb")
		pickle.dump(bst, pkl_file)
		pkl_file.close()

		test_y = bst.predict(xgbtest)

		categories = pd.get_dummies(self.train['Category']).columns

		submit = pd.DataFrame(test_y, columns=categories)
		submit['Id'] = self.test['Id']
		submit.to_csv("sfcc_pred.csv", index=None)


if __name__=="__main__":
	#train_path = '../input/train.csv'
	#test_path = '../input/test.csv'
	train_path = 'train.csv'
	test_path = 'test.csv'
	model = Models(train_path, test_path)
	model.RandomForestModel()
