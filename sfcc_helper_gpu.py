import datetime
import cupy
import cudf
import matplotlib.pyplot as plt
import cuml.linear_model
import cuml.ensemble
import cuml.preprocessing
import cuml.metrics
import xgboost as xgb
import s2sphere
import pickle

class Models:
	def __init__(self, train_path, test_path):
		self.train = cudf.read_csv(train_path)
		self.test = cudf.read_csv(test_path)

	def dummy_df(self, data, columns):
		for col in columns:
			dummies = cudf.get_dummies(data[col],prefix=col, dummy_na=False)
			data = data.drop(col, axis=1)
			data = cudf.concat([data, dummies], axis=1)
		return data

	def preProcessing(self):
		data = cudf.concat(objs=[self.train, self.test], axis=0).reset_index(drop=True)

		dates = list(map(lambda x: datetime.datetime.strptime(x, '%Y-%m-%d %H:%M:%S'), data['Dates'].to_pandas()))
		data.insert(loc=0, name='Year', value=list(map(lambda x: x.year, dates)))
		data.insert(loc=1, name='Month', value=list(map(lambda x: x.month, dates)))
		data.insert(loc=2, name='Hour', value=list(map(lambda x: x.hour, dates)))

		addresses = data['Address'].to_pandas()
		data.insert(loc=8, name='Intersection', value=list(map(lambda x : 1 if '/' in x else 0, addresses)))
		data.insert(loc=9, name='Block', value=list(map(lambda x : 1 if 'Block' in x else 0, addresses)))
		data.insert(loc=10, name='StreetSuffix', value=list(map(lambda x: x.strip().split()[-1] if len(x.strip().split()[-1])==2 else "0", addresses)))

		# data['SimulCrime'] = data[['Dates', 'Address']].apply(lambda x: ' '.join(x), axis=1)
		data['SimulCrime'] = list(map(lambda x: ' '.join(x), data[['Dates', 'Address']].to_numpy()))
		simultaneous_crimes = data['SimulCrime'].value_counts().to_dict()
		data['SimulCrime'] = list(map(lambda x: simultaneous_crimes[x], data['SimulCrime'].to_pandas()))

		data.drop('Dates', axis='columns', inplace=True)
		data.drop('Category', axis='columns', inplace=True)
		data.drop('Descript', axis='columns', inplace=True)
		data.drop('Id', axis='columns', inplace=True)
		data.drop('Address', axis='columns', inplace=True)
		data.drop('Resolution', axis='columns', inplace=True)

		data['X'] = list(map(lambda x: x if x<-122.3 else -122.3, data['X'].to_pandas()))
		data['Y'] = list(map(lambda x: x if x<37.82 else 37.82, data['Y'].to_pandas()))
		data['Coordinate'] = list(map(lambda x: s2sphere.CellId.from_lat_lng(s2sphere.LatLng.from_degrees(x[0], x[1])).level(), data[['X', 'Y']].to_numpy()))

		data.drop('X', axis='columns', inplace=True)
		data.drop('Y', axis='columns', inplace=True)

		data = self.dummy_df(data, data.columns)

		return data

	def split_data(self):
		data = self.preProcessing()
		# Temporarily converting self.train to pandas DataFrame because of "TypeError: String Arrays is not yet implemented in cudf"
		# Should remove `.to_pandas()` in the future when String Arrays is implemented for cudf
		return [data[:len(self.train)].astype(dtype=cupy.float32).astype(dtype=cupy.float32), cuml.preprocessing.LabelEncoder().fit_transform(self.train.to_pandas().Category.values).astype(dtype=cupy.float32), data[len(self.train):].astype(dtype=cupy.float32)]

	def LogisticRegressionModel(self):
		datasets = self.split_data()
		lr = cuml.linear_model.LogisticRegression().fit(datasets[0], datasets[1])

		pkl_filename = 'sfcc-logistic_regression-model.pkl'
		pkl_file = open('sfcc-logistic_regression-model.pkl', "wb")
		pickle.dump(lr, pkl_file)
		pkl_file.close()

		test_y = lr.predict_proba(datasets[2])

		categories = cudf.get_dummies(self.train['Category']).columns

		submit = cudf.DataFrame(test_y, columns=categories)
		submit['Id'] = self.test['Id']
		submit.to_csv("sfcc-logistic_regression-predictions.csv", index=None)

	def RandomForestModel(self):
		datasets = self.split_data()
		rf = cuml.ensemble.RandomForestClassifier(n_estimators=300, max_depth=20, min_samples_leaf=1, n_streams=1, random_state=10).fit(datasets[0], datasets[1])

		pkl_filename = 'sfcc-random_forest-model.pkl'
		pkl_file = open('sfcc-random_forest-model.pkl', "wb")
		pickle.dump(rf, pkl_file)
		pkl_file.close()

		test_y = rf.predict_proba(datasets[2])

		categories = cudf.get_dummies(self.train['Category']).columns

		submit = cudf.DataFrame(test_y, columns=categories)
		submit['Id'] = self.test['Id']
		submit.to_csv("sfcc-random_forest-predictions.csv", index=None)

	def XGBoostModel(self):
		datasets = self.split_data()
		param = {'max_depth':5, 'silent':1, 'eta':0.4, 'objective':'multi:softprob', 'sub_sample':0.9, 'num_class':36, 'eval_metric':'mlogloss'}
		xgbtrain = xgb.DMatrix(datasets[0], datasets[1])
		xgbtest = xgb.DMatrix(datasets[2])
		bst = xgb.train(param, xgbtrain, 50)

		pkl_filename = 'sfcc-xgboost-model.pkl'
		pkl_file = open('sfcc-xgboost-model.pkl', "wb")
		pickle.dump(bst, pkl_file)
		pkl_file.close()

		test_y = bst.predict(xgbtest)

		categories = cudf.get_dummies(self.train['Category']).columns

		submit = cudf.DataFrame(test_y, columns=categories)
		submit['Id'] = self.test['Id']
		submit.to_csv("sfcc-xgboost-predictions.csv", index=None)
