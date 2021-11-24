
from soreva_metrics import calculate_metrics
import json
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.model_selection import StratifiedKFold, RepeatedStratifiedKFold, cross_val_score, train_test_split
from matplotlib import pyplot
import seaborn as sns
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.grid_search import GridSearchCV
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, AdaBoostClassifier, StackingClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import make_scorer
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler, RobustScaler
from imblearn.pipeline import make_pipeline
from imblearn.under_sampling import EditedNearestNeighbours, RandomUnderSampler, TomekLinks, OneSidedSelection, CondensedNearestNeighbour
from imblearn.ensemble import RUSBoostClassifier, EasyEnsembleClassifier, BalancedRandomForestClassifier, BalancedBaggingClassifier
from imblearn.over_sampling import RandomOverSampler, SMOTE, ADASYN, KMeansSMOTE, BorderlineSMOTE, SVMSMOTE

def extract_basic_features(breast):
    predictors = {}
    # basic features
    for key in ["tissue_density_predicted", "cancer_probability_predicted"]:
        predictors[key] = breast[key]
    predictors["max_malignant"] = 0.0
    predictors["max_benign"] = 0.0
    # get max probability for the objects that contain malignant and benign in the class name
    for view in ["CC", "MLO"]:
        malignant_objs_probs = [obj["probability"] for obj in breast[view] if "malignant" in obj["object_type"]]
        benign_objs_probs = [obj["probability"] for obj in breast[view] if "benign" in obj["object_type"]]
        if malignant_objs_probs:
            predictors["max_malignant"] = max(malignant_objs_probs)
        if benign_objs_probs:
            predictors["max_benign"] = max(benign_objs_probs)
    
    return predictors


with open("data_train/data_train.json", "r") as fin:
    data_train = json.load(fin)

targets_train = pd.read_csv("data_train/targets_train.csv", index_col=0)

predictors = {}
for key, value in data_train.items():
    predictors[key] = extract_basic_features(value)

df_train = pd.DataFrame.from_dict(predictors, orient="index")
df_train = pd.merge(df_train, targets_train, left_index=True, right_index=True)
y_train = df_train["BiRads"].copy()
X_train = df_train.loc[:, "tissue_density_predicted":"max_benign"].copy()

with open("data_test/data_test.json", "r") as fin:
    data_train = json.load(fin)


targets_test = pd.read_csv("data_test/targets_test.csv", index_col=0)
predictors = {}
for key, value in targets_test.items():
    predictors[key] = extract_basic_features(value)

df_test = pd.DataFrame.from_dict(predictors, orient="index")
X_test = df_test.loc[:, "tissue_density_predicted":"max_benign"].copy()


sns.countplot(x='BiRads', data=df_train)

sns.boxplot(y=df_train['probability'],x=df_train['BiRads'])

sns.boxplot(y=df_train['tissue_density_predicted'],x=df_train['BiRads'])

p = sns.countplot(x='BiRads',hue='tissue_density_predicted', data=df_train)
p.set_yscale("log")

knn = KNeighborsClassifier()
x_tr, x_t, y_tr, y_t = train_test_split(X_train, y_train, test_size=0.2,stratify=y_train)

params = {'weights': ['uniform', 'distance'], 'n_neighbors': list(range(1, 20))}
search_grid = GridSearchCV(knn, params, n_jobs=-1,scoring={'score':make_scorer(calculate_metrics)},refit='score',cv=5)

search_grid.fit(x_tr, y_tr)
search_grid.best_score_, search_grid.best_params_


def get_stacking():
	# define the base models
    level0 = list()
    level0.append(('lr', LogisticRegression()))
    level0.append(('knn', KNeighborsClassifier()))
    level0.append(('cart', DecisionTreeClassifier()))
    level0.append(('svm', SVC()))
    level0.append(('bayes', GaussianNB()))
    level0.append(('rfr', RandomForestClassifier()))
    level0.append(('gbc', GradientBoostingClassifier()))
    level0.append(('ada', AdaBoostClassifier()))
    level0.append(('RUS', RUSBoostClassifier()))
    level0.append(('BBC', BalancedBaggingClassifier()))
    level0.append(('EAC', EasyEnsembleClassifier()))
	# define meta learner model
    level1 = LogisticRegression()
	# define the stacking ensemble
    model = StackingClassifier(estimators=level0, final_estimator=level1, cv=5)
    return model

# get a list of models to evaluate
def get_models():
    models = dict()
    models['lr'] = LogisticRegression()
    models['knn'] = KNeighborsClassifier()
    models['cart'] = DecisionTreeClassifier()
    models['bayes'] = GaussianNB()
    models['rfr'] = RandomForestClassifier()
    models['gbc'] = GradientBoostingClassifier()
    models['svm'] = SVC()
    models['ada'] = AdaBoostClassifier()
    models['RUS'] = RUSBoostClassifier()
    models['BRF'] = BalancedRandomForestClassifier()
    models['BBC'] = BalancedBaggingClassifier()
    models['EAC'] = EasyEnsembleClassifier()
    models['stacking'] = get_stacking()
    return models


pca = PCA()
splitter = StratifiedKFold(3, shuffle=True)
scaler=StandardScaler()
preprocess = RobustScaler(quantile_range=(0.0, 90.0))
under = OneSidedSelection() #CondensedNearestNeighbour()###TomekLinks()#RandomUnderSampler()#
over = SMOTE() #RandomOverSampler()
clf = RandomForestClassifier()

def evaluate_model_pipeline(model, X, y):
    cv = RepeatedStratifiedKFold(n_splits=5, n_repeats=10)
    pipe = make_pipeline(pca, model)
    scores = cross_val_score(pipe, X, y, scoring=make_scorer(calculate_metrics), cv=cv, n_jobs=None, error_score='raise')
    return scores

# get the models to evaluate
models = get_models()
# evaluate the models and store results
results, names = list(), list()
for name, model in models.items():
	scores = evaluate_model_pipeline(model, X_train, y_train)
	results.append(scores)
	names.append(name)
	
# plot model performance for comparison
pyplot.boxplot(results, labels=names, showmeans=True)
pyplot.show()


level0 = list()
level0.append(('lr', LogisticRegression()))
level0.append(('knn', KNeighborsClassifier()))
level0.append(('cart', DecisionTreeClassifier()))
level0.append(('svm', SVC()))
level0.append(('bayes', GaussianNB()))
level0.append(('rfr', RandomForestClassifier()))
level0.append(('gbc', GradientBoostingClassifier()))
level0.append(('ada', AdaBoostClassifier()))
level0.append(('RUS', RUSBoostClassifier()))
level0.append(('BRF', BalancedRandomForestClassifier()))
level0.append(('BBC', BalancedBaggingClassifier()))
level0.append(('EAC', EasyEnsembleClassifier()))
level1 = LogisticRegression()
model = StackingClassifier(estimators=level0, final_estimator=level1, cv=5, passthrough = True)

pipe = make_pipeline(pca, model)
Y_predicted_test = model.fit(X_train, y_train.reshape(-1)).predict(X_test)
Y_predicted_train = model.predict(X_train)