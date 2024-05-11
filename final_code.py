import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import GridSearchCV, train_test_split, cross_validate
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.svm import SVC
from sklearn.feature_selection import SelectFromModel, SelectKBest, mutual_info_classif, mutual_info_regression
from sklearn import metrics
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler
from collections import Counter


#NOTE a lot of this code is borrowed from the template and previous assignments. 


game_data = pd.read_csv('games.csv')

####################################
#
# Global Params
#
####################################

feat_select = 0     #(Wrappers) 1:Random Forest  2:SVM    3:Gradient Boosting   4:univariate
random_forest = 0
svm = 0
keras = 1
gradient_boost = 0
cross_val = 0
class_rebalance = 0
grid_search = 0
rand_state = 117

####################################
#
# DATA PREPROCESSING
#
####################################

print("--- Preprecessing Data ---")

#dropping unwanted player names and match ID
game_data.drop(["id", "black_id", "white_id", "opening_name", "increment_code"], axis=1, inplace=True)

#dropping draw samples
game_data.drop(game_data[game_data['victory_status'] == 'draw'].index, inplace=True)
game_data.drop(game_data[game_data['victory_status'] == 'outoftime'].index, inplace=True)

#drop duplicate values
game_data.drop_duplicates(inplace=True)

#binning eco openings
for item in game_data["opening_eco"]:
    if "A" in item:
        game_data.replace(to_replace=item, value="Flank Opening", inplace=True)
    if "B" in item:
        game_data.replace(to_replace=item, value="Semi Open", inplace=True)
    if "C" in item:
        game_data.replace(to_replace=item, value="Open/FD", inplace=True)
    if "D" in item:
        game_data.replace(to_replace=item, value="Closed/Semi ", inplace=True)
    if "E" in item: 
        game_data.replace(to_replace=item, value="Indian Defense", inplace=True)

#discretize target
mapping = {'resign': 0, 'mate': 1, 'outoftime': 2, 'draw': 3}

game_data['victory_status'] = game_data['victory_status'].replace(mapping)

#replace moves with queen_moves
game_data["knight_moves"] = game_data["moves"].str.count("N")
game_data["king_moves_moves"] = game_data["moves"].str.count("K")
game_data["bishop_moves"] = game_data["moves"].str.count("B")
game_data["rook_moves"] = game_data["moves"].str.count("R")
game_data["moves"] = game_data["moves"].str.count("Q")
game_data = game_data.rename(columns={'moves': 'queen_moves'})

#merging start time and end time
game_data["game_time"] = (game_data["last_move_at"] - game_data["created_at"]) / 1000
game_data.drop(["created_at", "last_move_at"], axis=1, inplace=True)

#creating dummies
game_data = pd.get_dummies(data=game_data)

#normalizing values
scaler = MinMaxScaler()
game_data[[
    "game_time", "turns", "white_rating", "black_rating", "queen_moves", "opening_ply"]] = scaler.fit_transform(game_data[["game_time", "turns", "white_rating", "black_rating", "queen_moves", "opening_ply"]])

print("Data Frame Shape:\n", game_data.shape)
print("Data Types:\n", game_data.dtypes )
#seperate target and data
target = game_data["victory_status"]
data = game_data.drop('victory_status', axis=1)

####################################
#
# Feature Selection
#
####################################



if feat_select == 1:
    print("\n--- Feature Selection ON ---")
    clf = RandomForestClassifier(n_estimators=200, max_depth=None, min_samples_split=5, criterion='log_loss', random_state=rand_state)
    sel = SelectFromModel(clf, prefit=False, threshold='mean', max_features=None)                                                           #to select only based on max_features, set to integer value and set threshold=-np.inf
    print ('Wrapper Select - Random Forest: \n')

    fit_mod=sel.fit(data, target)    
    sel_idx=fit_mod.get_support()

    selected_feat_names = [col for (col, sel) in zip(data.columns, sel_idx) if sel]
    print("Selected Features: ", selected_feat_names,"\n")

    #updated data with selected features
    data = data[selected_feat_names]

if feat_select == 2:
    print("\n--- Feature Selection ON ---")
    clf = SVC(kernel='linear', gamma='scale', C=1.0, probability=True, random_state=rand_state)
    sel = SelectFromModel(clf, prefit=False, threshold='mean', max_features=None)                                                           #to select only based on max_features, set to integer value and set threshold=-np.inf
    print ('Wrapper Select - SVM: \n')

    fit_mod=sel.fit(data, target)    
    sel_idx=fit_mod.get_support()

    selected_feat_names = [col for (col, sel) in zip(data.columns, sel_idx) if sel]
    print("Selected Features: ", selected_feat_names,"\n")

    #updated data with selected features
    data = data[selected_feat_names]

if feat_select == 3:
    print("\n--- Feature Selection ON ---")
    clf = GradientBoostingClassifier(n_estimators=100, criterion="squared_error", loss="exponential", learning_rate=0.2, max_depth=None, min_samples_split=3, random_state=rand_state)
    sel = SelectFromModel(clf, prefit=False, threshold='mean', max_features=None)                                                           #to select only based on max_features, set to integer value and set threshold=-np.inf
    print ('Wrapper Select - Gradient Boosting: \n')

    fit_mod=sel.fit(data, target)    
    sel_idx=fit_mod.get_support()

    selected_feat_names = [col for (col, sel) in zip(data.columns, sel_idx) if sel]
    print("Selected Features: ", selected_feat_names,"\n")

    #updated data with selected features
    data = data[selected_feat_names]

if feat_select == 4:
    #Univariate Feature Selection - Mutual Info 
    print("\n--- Feature Selection ON ---")
    sel=SelectKBest(mutual_info_classif, k=10)
    fit_mod=sel.fit(data, target)    
    sel_idx=fit_mod.get_support()
    print ('Univariate Select - Mutual info regression: \n')

    selected_feat_names = [col for (col, sel) in zip(data.columns, sel_idx) if sel]
    print("Selected Features: ", selected_feat_names,"\n")

    #updated data with selected features
    data = data[selected_feat_names]

####################################
#
# Train Models
#
####################################
    
print("--- Model Output ---")

#train test split
data_train, data_test, target_train, target_test = train_test_split(data, target, test_size=0.30)

if class_rebalance == 1:
    #print("\n --- Class Rebalance ON ---")
    #sm = SMOTE(random_state=rand_state)
    #data_train, target_train = sm.fit_resample(data_train, target_train)
    #print('Resampled dataset shape %s' % Counter(target_train))

    rus = RandomUnderSampler(random_state=rand_state)
    data_train, target_train = rus.fit_resample(data_train, target_train)
    print('After RandomUnderSampler, resampled dataset shape %s' % Counter(target_train))

#----------------------------------------------------------------------------------------#

if random_forest == 1 and grid_search == 1:
    print("\n --- Grid Search ON --- ")
    param_grid = {'n_estimators': [10, 100, 150, 200, 500], 
              'min_samples_split': [1, 2, 3, 4, 5],
              'criterion': ['gini', 'entropy', 'log_loss']}
    grid = GridSearchCV(RandomForestClassifier(), param_grid, refit=True, verbose=3)
    grid.fit(data_train, target_train)
    print("\nBest Params: ", grid.best_params_)
    print('Random Forest Acc:', grid.score(data_test, target_test)) 
    print("Random Forest AUC: ", metrics.roc_auc_score(target_test, grid.predict_proba(data_test)[:,1]))


if random_forest == 1 and cross_val == 0:
    clf = RandomForestClassifier(n_estimators=200, max_depth=None, min_samples_split=5, criterion='log_loss', random_state=rand_state, class_weight="balanced")
    fit = clf.fit(data_train, target_train)
    print('Random Forest Acc:', clf.score(data_test, target_test)) 
    print("Random Forest AUC: ", metrics.roc_auc_score(target_test, clf.predict_proba(data_test)[:,1]))

    target_pred = clf.predict(data_test)
    cm = metrics.confusion_matrix(target_test, target_pred)
    print(cm)


if random_forest == 1 and cross_val == 1:
    print("\n --- Cross Validation ON --- ")
    scorers = {'Accuracy': 'accuracy', 'roc_auc': 'roc_auc'}
    clf = RandomForestClassifier(n_estimators=200, max_depth=None, min_samples_split=5, criterion='log_loss', random_state=rand_state, class_weight="balanced")
    scores = cross_validate(clf, data, target, scoring=scorers, cv=5)

    scores_Acc = scores['test_Accuracy']
    print("Random Forest Acc: %0.2f (+/- %0.2f)" % (scores_Acc.mean(), scores_Acc.std() * 2)) 
    scores_AUC= scores['test_roc_auc'] 
    print("Random Forest AUC: %0.2f (+/- %0.2f)" % (scores_AUC.mean(), scores_AUC.std() * 2))

    clf.fit(data_train, target_train)
    target_pred = clf.predict(data_test)
    cm = metrics.confusion_matrix(target_test, target_pred)
    print(cm)

#------------------------------------------------------------------------------------------------------#
    
if svm == 1 and grid_search == 1:  
    print("\n --- Grid Search ON --- ")
    param_grid = {'kernel': ["linear", "rbf", "sigmoid"], 
              'gamma': ["scale", "auto"],
              "C": [0.01, 0.1, 1, 5]}
    grid = GridSearchCV(SVC(probability=True), param_grid, refit=True, verbose=3)
    grid.fit(data_train, target_train)
    print("\nBest Params: ", grid.best_params_)
    print('SVM Acc:', grid.score(data_test, target_test)) 
    print("SVM AUC: ", metrics.roc_auc_score(target_test, grid.predict_proba(data_test)[:,1])) 

if svm == 1 and cross_val == 0:
    clf = SVC(kernel='linear', gamma='scale', C=5.0, probability=True, random_state=rand_state)
    clf.fit(data_train, target_train)
    print('SVM Acc:', clf.score(data_test, target_test)) 
    print("SVM AUC: ", metrics.roc_auc_score(target_test, clf.predict_proba(data_test)[:,1]))

    target_pred = clf.predict(data_test)
    cm = metrics.confusion_matrix(target_test, target_pred)
    print(cm)

if svm == 1 and cross_val == 1:
    scorers = {'Accuracy': 'accuracy', 'roc_auc': 'roc_auc'}
    clf = SVC(kernel='linear', gamma='scale', C=5.0, probability=True, random_state=rand_state)
    scores = cross_validate(estimator=clf, X=data, y=target, scoring=scorers, cv=5)

    scores_Acc = scores['test_Accuracy']                                                                                                                                    
    print("SVM Acc: %0.2f (+/- %0.2f)" % (scores_Acc.mean(), scores_Acc.std() * 2))                                                                                                    
    scores_AUC= scores['test_roc_auc']                                                                     #Only works with binary classes, not multiclass                  
    print("SVM AUC: %0.2f (+/- %0.2f)" % (scores_AUC.mean(), scores_AUC.std() * 2)) 

    target_pred = clf.predict(data_test)
    cm = metrics.confusion_matrix(target_test, target_pred)
    print(cm)

#-----------------------------------------------------------------------------------------------------------#

if gradient_boost == 1 and cross_val == 0:
    clf = GradientBoostingClassifier(n_estimators=100, criterion="squared_error", loss="exponential", learning_rate=0.2, max_depth=None, min_samples_split=3, random_state=rand_state)
    clf.fit(data_train, target_train)
    print('Gradient Boosting Acc:', clf.score(data_test, target_test)) 
    print("Gradient Boosting AUC: ", metrics.roc_auc_score(target_test, clf.predict_proba(data_test)[:,1]))

    target_pred = clf.predict(data_test)
    cm = metrics.confusion_matrix(target_test, target_pred)
    print(cm)

if gradient_boost == 1 and cross_val == 1:
    print("\n --- Cross Validation ON --- ")
    scorers = {'Accuracy': 'accuracy', 'roc_auc': 'roc_auc'}
    clf = GradientBoostingClassifier(n_estimators=100, criterion="squared_error", loss="exponential", learning_rate=0.2, max_depth=3, min_samples_split=3, random_state=rand_state)
    scores = cross_validate(estimator=clf, cv= 5, X=data, y=target, scoring=scorers)

    scores_Acc = scores['test_Accuracy']                                                                                                                                    
    print("Gradient Boosting Acc: %0.2f (+/- %0.2f)" % (scores_Acc.mean(), scores_Acc.std() * 2))                                                                                                    
    scores_AUC= scores['test_roc_auc']                                                                     #Only works with binary classes, not multiclass                  
    print("Gradient Boosting AUC: %0.2f (+/- %0.2f)" % (scores_AUC.mean(), scores_AUC.std() * 2))   

    clf.fit(data_train, target_train)
    target_pred = clf.predict(data_test)
    cm = metrics.confusion_matrix(target_test, target_pred)
    print(cm)    

if gradient_boost == 1 and grid_search == 1:  
    print("\n --- Grid Search ON --- ")
    param_grid = {'n_estimators': [100, 150, 200, 300], 
              'min_samples_split': [2, 3, 4, 5],
              'loss': ['deviance', 'exponential', 'log_loss'],
              "criterion": ['friedman_mse', 'squared_error'],
              "learning_rate": [0.1, 0.2, 0.3]}
    grid = GridSearchCV(GradientBoostingClassifier(), param_grid, refit=True, verbose=3)
    grid.fit(data_train, target_train)
    print("\nBest Params: ", grid.best_params_)
    print('Gradient Boosting Acc:', grid.score(data_test, target_test)) 
    print("Gradient Boosting AUC: ", metrics.roc_auc_score(target_test, grid.predict_proba(data_test)[:,1]))                  

#-----------------------------------------------------------------------------------------#

