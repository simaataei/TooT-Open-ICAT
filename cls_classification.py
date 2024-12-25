from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
import ast
import numpy as np
from sklearn.metrics import matthews_corrcoef 
from joblib import dump
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.metrics import make_scorer, matthews_corrcoef


def load_embeddings(file_path):
    embeddings = []
    with open(file_path, 'r') as file:
        for line in file:
            # Use ast.literal_eval to safely convert string back to list
            embedding = ast.literal_eval(line.strip())
            embeddings.append(embedding)
    return np.array(embeddings)




def load_labels(file_path):
    with open(file_path, 'r') as file:
        labels = [int(line.strip()) for line in file]
    return np.array(labels)

# Paths to your files

'''
train_embeddings_path = 'Dataset/DeepIon/DeepIon_emb_train.txt'
train_labels_path = 'Dataset/DeepIon/y_train.txt'
test_embeddings_path = 'Dataset/DeepIon/DeepIon_emb_test.txt'
test_labels_path = 'Dataset/DeepIon/y_test.txt'
'''
val_embeddings_path = 'Dataset/frag/bin_emb_val.txt'
val_labels_path = 'Dataset/frag/y_val_bin.txt'

train_embeddings_path = 'Dataset/frag/bin_emb_train.txt'
train_labels_path = 'Dataset/frag/y_train_bin.txt'
test_embeddings_path = 'Dataset/frag/bin_emb_test.txt'
test_labels_path = 'Dataset/frag/y_test_bin.txt'


# Loading embeddings and labels
train_embeddings = load_embeddings(train_embeddings_path)
train_labels = load_labels(train_labels_path)
#val_embeddings = load_embeddings(val_embeddings_path)
#val_labels = load_labels(val_labels_path)
test_embeddings = load_embeddings(test_embeddings_path)
test_labels = load_labels(test_labels_path)


#combined_embeddings = np.concatenate((train_embeddings, val_embeddings))
#combined_labels = np.concatenate((train_labels, val_labels))
combined_embeddings = train_embeddings
combined_labels = train_labels
print(combined_labels)
#clf = RandomForestClassifier(n_estimators=100, random_state=42)




clf = LogisticRegression(random_state=42, max_iter=1000)

clf.fit(combined_embeddings, combined_labels)
test_predictions = clf.predict(test_embeddings)
mcc = matthews_corrcoef(test_labels, test_predictions)
print(f"Matthews Correlation Coefficient: {mcc}")
# Perform cross-validation
# cv=5 indicates 5-fold cross-validation
scores = cross_val_score(clf, combined_embeddings,combined_labels, cv=5)

print(f"Accuracy scores for each fold: {scores}")
print(f"Mean cross-validation accuracy: {scores.mean()}")

#LR
pipeline = make_pipeline(StandardScaler(), LogisticRegression(random_state=42, max_iter=1000))

# Setting up the parameter grid for hyperparameter tuning
param_grid = {
    'logisticregression__C': [0.001, 0.01, 0.1, 0.5, 1, 5, 10, 50, 100],
    'logisticregression__penalty': ['l2', 'l1'],  # 'l1' might require a solver that supports it, like 'liblinear' or 'saga'
    'logisticregression__solver': ['liblinear', 'saga'],  # 'saga' supports both 'l1' and 'l2'
    'logisticregression__class_weight': [None, 'balanced', {0: 0.25, 1: 0.75}, {0: 0.5, 1: 0.5}, {0: 0.75, 1: 0.25}]
}

'''
# RF

pipeline = make_pipeline(StandardScaler(), RandomForestClassifier(random_state=42))

# Setting up the parameter grid for hyperparameter tuning
param_grid = {
    'randomforestclassifier__n_estimators': [10, 50, 100, 200],  # Number of trees in the forest
    'randomforestclassifier__max_depth': [None, 10, 20, 30],  # Maximum depth of the tree
    'randomforestclassifier__min_samples_split': [2, 5, 10],  # Minimum number of samples required to split an internal node
    'randomforestclassifier__min_samples_leaf': [1, 2, 4],  # Minimum number of samples required to be at a leaf node
    'randomforestclassifier__max_features': ['auto', 'sqrt'],  # The number of features to consider when looking for the best split
    'randomforestclassifier__class_weight': [None, 'balanced', 'balanced_subsample']  # Weights associated with classes
}
'''
# Creating a custom scorer based on Matthews Correlation Coefficient
mcc_scorer = make_scorer(matthews_corrcoef)

# Using GridSearchCV for hyperparameter tuning and cross-validation
clf = GridSearchCV(pipeline, param_grid, cv=5, scoring=mcc_scorer)

# Fit the model on your training data
clf.fit(combined_embeddings, combined_labels)

# Best parameters found and the best score
print("Best parameters found:", clf.best_params_)
print("Best MCC score:", clf.best_score_)

# Make predictions on your test data
test_predictions = clf.predict(test_embeddings)
model_filename = 'LR_classifier_bin_mydata.joblib'
dump(clf, model_filename)


