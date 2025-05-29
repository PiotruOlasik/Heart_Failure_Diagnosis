import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, classification_report, roc_auc_score, roc_curve, confusion_matrix
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.decomposition import PCA
from KNN import *
from sklearn.naive_bayes import GaussianNB
from Naive_Bayes import *
import joblib
from tensorflow.keras.models import load_model
from Data_exploration import Data_exploration
from load_data import load_heart_data, data_to_int
from preprocess import preprocess_data

#Analiza danych
dataset = load_heart_data("heart.csv")
data_int = data_to_int(dataset)
data_exploration = Data_exploration(data_int)
data_exploration.summarize()

X_train, X_test, Y_train, Y_test = preprocess_data(dataset)

#Zbiory po redukcji wymiarowości
pca_4 = PCA(n_components=4)
pca_6 = PCA(n_components=6)
pca_8 = PCA(n_components=8)
X_train_pca_4 = pca_4.fit_transform(X_train)
X_test_pca_4 = pca_4.transform(X_test)
X_train_pca_6 = pca_6.fit_transform(X_train)
X_test_pca_6 = pca_6.transform(X_test)
X_train_pca_8 = pca_8.fit_transform(X_train)
X_test_pca_8 = pca_8.transform(X_test)


datasets = [
    ('Normal - Scaled', X_train, X_test, Y_train, Y_test),
    ('PCA_4', X_train_pca_4, X_test_pca_4, Y_train, Y_test),
    ('PCA_6', X_train_pca_6, X_test_pca_6, Y_train, Y_test),
    ('PCA_8', X_train_pca_8, X_test_pca_8, Y_train, Y_test)
]


#       -----KNN-----

param_grid = {
    'n_neighbors': [3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15],
    'weights': ['uniform', 'distance'],
    'metric': ['euclidean', 'manhattan', 'canberra', 'minkowski']
}

#Szukanie odpowiednich parametrów za pomocą GridSearcha
for name, X_tr, X_te, Y_tr, Y_te in datasets:
    grid_search = GridSearchCV(KNeighborsClassifier(), param_grid, cv=5, scoring='recall')
    grid_search.fit(X_tr, Y_tr)

    print(f'\n----- KNN on {name} dataset -----\n')
    print("Best parameters found: ", grid_search.best_params_)
    print("\nBest cross-validation accuracy:", grid_search.best_score_)

    best_knn = grid_search.best_estimator_
    Y_pred = best_knn.predict(X_te)

    print("\nTEST ACCURACY:", accuracy_score(Y_te, Y_pred))
    print('\nClassification Report:')
    print("\n", classification_report(Y_te, Y_pred))
    conf_mat = confusion_matrix(Y_te, Y_pred)
    class_names = ['healthy', 'HeartDisease']
    plt.figure(figsize=(6, 5))
    sns.heatmap(conf_mat, annot=True, fmt='d', cmap='Blues', cbar=False, xticklabels=class_names,
                yticklabels=class_names)
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.title(f'Confusion Matrix - best KNN on {name} dataset')
    plt.tight_layout()
    plt.show()


#Sprawdzenie własnym KNN zbudowanym na bazie poprzednich wskazań
for neighbours in range(8,19):
    my_knn = KNNClass()
    Y_pred_handmade = my_knn.knn(X_train, Y_train, X_test, k=neighbours)
    accuracy=accuracy_score(Y_test, Y_pred_handmade)
    print(f"FOR K={neighbours}")
    print("Accuracy: ",accuracy)
    print("Classification Report:\n")
    print(classification_report(Y_test, Y_pred_handmade))
    conf_mat = confusion_matrix(Y_test, Y_pred_handmade)
    class_names = ['healthy', 'HeartDisease']
    plt.figure(figsize=(6, 5))
    sns.heatmap(conf_mat, annot=True, fmt='d', cmap='Blues', cbar=False, xticklabels=class_names, yticklabels=class_names)
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.title(f'Confusion Matrix - handmade KNN - for k={neighbours}')
    plt.tight_layout()
    plt.show()

#Najlepsze wyniki dla k=13 i k=17




#      -----NAIVE BAYES-----

#from sklearn
for name, X_tr, X_te, Y_tr, Y_te in datasets:
    model_bayes = GaussianNB()
    model_bayes.fit(X_tr, Y_tr)
    Y_pred = model_bayes.predict(X_te)
    print("-----NAIVE BAYES, DATASET: ",name)
    print("\nTEST ACCURACY:", accuracy_score(Y_te, Y_pred))
    print('\nClassification Report:')
    print("\n",classification_report(Y_te, Y_pred))
    conf_mat = confusion_matrix(Y_te, Y_pred)
    class_names = ['healthy', 'HeartDisease']
    plt.figure(figsize=(6, 5))
    sns.heatmap(conf_mat, annot=True, fmt='d', cmap='Blues', cbar=False, xticklabels=class_names, yticklabels=class_names)
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.title(f'Confusion Matrix - Naive Bayes on {name} dataset')
    plt.tight_layout()
    plt.show()


#Handmade Naive Bayes
for name, X_tr, X_te, Y_tr, Y_te in datasets:
    my_bayes = NaiveBayes()
    Y_pred_my_NB = my_bayes.get_predictions(X_tr, Y_tr, X_te)
    print("-----MY Naive BAYES, DATASET: ",name)
    print("\nTEST ACCURACY:", accuracy_score(Y_te, Y_pred_my_NB))
    print('\nClassification Report:')
    print("\n",classification_report(Y_te, Y_pred_my_NB))
    conf_mat = confusion_matrix(Y_te, Y_pred_my_NB)
    class_names = ['healthy', 'HeartDisease']
    plt.figure(figsize=(6, 5))
    sns.heatmap(conf_mat, annot=True, fmt='d', cmap='Blues', cbar=False, xticklabels=class_names, yticklabels=class_names)
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.title(f'Confusion Matrix - My Bayes on {name} dataset')
    plt.tight_layout()
    plt.show()





#  ---- MODEL TENSORFLOW W CELU DOKŁADNIEJSZEJ ANALIZY  ----

# Załadowanie modelu i danych testowych
model = load_model("heart_failure_nn_model.h5")
X_test_model, y_test_model = joblib.load("test_data.pkl")

# Przewidywanie i ocena
y_pred_probs_model = model.predict(X_test_model).flatten()
y_pred_model = np.round(y_pred_probs_model)

#Podsumowanie
print("\nClassification Report:")
print(classification_report(y_test_model, y_pred_model))
print("ROC AUC Score:", roc_auc_score(y_test_model, y_pred_probs_model))
print('\nConfusion Matrix:')
conf_mat = confusion_matrix(y_test_model, y_pred_model)
class_names = ['healthy', 'HeartDisease']
plt.figure(figsize=(6, 5))
sns.heatmap(conf_mat, annot=True, fmt='d', cmap='Blues', cbar=False, xticklabels=class_names, yticklabels=class_names)
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.title('Confusion Matrix based on model from Tensorflow:')
plt.tight_layout()
plt.show()

# WYkres
fpr, tpr, thresholds = roc_curve(y_test_model, y_pred_probs_model)
plt.figure()
plt.plot(fpr, tpr, label='ROC Curve (area = {:.2f})'.format(roc_auc_score(y_test_model, y_pred_probs_model)))
plt.plot([0, 1], [0, 1], 'k--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic')
plt.legend(loc='lower right')
plt.grid(True)
plt.tight_layout()
plt.savefig("roc_curve.png")
plt.show()
