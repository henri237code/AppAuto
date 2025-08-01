from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC

def get_classification_models():
    return {
        "Régression Logistique (newton-cg)": LogisticRegression(solver='newton-cg'),
        "Arbre de Décision (entropy)": DecisionTreeClassifier(criterion='entropy'),
        "Arbre de Décision (gini)": DecisionTreeClassifier(criterion='gini'),
        "Analyse Discriminante Linéaire (LDA)": LinearDiscriminantAnalysis(),
        "K-Nearest Neighbors (k=30)": KNeighborsClassifier(n_neighbors=30),
        "Naive Bayes Gaussien": GaussianNB(),
        "SVM (probability=True)": SVC(probability=True)
    }

def get_regression_model():
    return LinearRegression()
