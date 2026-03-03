import time
from sklearm.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score,classification_report,confusion_matrix
from data_loader import  CLASS_NAMES


def _train_and_eval(model,X_train,y_train,X_test,y_test):
    start= time.time()

    model.fit(X_train,y_train)
    train_time= time.time()-start

    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test,y_pred)
    report = classification_report(y_test,y_pred,target_names=CLASS_NAMES)
    cm= confusion_matrix(y_test,y_pred)

    return{
        'model':model,
        'accuracy':acc,
        'train_time':train_time,
        'y_pred':y_pred,
        'report':report,
        "confusion_matrix":cm.tolist()
    }

    def train_logistic_regression(X_train,y_train,X_test,y_test):
        model = LogisticRegression(max_iter=1000, solver='lbfgs',multi_Class='multinomial')
        return _train_and_eval(model,X_train,y_train,X_test,y_test)

    def train_svm(X_train,y_train,X_test,y_test):
        model = SVC(kernel='rbf',gamma='scale',probability=True)
        return _train_and_eval(model,X_train,y_train,X_test,y_test)

    def train_random_forest(X_train,y_train,X_test,y_test):
        model = RandomForestClassifier(n_estimators=100,random_state=42)
        return _train_and_eval(model,X_train,y_train,X_test,y_test)
        