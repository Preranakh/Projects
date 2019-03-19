from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB,BernoulliNB
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn import metrics
from sklearn.externals import joblib
import pandas as pd
import numpy as np
from sklearn import preprocessing
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import cross_val_score

df =pd.read_csv('data_finalized.csv', encoding='latin')

tt = TfidfVectorizer(use_idf=False)
tf_train = tt.fit_transform(df['lemmatized'])
x_train, x_test,y_train,y_test = train_test_split(tf_train,df['target'],test_size=0.2)


nb =MultinomialNB().fit(x_train,y_train)
# filename ="trainedmodel.sav"
# joblib.dump(nb,filename)
loaded_model =joblib.load("trainedmodel.sav")
# nb.score(x_train,y_train)
predicted = loaded_model.predict(x_test)
print("MultinomialNB Accuracy:",metrics.accuracy_score(y_test, predicted))
abc = cross_val_score(loaded_model,x_train,y_train,cv=10)
print("The accuracy by crossvalidation is: %0.10f"% abc.mean())
print(classification_report(y_test,predicted))
a = np.array(["That movie was awesome."])
b = tt.transform(a)
save = loaded_model.predict(b)
if save[0] == 0:
    print ("Sentiment : Negative")
else:
    print ("Sentiment :Positive")


print(confusion_matrix(y_test,predicted))
