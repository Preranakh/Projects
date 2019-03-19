import re
import numpy as np
from nltk.corpus import stopwords
import nltk
from nltk.stem import PorterStemmer
from nltk.stem import WordNetLemmatizer
from sklearn.externals import joblib
from sklearn.feature_extraction.text import CountVectorizer,TfidfVectorizer

tt = TfidfVectorizer()
loaded_model = joblib.load('trainedmodel.sav')
list = []
stops = stopwords.words('english')
st = PorterStemmer()
lemmatizer = WordNetLemmatizer()
vect =CountVectorizer()
input_str = input('Enter a string: ')
a = re.sub(r'@[a-zA-Z0-9_]+','',input_str)
b = re.sub(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+','',a)
c = re.sub(r'[.%+*/0-9?&#]+','',b)
d = re.sub(r'([-,]+)|((\')+)|([;:()!@#=$]+)','',c)
e = lambda x: " ".join(x for x in x.split() if x not in stops)
f = e(d)
g = lambda x: " ".join(x.lower() for x in x.split())
h = g(f)
print(type(h))
# i =nltk.word_tokenize(h)
# print(i)
j = lambda x: " ".join(st.stem(word) for word in x.split())
k= j(h)
l = lambda x: " ".join(lemmatizer.lemmatize(word) for word in x.split())
m = l(k)
print(m)
ques = np.array(m)
print(type(ques))
testing = tt.transform(ques)
loaded_model.predict(testing)
# print(vect.get_feature_names())
# print(testing.toarray())
# predicted =loaded_model.predict(m)
# print (i)

# clean_1 = re.compile(r'[.%+*/0-9?&#]+')
# pattern = clean_1.finditer(input_str)
# for a in pattern:
#     print (a)