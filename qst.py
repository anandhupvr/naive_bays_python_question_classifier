import nltk
import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.cross_validation import train_test_split
# import and instantiate a Multinomial Naive Bayes model
from sklearn.naive_bayes import MultinomialNB
from sklearn import metrics
from sklearn.datasets import load_iris
iris = load_iris()


def qst_type(test_qst):
	path = 'qstns.tsv'
	sent = pd.read_table(path, header=None, names=['qtype', 'sentance'])
	sent.shape
	sent['label_num'] = sent.qtype.map({'whqst':0, 'statement':1,'yesOrNo':2})
	X = iris.data
	y = iris.target
	X = sent.sentance
	y = sent.label_num
	sent.head(5)


	X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=1)

	vect = CountVectorizer()
	vect.fit(X_train)
	X_train_dtm = vect.transform(X_train)
	X_train_dtm = vect.fit_transform(X_train)
	X_train_dtm
	X_test_dtm = vect.transform(X_test)
	X_test_dtm


	nb = MultinomialNB()
	nb.fit(X_train_dtm, y_train)
	y_pred_class = nb.predict(X_test_dtm)
	#a=y_pred_class.tostring()
	#print (a)
	#metrics.accuracy_score(y_test, y_pred_class)

	
	test_qst_dtm = vect.transform(test_qst)
	test_qst_dtm
	y_pred_class = nb.predict(test_qst_dtm)
	a = y_pred_class
	if a == 0:
		c = "wh_questions"
		return(c)
	elif a == 1:
		c = ["statement"]
	elif a == 2:
		c == ["yes or no question"]

	return(a)

#Just edit the question here and try type of qustions
test_qst = ["Tell me city which is capital of India"]
q_type = qst_type(test_qst)
print(q_type)
