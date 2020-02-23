#! /usr/bin/env python3

import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.svm import LinearSVC
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.multiclass import OneVsRestClassifier
from sklearn import preprocessing

X_train = np.array(["POS MERCHANDISE",
"POS MERCHANDISE STARBUCKS #86",
"POS MERCHANDISE NETFLIX #2966",
"POS MERCHANDISE HOPS N GRAINS",
"JC PENNY #3499",
"POS MERCHANDISE AMAZON PRIME #3455",
"FARM BOY #90 NEPEAN ON",
"WAL-MART #3455 NEPEAN ON",
"COSTCO GAS W1263 NEPEAN ON",
"COSTCO WHOLESALE W1263 NEPEAN ON",
"FARM BOY #90",
"LOBLAWS 1035",
"ARMANI EXCAHNGE #625",
"POS MERCHANDISE PLAYSTORE #456"
])
y_train_text = [["NETFLIX","Expenses:Miscellenaous"],
["TIM HORTONS","Expenses:Food:Dinning"],
["HOPS N GRAINS","Expenses:Food:Alcohol-tobacco"],
["RONA HOME & GARDEN","Expenses:Auto"],
["JC PENNY","Expenses:Shopping:Clothing"],
["ARMANI EXCHANGE","Expenses:Shopping:Clothing"],
["FARM BOY","Expenses:Food:Groceries"],
["STARBUCKS","Expenses:Food:Dining"],
["COSTCO GAS","Expenses:Auto:Gas"],
["COSTCO","Expenses:Food:Groceries"],
["FARM BOY","Expenses:Food:Groceries"],
["LOBLAWS","Expenses:Food:Groceries"],
["WAL-MART","Expenses:Food:Groceries"],
["STARBUCKS","Expenses:Food:Dinning"]]

X_test = np.array(['POS MERCHANDISE STARBUCKS #123',
                   'STARBUCKS #589',
                   'POS COSTCO GAS',
                   'COSTCO WHOLESALE',
                   "AMAZON PRIME",
                   'HOPS N GRAINS',
                   'TRANSFER OUT',
                   'TRANSFER IN',
                   'NETFLIX',
                   'ARMANI EXCHANGE',
                   'WAL-MART',
                   'WALMART'])

#target_names = ['New York', 'London']

lb = preprocessing.MultiLabelBinarizer()
Y = lb.fit_transform(y_train_text)

classifier = Pipeline([
    ('vectorizer', CountVectorizer()),
    ('tfidf', TfidfTransformer()),
    ('clf', OneVsRestClassifier(LinearSVC()))])

classifier.fit(X_train, Y)
predicted = classifier.predict(X_test)
all_labels = lb.inverse_transform(predicted)

for item, labels in zip(X_test, all_labels):
    print ('%s => %s' % (item, ', '.join(labels)))
