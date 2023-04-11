import numpy as np
from nltk.corpus import movie_reviews

#Get data and create instances/classes
instances = [(movie_reviews.raw(fileid), category) for category in movie_reviews.categories() for fileid in movie_reviews.fileids(category)]
np.random.shuffle(instances)

texts = [instance[0] for instance in instances]
classes = [instance[1] for instance in instances]



#Make your own sklearn feature vectorizers
#https://www.andrewvillazon.com/custom-scikit-learn-transformers

from numpy.random import randint
from sklearn.base import BaseEstimator, TransformerMixin


class CustomTransformer(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        # Perform arbitary transformation

        vow_arr = np.array([np.array([[ch=="a",ch=="e",ch=="i",ch=="o",ch=="u"] for ch in text]).sum(axis=0) for text in X])
        return vow_arr


ct = CustomTransformer()
out = ct.fit_transform(texts[:5])

print (out.shape)
