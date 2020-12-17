from sklearn.base import BaseEstimator, TransformerMixin
from gensim.models.doc2vec import Doc2Vec, TaggedDocument
from nltk.tokenize import word_tokenize
 

class Article2vec():
    """ This class uses Dot2vec gensim model to provice article recommendations. 
    The methodolody is based on this awesome tutorial: 
    https://medium.com/@mishra.thedeepak/doc2vec-simple-implementation-example-df2afbbfbad5 \
    """
    
    def __init__(self, epochs=100, vec_size=20, alpha=0.025):
        self.model = None
        self.epochs = epochs
        self.vec_size = vec_size
        self.alpha = alpha

    def fit(self, X):
        tagged_data = []

        for idx, row in X.iterrows():
            tagged_data.append(TaggedDocument(words=word_tokenize(row['doc_full_name'].lower()), tags=[str(row['article_id'])]))

        model = Doc2Vec(size=self.vec_size,
                            alpha=self.alpha, 
                            min_alpha=0.00025,
                            min_count=1,
                            dm=1)
                            
        model.build_vocab(tagged_data)

        for epoch in range(self.epochs):
            print('epoch {0}'.format(epoch))

            model.train(tagged_data, total_examples=model.corpus_count, epochs=model.iter)

            # decrease the learning rate
            model.alpha -= 0.0002

            # fix the learning rate, no decay
            model.min_alpha = model.alpha

        self.model = model

        return self

    def save(self, name='article2v.model'):
        self.model.save(name)
    
    def load(self, name='article2v.model'):
        self.model = Doc2Vec.load(name)

    def recommend(self, article_id):
        return self.model.docvecs.most_similar(article_id)