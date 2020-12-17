from sklearn.base import BaseEstimator, TransformerMixin
from gensim.models.doc2vec import Doc2Vec, TaggedDocument
from nltk.tokenize import word_tokenize
 

class Article2vec():
    """ This class uses Dot2vec gensim model to provice article recommendations. 
    The methodolody is based on this awesome tutorial: 
    https://medium.com/@mishra.thedeepak/doc2vec-simple-implementation-example-df2afbbfbad5 \
    """
    
    def __init__(self, epochs=100, vec_size=20, alpha=0.025):
        self.epochs = epochs
        self.vec_size = vec_size
        self.alpha = alpha

        self.model =  Doc2Vec(size=self.vec_size,
                            alpha=self.alpha, 
                            min_alpha=0.00025,
                            min_count=1,
                            dm=1)

    def fit(self, X):
        """
        Train dot2vec model with provided data
        INPUT
            X - (pd.DataFrame) with training data
        OUTPUT
            self : (object) Article2vec object
        """
        tagged_data = []

        for idx, row in X.iterrows():
            tagged_data.append(TaggedDocument(words=word_tokenize(row['doc_full_name'].lower()), tags=[str(row['article_id'])]))

        self.model.build_vocab(tagged_data)

        for epoch in range(self.epochs):
            print('epoch {0}'.format(epoch))
            self.model.train(tagged_data, total_examples=self.model.corpus_count, epochs=self.model.iter)

            # Decrease the learning rate
            self.model.alpha -= 0.0002

            # Fix the learning rate, no decay
            self.model.min_alpha = self.model.alpha

        return self

    def save(self, name='article2v.model'):
        """
        Save model into disk
        INPUT
            name - (str) model file name
        OUTPUT
           None
        """
        self.model.save(name)
    
    def load(self, name='article2v.model'):
        """
        Load model from disk
        INPUT
            name - (str) model file name
        OUTPUT
           None
        """
        self.model = Doc2Vec.load(name)

    def recommend(self, article_id):
        """
        Find similar articles to the provided article id
        INPUT
            article_id - (str) article id
        OUTPUT
           None
        """
        return self.model.docvecs.most_similar(article_id)