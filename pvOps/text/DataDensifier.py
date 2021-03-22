from sklearn.base import BaseEstimator
import scipy

class DataDensifier(BaseEstimator):
    '''A data structure transformer which converts sparse data to dense data. This process is usually 
    incorporated in this library when doing unsupervised machine learning. This class is built 
    specifically to work inside a sklearn pipeline. Therefore, it uses the default ``transform``, ``fit``, 
    ``fit_transform`` method structure.
    '''
    def transform(self, X, y=None):
        '''Return a dense array if the input array is sparse.

        Parameters
        
        ----------
        X : array
            Input data of numerical values. For this package, these values could
            represent embedded representations of documents. 
        
        Returns

        -------
        dense array
        '''
        if scipy.sparse.issparse(X):
            return X.toarray()
        else:
            return X.copy()

    def fit(self, X, y=None):
        '''Placeholder method to conform to the sklearn class structure.

        Parameters
        
        ----------
        X : array
            Input data
        y : Not utilized.

        Returns

        -------
        DataDensifier object
        '''
        return self

    def fit_transform(self, X, y=None):
        '''Performs same action as ``DataDensifier.transform()``, 
        which returns a dense array when the input is sparse. 

        Parameters
        
        ----------
        X : array
            Input data
        y : Not utilized.

        Returns

        -------
        dense array
        '''
        return self.transform(X=X, y=y)