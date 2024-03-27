#file
import shap 
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import LinearSVC
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
from sklearn.cluster import KMeans
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier

#one possible way to implement 


#another way
class ShapExplainer:
    def __init__(self, sad): 
        self.model = sad.model
        self.X_train = sad.X_train
        self.X_test = sad.X_test
        self.vec = sad.vectorizer
        self.Svals = None
        self.expl = None
        self.istree =  False # should be false 
        self.func = self._scuffmapping()



    def test(self):
        print("you called?")

    # Different shap explainers
    def linear(self):
        self.expl = shap.LinearExplainer(self.model, self.X_train)
        return self.expl

    # shjould xgb be a tree?
    def tree(self):
        self.expl  = shap.TreeExplainer(self.model, data=self.X_train.toarray())
        return self.expl

    def kernel(self):
        self.expl = shap.KernelExplainer(self.model.predict, self.X_train)
        return self.expl


    def plain(self):
        self.expl = shap.Explainer(self.model)
        return self.expl


    # a function for mapping, could be changed to cases in python 3.10
    def _scuffmapping(self):
        """_summary_

        Returns:
            _type_: _description_
        """
        # linear
        if isinstance(self.model, (LogisticRegression,LinearSVC)):
            return self.linear
        #tree
        elif isinstance(self.model, (IsolationForest,DecisionTreeClassifier,RandomForestClassifier)):
            self.istree = True
            return self.tree

        elif isinstance(self.model, (XGBClassifier)):
            return self.plain
        else:
            raise "joku hyvä errror"
            pass

    # sample default test data??
    # should this return?
    def calc_shapvalues(self, test_data=None):
        """
        This function creates shap values for given dataset. The data should be accpetable
        by trained anomaly detection model.
        """
        if test_data == None:
            test_data = self.X_test 

        # change to be in init
        try:
            func = self._scuffmapping()
            expl = func()
            if self.istree:
                self.Svals = expl(test_data.toarray())      # to array ?=??
            else:
                elf.Svals = expl(test_data)
            return self.Svals
        except TypeError:
            print("Current model is unsupported")



    # implement slice
    def plot(self, data=None,plottype="summary", slice=None):
        """
        Create a plot with given data or previous shapvalues.
        plots: summary, bar
        """

        # bad structre -> shap values again?
        if data == None:
            self.calc_shapvalues()
            data = self.X_test 
            #print(self.Svals.shape, data.shape)
        else:
            self.calc_shapvalues(data)

        # move elsewhere?
        if self.istree:
            # 0 positive, 1 negative which do we want?
            # now postivie is anomaly -> make uniform with other models
            self.Svals = self.Svals[:,:,1]

        if plottype == "summary":
            shap.summary_plot(self.Svals, data,feature_names=self.vec.get_feature_names_out(), max_display=16)
        # get featurenames!
        elif plottype == "bar":
            shap.plots.bar(self.Svals)


    #saving to file how ?