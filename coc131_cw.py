import numpy as np
from PIL import Image
from sklearn.datasets import load_files
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split, KFold, StratifiedKFold, cross_val_score
from scipy.stats import ttest_rel
from sklearn.manifold import LocallyLinearEmbedding

import pickle
# later remove pickle


# Please write the optimal hyperparameter values you obtain in the global variable 'optimal_hyperparm' below. This
# variable should contain the values when I look at your submission. I should not have to run your code to populate this
# variable.
optimal_hyperparam = {}

class COC131:
    #samples
    x = None
    #labels
    y = None

    standardised_x = None

    best_params = None

    with open('dataset.pkl','rb') as f:
        x, y = pickle.load(f)

    #static function for separating the individual file extraction logic
    @staticmethod
    def _process_file(filename):
        res1 = np.zeros(1)
        res2 = ''
        if filename is None:
            return res1, res2
        img = Image.open(filename)

        img = img.resize((32, 32), Image.LANCZOS)
        arr = np.array(img, dtype=float)
        res1 = arr.flatten(order='C')
        clean = filename.replace('\\', '/')
        res2  = clean.split('/')[-2]
        return res1, res2
    
    def q1(self, filename=None):
        """
        This function should be used to load the data. To speed-up processing in later steps, lower resolution of the
        image to 32*32. The folder names in the root directory of the dataset are the class names. After loading the
        dataset, you should save it into an instance variable self.x (for samples) and self.y (for labels). Both self.x
        and self.y should be numpy arrays of dtype float.

        :param filename: this is the name of an actual random image in the dataset. You don't need this to load the
        dataset. This is used by me for testing your implementation.
        :return res1: a one-dimensional numpy array containing the flattened low-resolution image in file 'filename'.
        Flatten the image in the row major order. The dtype for the array should be float.
        :return res2: a string containing the class name for the image in file 'filename'. This string should be same as
        one of the folder names in the originally shared dataset.
        """
        #check if key variables have been initialised already. 
        if self.x is not None and self.y is not None:
            print("Skipping complete data processing step")
            return self._process_file(filename=filename)
        
        #if x and y have not been initialised, then go through the whole process
        print("Commencing complete data processing")
        data = load_files('EuroSAT_RGB', load_content=False)
        file_paths, label_indices = data['filenames'], data['target']

        samples = []
        for fp in file_paths:
            img = Image.open(fp).resize((32, 32), Image.LANCZOS)
            arr = np.array(img, dtype=float)
            samples.append(arr.flatten(order='C'))

        X = np.stack(samples)
        y = label_indices.astype(float)    
        with open('dataset.pkl','wb') as f:
            pickle.dump((X, y), f)
        self.x, self.y = X, y

        return self._process_file(filename=filename)


    def q2(self, inp):
        """
        This function should compute the standardized data from a given 'inp' data. The function should work for a
        dataset with any number of features.

        :param inp: an array from which the standardized data is to be computed.
        :return res2: a numpy array containing the standardized data with standard deviation of 2.5. The array should
        have the same dimensions as the original data
        :return res1: sklearn object used for standardization.
        """

        standardScaler = StandardScaler()
        data = standardScaler.fit_transform(inp)

        res1 = standardScaler
        res2 = data * 2.5
        self.standardised_x = res2
        return res1, res2

    def q3(self, test_size=None, pre_split_data=None, hyperparam=None):
        """
        This function should build a MLP Classifier using the dataset loaded in function 'q1' and evaluate model
        performance. You can assume that the function 'q1' has been called prior to calling this function. This function
        should support hyperparameter optimizations.

        :param test_size: the proportion of the dataset that should be reserved for testing. This should be a fraction
        between 0 and 1.
        :param pre_split_data: Can be used to provide data already split into training and testing.
        :param hyperparam: hyperparameter values to be tested during hyperparameter optimization.
        :return: The function should return 1 model object and 3 numpy arrays which contain the loss, training accuracy
        and testing accuracy after each training iteration for the best model you found.
        """
        res1 = object()
        res2 = np.zeros(1)
        res3 = np.zeros(1)
        res4 = np.zeros(1)

        if test_size is None:
            test_size = 0.2
        if pre_split_data is None:
            x_train, x_test, y_train, y_test = train_test_split(self.standardised_x, self.y, test_size=test_size)
        else:
            x_train, x_test, y_train, y_test = pre_split_data 
        if hyperparam is None:
            hyperparam = {
                        'hidden_layer_sizes': (200,),
                        'activation'       : 'relu',
                        'solver'           : 'adam',
                        'alpha'            : 1e-4,
                        'learning_rate_init': 1e-3,
                        'batch_size'       : 'auto',
                        'max_iter'         : 50
                        }
        n_epochs = int(hyperparam.pop('max_iter', 50)) 

        res1 = MLPClassifier(
        warm_start=True,      
        max_iter=1,                   
        random_state=0,
        **hyperparam)

        res2, res3, res4 = [], [], []    
        for _ in range(n_epochs):                   
            res1.fit(x_train, y_train)                
            res2.append(res1.loss_)
            res3.append(res1.score(x_train, y_train))
            res4.append(res1.score(x_test,  y_test))

        return res1, res2, res3, res4

    def q4(self):
        """
        This function should study the impact of alpha on the performance and parameters of the model. For each value of
        alpha in the list below, train a separate MLPClassifier from scratch. Other hyperparameters for the model can
        be set to the best values you found in 'q3'. You can assume that the function 'q1' has been called
        prior to calling this function.

        :return: res should be the data you visualized.
        """
        X_train, X_test, y_train, y_test = train_test_split(
            self.standardised_x, self.y, test_size=0.2, random_state=0
        )
        base_params = dict(self.best_params)
        base_params.pop('alpha', None)
        alpha_values = [0, 0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1, 2, 5, 10, 50, 100]

        results = []
        for a in alpha_values:
            clf = MLPClassifier(random_state=0, **base_params, alpha=a)
            clf.fit(X_train, y_train)
            train_acc = clf.score(X_train, y_train)
            test_acc  = clf.score(X_test,  y_test)

            results.append((a, train_acc, test_acc))
        res = np.array(results, dtype=float)
        return res

    def q5(self):
        """
        This function should perform hypothesis testing to study the impact of using CV with and without Stratification
        on the performance of MLPClassifier. Set other model hyperparameters to the best values obtained in the previous
        questions. Use 5-fold cross validation for this question. You can assume that the function 'q1' has been called
        prior to calling this function.

        :return: The function should return 4 items - the final testing accuracy for both methods of CV, p-value of the
        test and a string representing the result of hypothesis testing. The string can have only two possible values -
        'Splitting method impacted performance' and 'Splitting method had no effect'.
        """

        clf = MLPClassifier(random_state=0, **self.best_params)
        #5 fold without stratification
        kf = KFold(n_splits=5, shuffle=True, random_state=0)
        scores_kf = cross_val_score(clf, self.,standardised_x self.y, cv=kf)
        #with stratification
        skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=0)
        scores_skf = cross_val_score(clf, self.standardised_x, self.y, cv=skf)

        res1 = np.mean(scores_kf)
        res2 = np.mean(scores_skf)

        _, res3 = ttest_rel(scores_kf, scores_skf)
        res4 = "Splitting method impacted performance" if res3 < 0.01 else "Splitting method had no effect"
        
        return res1, res2, res3, res4

    def q6(self):
        """
        This function should perform unsupervised learning using LocallyLinearEmbedding in Sklearn. You can assume that
        the function 'q1' has been called prior to calling this function.

        :return: The function should return the data you visualize.
        """
        lle = LocallyLinearEmbedding(n_neighbors=100, n_components=2, random_state=0)
        res = lle.fit_transform(self.standardised_x)
        return res



# coc131 = COC131()
# print(coc131.q1())
# print(coc131.q1("EuroSAT_RGB\\AnnualCrop\\AnnualCrop_1.jpg"))