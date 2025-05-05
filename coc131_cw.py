import numpy as np
from PIL import Image
from sklearn.datasets import load_files
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split, KFold, StratifiedKFold, cross_val_score
from scipy.stats import ttest_rel
from sklearn.manifold import LocallyLinearEmbedding

# Please write the optimal hyperparameter values you obtain in the global variable 'optimal_hyperparm' below. This
# variable should contain the values when I look at your submission. I should not have to run your code to populate this
# variable.
optimal_hyperparam = {'hidden_layer_sizes': (100, 100),
 'solver': 'sgd',
 'alpha': 0.5,
 'learning_rate_init': 0.001,
 'batch_size': 64,
 'momentum': 0.8,
 'max_iter': 50}

class COC131:
    #samples
    x = None
    #labels
    y = None
    #store standardised samples
    standardised_x = None
    #store best params found
    best_params = None

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
        #logic for loading dataset
        data = load_files('EuroSAT_RGB', load_content=False)
        file_paths, label_indices = data['filenames'], data['target']

        samples = []
        for fp in file_paths:
            img = Image.open(fp).resize((32, 32), Image.LANCZOS)
            arr = np.array(img, dtype=float)
            samples.append(arr.flatten(order='C'))

        X = np.stack(samples)
        y = label_indices.astype(float)    
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

        res1 = StandardScaler()
        data = res1.fit_transform(inp)
        #since it is initially standardised with (mean=0, std=1), multiplying by 2.5 gives std=2.5
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

        #hyperparameter optimisation is done outside of the function, so that I can visualise each attempt
        res1 = object()
        res2 = np.zeros(1)
        res3 = np.zeros(1)
        res4 = np.zeros(1)

        #using if statements for setting defaults
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

        #warm start is used to maintain the weights learned in the last run.
        #this allows the mlp to be run for 1 epoch repeatedly
        res1 = MLPClassifier(
        warm_start=True,    
        max_iter=1,                   
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

        x_train, x_test, y_train, y_test = train_test_split(
            self.standardised_x, self.y, test_size=0.2, random_state=0
        )
        #using the best params found, but replacing alpha with the value to be tested
        base_params = dict(self.best_params)
        base_params.pop('alpha', None)
        base_params['max_iter'] = 50
        alpha_values = [0, 0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1, 2, 5, 10, 50, 100]

        results = []
        for a in alpha_values:
            clf = MLPClassifier(**base_params, alpha=a)
            clf.fit(x_train, y_train)
            train_acc = clf.score(x_train, y_train)
            test_acc  = clf.score(x_test,  y_test)
            results.append((a, train_acc, test_acc))

        #return an array of tuples with training and testing accuracy for each alpha value
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

        classifier = MLPClassifier(**self.best_params)
        #5 fold without stratification
        kf = KFold(n_splits=5, shuffle=True)
        kf_score = cross_val_score(classifier, self.standardised_x, self.y, cv=kf)
        #with stratification
        skf = StratifiedKFold(n_splits=5, shuffle=True)
        skf_score = cross_val_score(classifier, self.standardised_x, self.y, cv=skf)

        res1 = np.mean(kf_score)
        res2 = np.mean(skf_score)

        _, res3 = ttest_rel(kf_score, skf_score)
        # < 1% chance of observing fold‐score difference under null hypothesis
        res4 = "Splitting method impacted performance" if res3 < 0.01 else "Splitting method had no effect"
        
        return res1, res2, res3, res4

    def q6(self):
        """
        This function should perform unsupervised learning using LocallyLinearEmbedding in Sklearn. You can assume that
        the function 'q1' has been called prior to calling this function.

        :return: The function should return the data you visualize.
        """

        #projecting data onto 2D plane
        lle = LocallyLinearEmbedding(n_neighbors=100, n_components=2)
        res = lle.fit_transform(self.standardised_x)
        return res