from os.path import join as pjoin
import scipy.io as sio
import numpy as np
import h5py

class Experiment:
    def __init__(self, db, sets_total=10):
        mat_contents = sio.loadmat(db.DBfile)
        self.allimagenums = mat_contents['allimagenames'].shape[0]
        self.traininds_set = mat_contents['traininds_set']
        self.testinds_set = mat_contents['testinds_set']
        self.trainlabels_set = mat_contents['trainlabels_set']
        self.testlabels_set = mat_contents['testlabels_set']
        self.traincamIDs_set = mat_contents['traincamIDs_set']
        self.testcamIDs_set = mat_contents['testcamIDs_set']
        self.trainimagenames_set = mat_contents['trainimagenames_set']
        self.testimagenames_set = mat_contents['testimagenames_set']
        self.set_num = None
        self.sets_total = sets_total
        self.feature_cell_all = []
        self.feature_cell = []
        self.feature = []
        self.train_or_test = None
        self.mean_cell =[]
        self.camA_trainData = None
        self.camB_trainData = None
        self.camA_trainLabels = None
        self.camB_trainLabels = None
        self.camA_testData = None
        self.camB_testData = None
        self.camA_testLabels = None
        self.camB_testLabels = None

    def load_features_all(self, db, parFeat):
        print('*** load all extracted features ***')

        # Creating empty nested lists to store the features for the full dataset samples
        self.feature_cell_all = []
        for i in range(parFeat.featurenum):
            self.feature_cell_all.append([])

        for f in range(parFeat.featurenum):
            if parFeat.usefeature[f] == 1:
                print('feature = ', str(f), '[', parFeat.featureConf[f].name, ']')
                name1 = "feature_all_" + parFeat.featureConf[f].name
                name2 = db.databaseName + "_" + name1 + ".mat"
                name =  pjoin(parFeat.featuredirname, name2)
                print(name)
                with h5py.File(name, 'r') as matfile:
                    feature_all = np.array(matfile['feature_all'])
                self.feature_cell_all[f] = feature_all.transpose()

    def extract_feature_cell_from_all(self, parFeat):
        # extract feature cells for particular training / test division

        self.feature_cell = []
        for i in range(parFeat.featurenum):
            self.feature_cell.append([])

        if self.train_or_test == 1:
            numimages_train = self.traininds_set[self.set_num][0].shape[0]
            for f in range(parFeat.featurenum):
                if parFeat.usefeature[f] == 1:
                    self.feature_cell[f] = np.zeros((numimages_train, self.feature_cell_all[f].shape[1]))
                    for ind in range(numimages_train):
                        idx = self.traininds_set[self.set_num][0][ind]
                        idx = idx - 1           #Python idx is one less than MATLAB idx
                        self.feature_cell[f][ind, :] = self.feature_cell_all[f][idx, :]
            del numimages_train

        if self.train_or_test == 2:
            numimages_test = self.testinds_set[self.set_num][0].shape[0]
            for f in range(parFeat.featurenum):
                if parFeat.usefeature[f] == 1:
                    self.feature_cell[f] = np.zeros((numimages_test, self.feature_cell_all[f].shape[1]))
                    for ind in range(numimages_test):
                        #print(ind)
                        idx = self.testinds_set[self.set_num][0][ind]
                        idx = idx - 1           #Python idx is one less than MATLAB idx
                        self.feature_cell[f][ind, :] = self.feature_cell_all[f][idx, :]

    def apply_normalization(self, parFeat):
        if self.train_or_test == 1:
            for i in range(parFeat.featurenum):
                self.mean_cell.append([])

        for f in range(parFeat.featurenum):
            if parFeat.usefeature[f] == 1:
                X = self.feature_cell[f] # X: feature vectors of feature type f.Size: [datanum, feature dimension]

            if self.train_or_test == 1: # training data
                meanX = X.mean(axis=0) # meanX - - mean vector of features
                self.mean_cell[f] = meanX

            if self.train_or_test == 2: # test data
                meanX = self.mean_cell[f]

            Y = X - np.tile(meanX, (X.shape[0], 1)) # Mean removal
            for dnum in range(X.shape[0]):
                Y[dnum,:] = Y[dnum,:]/np.linalg.norm(Y[dnum,:], ord=2)

            self.feature_cell[f] = Y

    def conc_feature_cell(self, parFeat):
        isfirst = 1
        for f in range(parFeat.featurenum):
            if parFeat.usefeature[f] == 1:
                if isfirst == 1:
                    self.feature = self.feature_cell[f]
                    isfirst = 0
                else:
                    feature2 = self.feature_cell[f]
                    self.feature = np.hstack((self.feature, feature2))

        self.feature = np.float64(self.feature)

    def extract_CamA_CamB_trainData(self):
        camIDs = self.traincamIDs_set[self.set_num][0]
        camIDs = np.squeeze(camIDs)
        self.camA_trainData = self.feature[camIDs == 1, :]
        self.camB_trainData = self.feature[camIDs == 2, :]
        labels = self.trainlabels_set[self.set_num][0]
        labels = np.squeeze(labels)
        self.camA_trainLabels = labels[camIDs == 1]
        self.camB_trainLabels = labels[camIDs == 2]

    def extract_CamA_CamB_testData(self):
        camIDs = self.testcamIDs_set[self.set_num][0]
        camIDs = np.squeeze(camIDs)
        self.camA_testData = self.feature[camIDs == 1, :]
        self.camB_testData = self.feature[camIDs == 2, :]
        labels = self.testlabels_set[self.set_num][0]
        labels = np.squeeze(labels)
        self.camA_testLabels = labels[camIDs == 1]
        self.camB_testLabels = labels[camIDs == 2]

    def calc_CMC(self, scores):
        CMC = np.zeros((len(self.camB_testLabels), 1))
        for p in range(len(self.camA_testLabels)):
            score = scores[p, :]
            ind = np.argsort(score)
            correctind = (self.camB_testLabels[ind] == self.camA_testLabels[p]).nonzero()[0].__int__()
            CMC[correctind:] = CMC[correctind:] + 1

        CMC = 100 * CMC / len(self.camA_testLabels)
        return CMC

class Database:
    def __init__(self, directry, databaseName):
        self.numperson_train = 0
        self.numperson_probe = 0
        self.numperson_gallary = 0
        self.databaseName = databaseName
        self.DBfile = ""
        self.isSingleShotTesting = None
        self.set_database(directry)

    def set_database(self, directry):
        if self.databaseName == 'CUHK01M1':
            self.numperson_train = 486
            self.numperson_probe = 485
            self.numperson_gallary = 485
            self.databaseName = 'CUHK01'
            self.DBfile = pjoin(directry, 'DB', 'CUHK01M1.mat')
            self.isSingleShotTesting = True
        elif self.databaseName == 'CUHK01M2':
            self.numperson_train = 485
            self.numperson_probe = 486
            self.numperson_gallary = 486
            self.databaseName = 'CUHK01'
            self.DBfile = pjoin(directry, 'DB', 'CUHK01M2.mat')
            self.isSingleShotTesting = False    # Multi-shot testing
        else:
            print("Invalid database name")
            raise ValueError



class Feature:
    def __init__(self, directry, featurenum, usefeature):
        self.featurenum = featurenum
        self.usefeature = usefeature
        self.featureConf = []
        self.featuredirname = pjoin(directry, 'Features')
        self.set_feature_names()

    def set_feature_names(self):
        "set default parameter"

        def numbers_to_strings(argument):
            switcher = {
                0: "yMthetaRGB",
                1: "yMthetaLab",
                2: "yMthetaHSV",
                3: "yMthetanRnG"
            }
            err_str = "lf_type = " + str(argument) + " is not defined"
            #return switcher.get(argument, "lf_type not defined")
            return switcher.get(argument, err_str)

        for lf_type in range(self.featurenum):
            lf_name = "GOG" + numbers_to_strings(lf_type)
            self.featureConf.append(self.Feature_name(lf_name))

    class Feature_name:
        def __init__(self, name):
            self.name = name







