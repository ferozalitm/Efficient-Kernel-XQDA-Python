#######################################################################################################
# Author: T M Feroz Ali
# Date: 27-Nov-2021
#
# This is the code for our paper:
# "Cross-View Kernel Similarity Metric Learning Using Pairwise Constraints for Person Re-identification"
# T M Feroz Ali, Subhasis Chaudhuri, Indian Conference on Computer Vision, Graphics and Image Processing
# ICVGIP-20-21
#
# If you find it useful, please kindly consider citing our work:
# @article{ali2019cross,
#   title={Cross-View Kernel Similarity Metric Learning Using Pairwise Constraints for Person Re-identification},
#   author={T M Feroz Ali and Subhasis Chaudhuri},
#   journal={arXiv preprint arXiv:1909.11316},
#   year={2019}
# }

#######################################################################################################
import os
import numpy as np
from Exp import Exp
from EK_XQDA import kernel_functions, EK_XQDA

#######################################################################################################
if __name__ == '__main__':
    print("Starting EK_XQDA person reID experiment:")

    # getting current directory
    directry = os.getcwd()
    print("directory: ", directry)

    ###################################################################################################
    # Configure database,
    # Set the database: 'CUHK01M1' (single-shot settings), 'CUHK01M2' (multi-shot settings)
    datasetName = 'CUHK01M2'
    print("datasetName: ", datasetName)
    db = Exp.Database(directry, datasetName)

    ###################################################################################################
    # configuration of features.
    featurenum = 4
    usefeature = np.zeros((featurenum,1))
    usefeature[0] = 1 # GOG_RGB
    usefeature[1] = 1 # GOG_Lab
    usefeature[2] = 1 # GOG_HSV
    usefeature[3] = 1 # GOG_nRnG
    parFeat = Exp.Feature(directry, featurenum, usefeature)

    ####################################################################################################
    # configuration of experiment
    exp = Exp.Experiment(db, sets_total=10)
    exp.load_features_all(db,parFeat)

    ####################################################################################################
    # Exp
    CMCs = np.zeros((exp.sets_total, db.numperson_gallary))

    for set_num in range(exp.sets_total):
        exp.set_num = set_num
        print('----------------------------------------------------------------------------------------------------')
        print('set_num = ', exp.set_num)
        print('----------------------------------------------------------------------------------------------------')

        # Load training data
        exp.train_or_test = 1
        exp.extract_feature_cell_from_all(parFeat)
        exp.apply_normalization(parFeat)
        exp.conc_feature_cell(parFeat)
        exp.extract_CamA_CamB_trainData()

        # Load test data
        exp.train_or_test = 2
        exp.extract_feature_cell_from_all(parFeat)
        exp.apply_normalization(parFeat)
        exp.conc_feature_cell(parFeat)
        exp.extract_CamA_CamB_testData()

        # Using only one image of each person while testing, if single-shot-setting)
        if (db.isSingleShotTesting == True):
            exp.camA_testData = exp.camA_testData[::2,:]
            exp.camA_testLabels = exp.camA_testLabels[::2]

        # Train EK_XQDA metric learning
        X = np.vstack((exp.camB_trainData, exp.camA_trainData))
        K,K_a,mu = kernel_functions.RBF_kernel(np.vstack((exp.camA_trainData, exp.camB_trainData)), exp.camA_testData)
        K_b = kernel_functions.RBF_kernel2(np.vstack((exp.camA_trainData, exp.camB_trainData)), exp.camB_testData, mu)
        K = (K+K.T)/2
        Theta,Gamma = EK_XQDA.EK_XQDA(K, exp.camA_trainLabels, exp.camB_trainLabels)
        Gamma_psd = EK_XQDA.projectPSD((Gamma+Gamma.T)/2)

        # Testing
        if(db.isSingleShotTesting == True):
            # singleshot matching for CUHK01
            scores = EK_XQDA.mahDist(Gamma_psd, K_a.T.dot(Theta), K_b.T.dot(Theta))
        else:
            # multishot matching for CUHK01
            exp.camA_testLabels = exp.camA_testLabels[::2]
            exp.camB_testLabels = exp.camB_testLabels[::2]
            scores1 = EK_XQDA.mahDist(Gamma_psd, K_a[:, 0::2].T.dot(Theta), K_b[:, 0::2].T.dot(Theta))
            scores2 = EK_XQDA.mahDist(Gamma_psd, K_a[:, 1::2].T.dot(Theta), K_b[:, 0::2].T.dot(Theta))
            scores3 = EK_XQDA.mahDist(Gamma_psd, K_a[:, 0::2].T.dot(Theta), K_b[:, 1::2].T.dot(Theta))
            scores4 = EK_XQDA.mahDist(Gamma_psd, K_a[:, 1::2].T.dot(Theta), K_b[:, 1::2].T.dot(Theta))
            scores = scores1 + scores2 + scores3 + scores4

        CMC = exp.calc_CMC(scores)
        CMCs[exp.set_num, :] = CMC.ravel()
        print(' Rank1, Rank5, Rank10, Rank15, Rank20')
        print("%5.2f%%, %5.2f%%, %5.2f%%, %5.2f%%, %5.2f%%" % tuple(CMC[np.array([0, 4, 9, 14, 19])]))

    print('----------------------------------------------------------------------------------------------------')
    print('  Mean Result')
    print('----------------------------------------------------------------------------------------------------')
    CMCmean = np.mean(CMCs[0:exp.sets_total, :], axis=0)
    print(' Rank1,  Rank5, Rank10, Rank15, Rank20')
    print("%5.2f%%, %5.2f%%, %5.2f%%, %5.2f%%, %5.2f%%" % tuple(CMCmean[np.array([0, 4, 9, 14, 19])]))

