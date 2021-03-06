import os

import xgboost  as xgb
from xgboost import plot_importance
from xgboost import plot_tree


class me():
    '''Implementation of the xgboost-based matching ensemble (ME) model
    '''
    def __init__(self, max_depth=6, num_boost_rounds=10, negPosRatio=1.0, min_child_weight=6, learning_rate=0.1):
        print('init ME model')
        self.model = None
        self.num_boost_rounds = num_boost_rounds

        self.params = {
            'booster': 'gbtree',
            'objective': 'binary:logistic',
            'learning_rate':learning_rate,
            'max_depth':max_depth,
            'min_child_weight':min_child_weight, 
            'gamma':0.3, 
            'subsample':0.7,
            'colsample_bytree':0.7,   
            'scale_pos_weight':negPosRatio, 
            'reg_alpha':0,
            'reg_lambda':1,
            'tree_method': 'exact',
            'nthread':-1,
            'seed':0,
            'silent':1,
        }       


    def train(self, X, y):          

        num_boost_rounds = self.num_boost_rounds
        
        negPosRatio = float(sum(y==0)) / sum(y==1)
        

        self.params['scale_pos_weight']=negPosRatio

        plst = self.params.items()
       

        dtrain = xgb.DMatrix(X, y)

        model = xgb.train(plst, dtrain, num_boost_rounds)

        self.model = model

        return model

   
    def inference(self, testx):

        model = self.model

        dtest = xgb.DMatrix(testx)
    
        ans = model.predict(dtest)

        return ans




    def test(self, testx, testy, rstpath='./', surname=''):
        '''
        params
        ---
            testx: feature vector
            testy: label
            rstpath: where to store
            surname: file name
        '''
        if not os.path.exists(rstpath):
            os.makedirs(rstpath)

        model = self.model

        dtest = xgb.DMatrix(testx)
        # test
        ans = model.predict(dtest)

        

        # ACC
        cnt1 = 0
        cnt2 = 0
        for i in range(len(testy)):
            if round(ans[i]) == testy[i]:
                cnt1 += 1
            else:
                cnt2 += 1

        classificationACC = 100 * cnt1 / (cnt1 + cnt2)

        print("\n[%s] Classification Accuracy on the testset: %.10f %% / correct:%d, wrong: %d, total: %d. " % (surname, classificationACC, cnt1, cnt2, cnt1+cnt2))


        with open(os.path.join(rstpath, 'rst_classification_'+surname+'.txt'), 'w') as fid:
            fid.writelines("[%s] Classification Accuracy (on the test data set): %.10f %% correct:%d, wrong: %d, total: %d.\n" % (surname, classificationACC, cnt1, cnt2, cnt1+cnt2))

        # store the EER, ROC, Genuine-Impostor Curve:
        self.genScores_GI_ROC(ans, testy, rstpath=rstpath, scorefilename='scores_'+surname+'.txt', ana_dir='ana_'+surname, run_roc=True)


        return cnt1, cnt1+cnt2, ans
        


    def genScores_GI_ROC(self, predict=None, y=None, rstpath='./', scorefilename='scores.txt', ana_dir='ana', run_roc=False):
        '''store fused matching scores (generated by the ME model)
        params
        ---
        predict: outputs of the ME model (fused matching scores)
        y: labels
        rstpath: where to store the fused matching scores
        scorefilename: file name of the fused matching scores
        run_roc: whether execute the ROC script
        ana_dir: folder name for storing the ROC-related results
        '''


        assert predict is not None
        assert y is not None

        if not os.path.exists(rstpath):
            os.makedirs(rstpath)

        pathScore = os.path.join(rstpath, scorefilename)

        with open(pathScore, 'w') as fid:
            for i in range(len(y)):
                if y[i] == 1:
                    label = 1
                else:
                    label = -1
                fid.writelines("%.6f\t%d\n" % (predict[i], label))
                # fid.writelines("%.3f\t%d\n" % (predict[i], label))

        if run_roc:
            os.system('python getGI.py' + ' ' + pathScore + ' ' + ana_dir)
            os.system('python getEER.py' + ' ' + pathScore + ' ' + ana_dir)

    def getModel(self):
        return self.model