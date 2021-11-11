import os
import numpy as np 

import matplotlib.pyplot as plt
import seaborn as sns

import xgboost  as xgb
from xgboost import plot_importance
from xgboost import plot_tree


class same():
    '''Implementation of the sharpness-aware matching ensemble (SaME) model
    '''
    def __init__(self, max_depth=6, num_boost_rounds=10, negPosRatio=1.0, min_child_weight=6, learning_rate=0.1, th1=0, th2=1.0):
        print('init SaME model')
        self.models = None
        self.num_boost_rounds = num_boost_rounds
        self.th1=th1
        self.th2=th2

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
        

    def train(self, trainx, trainy):
        th1 = self.th1  
        th2 = self.th2  
        trainx_sms = trainx[:, -1]
        models = []

        


        thranges = {'low':(0,th1),'mid':(th1, th2),'high':(th2, 1.0+1e-6)}

        for th in thranges:
            (L, H) = thranges[th]
            ids1 = trainx_sms < H
            ids2 = trainx_sms >= L
            ids = ids1 & ids2

            print('=> sms range: [', L, H, ')')
            
            assert(sum(ids) > 0)

        
            
            X = trainx[ids, :-1]
            y = trainy[ids]

            
            negPosRatio = float(sum(y==0)) / sum(y==1)
            

            self.params['scale_pos_weight']=negPosRatio           
            
            num_boost_rounds = self.num_boost_rounds

            plst = self.params.items()
         

            dtrain = xgb.DMatrix(X, y)

            model = xgb.train(plst, dtrain, num_boost_rounds)            
            
            models.append(model)
        
        self.models = models
        return models


    def inference(self, x):
        '''
        x: input 1-D vector
        '''
        assert len(self.models) == 3
        assert len(x.shape) == 1

        th1 = self.th1
        th2 = self.th2

        sms = x[:,-1]
        if sms < th1:
            model = self.models[0]
        elif sms < th2:
            model = self.models[1]
        else:
            model = self.models[2]
    

        dtest = xgb.DMatrix(x[:-1])
    
        ans = model.predict(dtest)

        return ans       



    def testTheGivenModel(self, testx, testy, model, rstpath='./', surname=''):
        '''
        params
        ---
            testx: feature vector (w/o sms)
            testy: label
            model: the given model
            rstpath: where to store
            surname: file name
        outputs
        ---
            ans: y
            cnt1: number of corrected predictions
            cnt1+cnt2: total number of predictions (len(y))

        '''
        assert testy is not None and len(testy) > 0

        if not os.path.exists(rstpath):
            os.makedirs(rstpath)

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

        print("[%s] Classification Accuracy on the testset: %.10f %% / correct:%d, wrong: %d, total: %d. " % (surname, classificationACC, cnt1, cnt2, cnt1+cnt2))


        with open(os.path.join(rstpath, 'rst_classification_'+surname+'.txt'), 'w') as fid:
            fid.writelines("[%s] Classification Accuracy (on the test data set): %.10f %% correct:%d, wrong: %d, total: %d.\n" % (surname, classificationACC, cnt1, cnt2, cnt1+cnt2))

        # store the EER, ROC, Genuine-Impostor Curve:
        self.genScores_GI_ROC(ans, testy, rstpath=rstpath, scorefilename='scores_'+surname+'.txt', ana_dir='ana_'+surname, run_roc=True)

        return ans, cnt1, cnt1+cnt2






    def test(self, testx, testy, rstpath='./', surname='same'):
        '''test the model on testing set
        params
        ---
            testx: matching scores (SMS is the last element)
            testy: labels
        '''

        assert len(self.models) == 3

        if not os.path.exists(rstpath):
            os.makedirs(rstpath)

        th1 = self.th1
        th2 = self.th2

        thranges = {'low':(0,th1),'mid':(th1, th2),'high':(th2, 1.0+1e-6)}

        testsms = testx[:,-1]
        id1 =  testsms < thranges['low'][1]
        id2 =  (testsms < thranges['mid'][1]) & (testsms >= thranges['mid'][0])
        id3 =  (testsms < thranges['high'][1]) & (testsms >= thranges['high'][0])

     
        print('len(id1):', len(id1), 'len(id2):', len(id2), 'len(id3):', len(id3))

        X1 = testx[id1, :-1]
        X2 = testx[id2, :-1]
        X3 = testx[id3, :-1]

        y1 = testy[id1]
        y2 = testy[id2]
        y3 = testy[id3]
   

        Xs = [X1, X2, X3]
        ys = [y1, y2, y3]
   

        ps = None
        labels = None

        ncorrect = 0
        ntotal = 0

        for i in range(3):
            
            X = Xs[i]
            y = ys[i]

        

            model = self.models[i]
            
            pred, c, t = self.testTheGivenModel(X, y, model, rstpath, surname+'_range_%d'%(i+1))
            ncorrect += c
            ntotal += t

            
            if ps is None: # first one
                ps = pred
                labels =y
            else:
                ps = np.concatenate((ps, pred))
                labels = np.concatenate((labels, y))

        print(ps.shape)
     

        #
        sns.kdeplot(ps)
        plt.savefig(os.path.join(rstpath,'SaME_FusedMatchingScore.png'))
        plt.close()
        #

        classificationACC = 100 * ncorrect / ntotal
        
        print("\n[%s] Classification Accuracy on the testset: %.10f %% correct:%d, wrong: %d, total: %d. " % (surname, classificationACC, ncorrect, ntotal-ncorrect, ntotal))

        with open(os.path.join(rstpath, 'rst_classification_'+surname+'.txt'), 'w') as fid:
            fid.writelines("[%s] Classification Accuracy (on the test data set): %.10f %% correct:%d, wrong: %d, total: %d.\n" % (surname, classificationACC, ncorrect, ntotal-ncorrect, ntotal))

        self.genScores_GI_ROC(predict=ps, y=labels, rstpath=rstpath, scorefilename='scores_'+surname+'.txt', ana_dir='ana_'+surname, run_roc=True)
        
        return ps, classificationACC



    def genScores_GI_ROC(self, predict=None, y=None, rstpath='./', scorefilename='scores.txt', ana_dir='ana', run_roc=False):
        '''store fused matching scores (generated by the SaME model)
        params
        ---
        predict: outputs of the SaME model (fused matching scores)
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

    def getModels(self):
        return self.models