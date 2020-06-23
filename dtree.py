import numpy as np
from sklearn.tree import DecisionTreeClassifier, _tree

class DTree():
    def __init__(self, max_nFeats=1500, criterion='entropy', max_depth=20, ccp_alpha=0.001, randSeed=None, patience=20, verbose=True):
        self.verbose = verbose
        self.patience = patience
        np.random.seed(randSeed)
    
        # fringe constraints
        self.max_nFeats = max_nFeats
        self.fringeFeats = dict()
        self.fringeFeatsBest = None
        
        # sklearn dtree params
        self.criterion = criterion
        self.max_depth = max_depth
        self.ccp_alpha = ccp_alpha
        self.random_state = randSeed
        
        # dtree
        self.dtree = None
        self.dtreeBest = None
        self.nFeats = None
        
    def __trainInt__(self, data, labels):
        assert len(data) == len(labels)
        self.dtree = DecisionTreeClassifier(random_state=self.random_state, criterion=self.criterion, max_depth=self.max_depth, ccp_alpha=self.ccp_alpha)
        self.dtree.fit(data, labels)
    
    def train(self, trainData, trainLabels, validData, validLabels):        
        self.nFeats = len(trainData[0])
        assert len(validData[0]) == self.nFeats
        
        trainAccBest, valAccBest = None, -1.0
        iter = 0
        while True:
            self.__trainInt__(trainData, trainLabels)
            _, trainAcc = self.predict(trainData, trainLabels, False, False)
            _, valAcc = self.predict(validData, validLabels, False, False)
            if self.verbose:
                print('iter {}: {} / {}'.format(str(iter), str(trainAcc), str(valAcc)))
            if valAcc > valAccBest:
                self.dtreeBest = self.dtree
                self.fringeFeatsBest = self.fringeFeats.copy()
                valAccBest = valAcc
                trainAccBest = trainAcc
            if self.nFeats + len(self.fringeFeats) > self.max_nFeats: break
            
            initSize = len(self.fringeFeats)
            featSet = self.__fringeDetect__()
            for feat in featSet:
                if feat in self.fringeFeats: continue
                id1, id2, op = feat
                trainData = self.__dataAug__(trainData, id1, id2, op)
                validData = self.__dataAug__(validData, id1, id2, op)
                self.fringeFeats[feat] = self.nFeats + len(self.fringeFeats)            
            
            if len(self.fringeFeats) == initSize: break
            iter += 1
        return trainAccBest, valAccBest
    
    def __trainInt2__(self, trainData, trainLabels, validData, validLabels):
        self.dtree = None
        assert len(trainData) == len(trainLabels)
        assert len(validData) == len(validLabels)
        
        def eval(dt, data, labels):
            preds = dt.predict(data)
            return np.sum(np.array(preds)==np.array(labels)) / len(labels)
        
        dt0 = DecisionTreeClassifier(random_state=self.random_state, criterion=self.criterion, max_depth=self.max_depth)
        
        ccp_alphas = dt0.cost_complexity_pruning_path(trainData, trainLabels).ccp_alphas
        ccp_alphas = np.array_split(ccp_alphas, min(10, len(ccp_alphas)))
        ccp_alphas = [x.mean() for x in ccp_alphas]
        
        vaccBest = -1.0
        for ccp_alpha in ccp_alphas:
            dt = DecisionTreeClassifier(random_state=self.random_state, criterion=self.criterion, max_depth=self.max_depth, ccp_alpha=ccp_alpha)
            dt.fit(trainData, trainLabels)
            vacc = eval(dt, validData, validLabels)
            if vacc > vaccBest:
                self.dtree = dt
                self.ccp_alpha = ccp_alpha
                vaccBest = vacc
        assert self.dtree is not None

    
    def train2(self, trainData, trainLabels, validData, validLabels):
        self.nFeats = len(trainData[0])
        assert len(validData[0]) == self.nFeats
        
        trainAccBest, valAccBest = None, -1.0
        iter = 0
        patience = self.patience
        while True:
            self.__trainInt2__(trainData, trainLabels, validData, validLabels)
            _, trainAcc = self.predict(trainData, trainLabels, False, False)
            _, valAcc = self.predict(validData, validLabels, False, False)
            if self.verbose:
                print('iter {}: {} / {}'.format(str(iter), str(trainAcc), str(valAcc)))
            if valAcc > valAccBest:
                self.dtreeBest = self.dtree
                self.fringeFeatsBest = self.fringeFeats.copy()
                valAccBest = valAcc
                trainAccBest = trainAcc
                patience = self.patience
            if self.nFeats + len(self.fringeFeats) > self.max_nFeats: break
            
            initSize = len(self.fringeFeats)
            featSet = self.__fringeDetect__()
            for feat in featSet:
                if feat in self.fringeFeats: continue
                id1, id2, op = feat
                trainData = self.__dataAug__(trainData, id1, id2, op)
                validData = self.__dataAug__(validData, id1, id2, op)
                self.fringeFeats[feat] = self.nFeats + len(self.fringeFeats)            
            
            if len(self.fringeFeats) == initSize: break
            iter += 1
            patience -= 1
            if patience < 0: break
        return trainAccBest, valAccBest
        
        
    def predict(self, data, labels=None, aug=True, useBest=True):
        getAcc = lambda preds, labels: np.sum(np.array(preds)==np.array(labels)) / len(labels)
        
        if aug: data = self.transformData(data)
        
        dtree = self.dtreeBest if useBest else self.dtree
        assert len(data[0]) == dtree.tree_.n_features
        preds = dtree.predict(data)
        
        if labels is None: return preds
        acc = getAcc(preds, labels)
        return preds, acc
        
    def transformData(self, data, useBest=True):
        assert len(data[0]) == self.nFeats
        feats = self.__getFringeFeats__(useBest)
        for id1, id2, op in feats:
            data = self.__dataAug__(data, id1, id2, op)
        return data
        
    def dumpFeats(self, fn, useBest=True):
        feats = self.__getFringeFeats__(useBest)
        with open(fn, 'w') as fp:
            fp.write('id1,id2,op\n')
            for id1, id2, op in feats:
                fp.write('{},{},{}\n'.format(str(id1), str(id2), op))
            
        
    def __getFringeFeats__(self, useBest=True):
        feats = self.fringeFeatsBest if useBest else self.fringeFeats
        feats = list(feats.items())
        feats.sort(key = lambda k: k[1])
        return [x[0] for x in feats]
    
        
    def __fringeDetect__(self):
        dtree_ = self.dtree.tree_
        
        isLeaf = lambda nId: dtree_.feature[nId] == _tree.TREE_UNDEFINED
        getLeftChild = lambda nId: dtree_.children_left[nId]
        getRightChild = lambda nId: dtree_.children_right[nId]
        
        def getChild(nId, br):
            if (nId is None) or isLeaf(nId):
                return None
            if br == 0:
                return getLeftChild(nId)
            else:
                return getRightChild(nId)
            
        def isLeaf1(nId):
            if (nId is None) or not(isLeaf(nId)):
                return False
            return np.argmax(dtree_.value[nId]) == 1
            
        def isLeaf0(nId):
            if (nId is None) or not(isLeaf(nId)):
                return False
            return np.argmax(dtree_.value[nId]) == 0
            
            
        def nodeDetect(nId):
            if (nId is None) or isLeaf(nId):
                return None
                
            if isLeaf0(getChild(getChild(nId, 0), 1)) and isLeaf0(getChild(getChild(nId, 1), 0)) and (getChild(nId, 0) == getChild(nId, 1)):
                return getChild(nId, 0), 'x=y'
            
            if isLeaf0(getChild(getChild(nId, 0), 0)) and isLeaf0(getChild(getChild(nId, 1), 1)) and (getChild(nId, 0) == getChild(nId, 1)):
                return getChild(nId, 0), 'x^y'
            
            if isLeaf1(getChild(getChild(nId, 0), 0)):
                if isLeaf1(getChild(nId, 1)) and isLeaf0(getChild(getChild(nId, 0), 1)):
                    op = 'x|~y'
                elif isLeaf1(getChild(getChild(nId, 1), 1)) and (getChild(nId, 0) == getChild(nId, 1)):
                    op = 'x=y'
                else:
                    op = '~x&~y'
                return getChild(nId, 0), op
            
            if isLeaf1(getChild(getChild(nId, 0), 1)):
                if isLeaf1(getChild(nId, 1)) and isLeaf0(getChild(getChild(nId, 0), 0)):
                    op = 'x|y'
                elif isLeaf1(getChild(getChild(nId, 1), 0)) and (getChild(nId, 0) == getChild(nId, 1)):
                    op = 'x^y'
                else:
                    op = '~x&y'
                return getChild(nId, 0), op
                
            if isLeaf1(getChild(getChild(nId, 1), 0)):
                if isLeaf1(getChild(nId, 0)) and isLeaf0(getChild(getChild(nId, 1), 1)):
                    op = '~x|~y'
                else:
                    op = 'x&~y'
                return getChild(nId, 1), op
                
            if isLeaf1(getChild(getChild(nId, 1), 1)):
                if isLeaf1(getChild(nId, 0)) and isLeaf0(getChild(getChild(nId, 1), 0)):
                    op = '~x|y'
                else:
                    op = 'x&y'
                return getChild(nId, 1), op
            
            return None
            
        ret = set()
        for nId in range(dtree_.node_count):
            feat_ = nodeDetect(nId)
            if feat_ is None: continue
            x = dtree_.feature[nId]
            y = dtree_.feature[feat_[0]]
            op = feat_[1]
            feat = (x, y, op)
            ret.add(feat)
        return ret
        
        
    def __dataAug__(self, data, id1, id2, op):
        d1 = data.transpose()[id1]
        d2 = data.transpose()[id2]
        
        AND = lambda x, y: x & y
        OR  = lambda x, y: x | y
        XOR = lambda x, y: x ^ y
        NOT = lambda x: 1 ^ x
        
        if op == 'x^y':
            res = XOR(d1, d2)
        elif op == 'x=y':
            res = NOT(XOR(d1, d2))
        else:
            assert ('&' in op) or ('|' in op)
            if '~x' in op: d1 = NOT(d1)
            if '~y' in op: d2 = NOT(d2)
            if '&' in op: res = AND(d1, d2)
            else: res = OR(d1, d2)
            
        res = np.array([res])
        data = np.concatenate((data, res.T), axis=1)
        return data
        
    def toBlif(self, fn, useBest=True):
        getFeatName = lambda x: 'x_' + str(x)
        getNodeName = lambda x: 'n_' + str(x)
        
        def fringeFeatsExtract(fLst):
            feats = self.__getFringeFeats__(useBest)
            n = self.nFeats
            for id1, id2, op in feats:
                names = [getFeatName(id1), getFeatName(id2), getFeatName(n)]
                if op == 'x^y':
                    pats = ['10 1', '01 1']
                elif op == 'x=y':
                    pats = ['00 1', '11 0']
                elif '&' in op:
                    b0 = '0' if ('~x' in op) else '1'
                    b1 = '0' if ('~y' in op) else '1'
                    pats = [b0 + b1 + ' 1']
                else:
                    assert '|' in op
                    b0 = '1' if ('~x' in op) else '0'
                    b1 = '1' if ('~y' in op) else '0'
                    pats = [b0 + b1 + ' 0']
                fLst.append((names, pats))
                n += 1
            
        def skTreeExtract(dtree_, nId, nLst):
            if dtree_.feature[nId] != _tree.TREE_UNDEFINED: # decision node
                skTreeExtract(dtree_, dtree_.children_left[nId], nLst)    # 0-branch
                skTreeExtract(dtree_, dtree_.children_right[nId], nLst)   # 1-branch
                nNd = getNodeName(nId)
                nC0 = getNodeName(dtree_.children_left[nId])
                nC1 = getNodeName(dtree_.children_right[nId])
                nCtrl = getFeatName(dtree_.feature[nId])
                names = [nC0, nC1, nCtrl, nNd]
                pats = ['-11 1', '1-0 1']
            else: # leaf node
                names = [getNodeName(nId)]
                val = np.argmax(dtree_.value[nId])
                pats = ['1'] if (val == 1) else []
            nLst.append((names, pats))  
            
        dtree = self.dtreeBest if useBest else self.dtree
        dtree_ = dtree.tree_
        
        nLst = []
        fringeFeatsExtract(nLst)
        skTreeExtract(dtree_, 0, nLst)
        
        fp = open(fn, 'w')
        fp.write('.model FringeTree\n')
        fp.write('.inputs ')
        fp.write(' '.join([getFeatName(i) for i in range(self.nFeats)]))
        fp.write('\n.outputs y\n')
        for names, pats in nLst:
            fp.write('.names {}\n'.format(' '.join(names)))
            for pat in pats:
                fp.write(pat + '\n')
        fp.write('.names {} y\n1 1\n.end\n'.format(getNodeName(0)))
        fp.close()
