import os
import pickle as pk
import numpy as np
from argparse import ArgumentParser
from dtree import DTree
from utils import readPLA


def getArgs():
    parser = ArgumentParser()
    parser.add_argument('-td', '--train_data', type=str)
    parser.add_argument('-vd', '--valid_data', type=str)
    parser.add_argument('-rs', '--random_seed', type=int, default=None)
    parser.add_argument('-mf', '--max_num_feats', type=int, default=1500)
    parser.add_argument('-md', '--max_depth', type=int, default=20)
    parser.add_argument('-cc', '--ccp_alpha', type=float, default=None)
    parser.add_argument('-cr', '--criterion', type=str, default='entropy', choices=['gini', 'entropy'])
    parser.add_argument('-vb', '--verbose', action='store_true')
    parser.add_argument('-sm', '--save_model', type=str, default=None)
    parser.add_argument('-lm', '--load_model', type=str, default=None)
    parser.add_argument('-db', '--dump_blif', type=str, default=None)
    #parser.add_argument('-dp', '--dump_PLA', type=str, default=None)
    #parser.add_argument('-df', '--dump_feats', type=str, default=None)
    args = parser.parse_args()    
    return args
    
    
if __name__ == '__main__':
    args = getArgs()
    ni1, trnData, trnLabels = readPLA(args.train_data)
    ni2, valData, valLabels = readPLA(args.valid_data)
    assert ni1 and ni2 and (ni1 == ni2)
    
    if args.load_model:
        dt = pk.load(open(args.load_model, 'rb'))
    else:
        mf, md = args.max_num_feats, args.max_depth
        cr, cc = args.criterion, args.ccp_alpha
        rs, vb = args.random_seed, args.verbose
        dt = DTree(max_nFeats=mf, criterion=cr, max_depth=md, ccp_alpha=cc, randSeed=rs, verbose=vb)
    
    if args.ccp_alpha is not None:
        # train with given ccp_alpha
        trnAcc, valAcc = dt.train(trnData, trnLabels, valData, valLabels)
    else:
        # automatically select the best ccp_alpha based on val. acc.
        trnAcc, valAcc = dt.train2(trnData, trnLabels, valData, valLabels)
    
    if args.verbose:
        print('overall results (tra/val acc.): {} / {}\n'.format(str(trnAcc), str(valAcc)))
    
    if args.save_model:
        pk.dump(dt, open(args.save_model, 'wb'))
    if args.dump_blif:
        fn = args.dump_blif
        assert fn.endswith('.blif')
        dt.toBlif(fn, True)
        dt.toBlif(fn.replace('.blif', '_last.blif'), False)
