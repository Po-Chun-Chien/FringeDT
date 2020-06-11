import os
import numpy as np

def readPLA(fn):
    if not os.path.isfile(fn):
        print('Warning: PLA "{}" not found.'.format(fn))
        return (None,) * 3
    getNum = lambda s, head: int(s.strip('\n').replace(head, '').replace(' ', ''))
    getPat = lambda s: s.strip('\n').replace(' ', '')
    getArr = lambda x: np.array(x, dtype=np.int8)
    with open(fn) as fp:
        ni = getNum(fp.readline(), '.i')
        no = getNum(fp.readline(), '.o')
        nl = getNum(fp.readline(), '.p')
        for line in fp:
            if line.startswith('.type fr'):
                break
        
        assert no == 1
        data, labels = [], []
        for i in range(nl):
            pat = getPat(fp.readline())
            assert(len(pat) == ni + no)
            data.append(getArr([b for b in pat[:-1]]))
            labels.append(pat[-1])
        
        for line in fp:
            if line.startswith('.e'):
                break
                
    return ni, getArr(data), getArr(labels)
    
def dumpPLA(data, labels, fn):
    if len(data) == 0: return
    assert len(data) == len(labels)
    ni = len(data[0])
    nr = len(data)
    with open(fn, 'w') as fp:
        fp.write('.i {}\n.o 1\n'.format(str(ni)))
        fp.write('.p {}\n.type fr\n'.format(str(nr)))
        for pat, lab in zip(data, labels):
            pat = ''.join([str(i) for i in pat])
            fp.write('{} {}\n'.format(pat, str(lab)))
        fp.write('.e\n')
            
