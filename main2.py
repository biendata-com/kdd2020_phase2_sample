from gcn import *
import pickle as pkl
from utils import *
import numpy as np
import sys
import os
from torch.autograd import Variable

if __name__ == "__main__":

    dims=[100,64,19]
    #print(dims)
    os.environ["CUDA_VISIBLE_DEVICES"] = '0'

    adj=pkl.load(open(sys.argv[1],'rb'))
    features=np.load(sys.argv[2])
    processed_adj=GCNadj(adj)

    featuretensor=torch.FloatTensor(features)
    featuretensor=Variable(featuretensor,requires_grad=True).cuda()

    #print(processed_adj.row)
    sparserow=torch.LongTensor(processed_adj.row).unsqueeze(1)
    sparsecol=torch.LongTensor(processed_adj.col).unsqueeze(1)
    sparseconcat=torch.cat((sparserow,sparsecol),1).cuda()
    sparsedata=torch.FloatTensor(processed_adj.data).cuda()
    #print(sparseconcat,sparsedata,processed_adj.shape)
    adjtensor=torch.sparse.FloatTensor(sparseconcat.t(),sparsedata,torch.Size(processed_adj.shape)).cuda()
    print("here5")
    wd=0

    model=GCN_norm(len(dims)-1,dims)

        
    best_val=0



    model.load_state_dict(torch.load("norm2"))
    #model=model.cpu()
    model=model.cuda()
    testout=model(featuretensor,adjtensor,dropout=0)

    testoc=testout.argmax(1)

    testo=testoc.data
    testo=testo+1

    f2=open(sys.argv[3],'w')

    for i in range(len(testo)):
        print(int(testo[i]),file=f2)
        
    f2.close()
