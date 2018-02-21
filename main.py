"""
李偉的第一個gmm模型用來做聚類
"""


print(__doc__)
from scipy.stats import multivariate_normal as MN
import numpy as np
from sklearn import preprocessing
from sklearn.datasets.samples_generator import make_blobs
from sklearn import metrics
## INIT PROCESS
ITER = 50
CLUSTER_NUM = 4
centers = [[2, 2], [-2, -2], [2, -2],[-2,2]]
DATA, LABEL = make_blobs(n_samples=1000, centers=centers, cluster_std=0.9)
FEAT_NUM = DATA.shape[1]
INS_NUM = DATA.shape[0]
MEAN = DATA.mean(axis=0)
MIN,MAX = DATA.min(axis=0),DATA.max(axis=0)

##DISTRI
DISTRI = [
    MN(mean=np.random.uniform(MIN,MAX))
    for i in range(CLUSTER_NUM)
]

for i in range(ITER):
    dis_all = np.array([
        [
            DISTRI[c].pdf(d)
            for c in range(CLUSTER_NUM)
        ]
        for d in DATA
    ],dtype=np.float64)
    label = np.argmax(dis_all,axis=1)
    loss = metrics.adjusted_rand_score(LABEL, label)
    print("iter %s loss %.3f" % (i,loss))
    dis_all = dis_all.transpose()/dis_all.sum(axis=1)
    dis_all = dis_all.transpose()
    mu_mo = np.matmul(DATA.transpose(),dis_all)
    mu_no = dis_all.sum(axis=0,keepdims=1)
    mu_mat = mu_mo/mu_no
    cov_mat = []
    for c in range(CLUSTER_NUM):
        #tmp = np.eye(FEAT_NUM,FEAT_NUM)
        tmp = np.zeros([FEAT_NUM,FEAT_NUM])
        for d in range(INS_NUM):
            D = DATA[d]-mu_mat.transpose()[c]
            D = np.expand_dims(D,axis=0)
            tmp += (
                np.matmul(
                    D.transpose(),
                    D
                ) * dis_all[d][c]
            )

        cov_mat.append(tmp/mu_no[0][c])
    DISTRI = [
        MN(mean=mu_mat.transpose()[i],cov=cov_mat[i])
        for i in range(CLUSTER_NUM) 
    ]

print(dis_all)
print(cov_mat)
#print([LABEL,np.argmax(dis_all,axis=1)])
label = np.argmax(dis_all,axis=1)
import matplotlib.pyplot as plt
from itertools import cycle

colors = cycle('bgrcmykbgrcmykbgrcmykbgrcmyk')
for k, col in zip(range(CLUSTER_NUM), colors):
    my_members = label == k
    plt.plot(DATA[my_members, 0], DATA[my_members, 1], col + '.')
plt.title('Estimated number of clusters: %d' % CLUSTER_NUM)
plt.show()
