import numpy as np
from PIL import Image
from torchvision import transforms as transforms
import matplotlib.pyplot as plt
from torchvision import utils as vutils
import torch

src = './data/21.png'
trg = './data/22.png'
ITER = 5
patch_size = 9  # default


def initialize():
    """
    A: src
    B: trg
    patch: patch size
    stride: 取patch的stride，其实没啥用
    f[x,y]: patch a(x,y) 对应 patch b的坐标差offset
    D[x,y]: patch a(x,y) 对应offset f(x,y)的distance
    """
    randbr = torch.randint(p, Bh - p, [Ah, Aw])  # random patch B 中心点坐标 initial
    randbc = torch.randint(p, Bw - p, [Ah, Aw])
    f = torch.zeros([Ah, Aw, 2],dtype=int)  # f(x,y): (x,y)算的是patch中心点坐标 a对应b的offset, 整个patch: x:[x-p:x+p+1] y:[y-p:y+p+1]
    D = torch.zeros([Ah,Aw])  # D distance (x,y) offset对应patch的distance
    for i in range(p, Ah - p):  # i,j 中心点坐标
        for j in range(p, Aw - p):  # 先横着扫patch再竖着扫
            a = A[:, i - p:i + p + 1, j - p:j + p + 1]
            rx,ry=randbr[i,j],randbc[i,j]
            b = B[:, rx - p:rx + p + 1, ry - p:ry + p + 1]  # random patch
            # print("a shape, b shape: ",a.shape,b.shape)
            f[i, j] = torch.tensor([randbr[i, j] - i, randbc[i, j] - j],dtype=int)  # offset,b-a
            D[i, j] = torch.sum(torch.abs(a - b))  # i,j 对应offset(f(i,j))的patch distance
    # 为了propagation方便计算f(x-1,y),f(x,y-1),f(x+1,y),f(x,y+1)这里对f进行填充
    # f有用的地方其实没有Ah,Aw那么大，对于边界值也是需要处理的，我们就把四条边的旁边的f用他们本身填充
    # f实际: [p:Ah-p+1,p:Aw-p+1]
    # 对D其实做nan填充就行了,因为我们选D小的进行propagate
    D[:p, :], D[Ah - p:, :], D[:, :p], D[:, Aw - p:] = np.nan, np.nan, np.nan, np.nan
    return f, D


def propagation(x, y, itr):
    """
    x: current patch a in A'x
    y: current patch a in A'y
    itr: iter num
    A: src
    B: trg
    patch: patch size
    由于是global f,D 所以这里没有return
    """
    if itr % 2 == 1:  # odd奇数，左往右，上往下迭代
        if D[x, y] > D[x - 1, y] or D[x, y] > D[x, y - 1]:
            f[x,y]= f[x-1,y] if D[x-1,y]<D[x,y-1] else f[x,y-1] if D[x,y-1]<D[x,y] else f[x,y]
    else:  # even偶数，右往左，下往上迭代
        if D[x, y] > D[x + 1, y] or D[x, y] > D[x, y + 1]:
            f[x,y]=f[x+1,y] if D[x+1,y]<D[x,y+1] else f[x,y+1] if D[x,y+1]<D[x,y] else f[x,y]
    offset=f[x,y]
    print("x,y,offset",x,y,offset)
    a=A[:,x-p:x+p+1,y-p:y+p+1]

    b=B[:,x+offset[0]-p:x+offset[0]+p+1,y+offset[1]-p:y+offset[1]+p+1]
    print("a shape, b shape, itr: ",a.shape,b.shape,itr)
    D[x,y]=torch.sum(torch.abs(a-b))
    # D[x, y] = torch.sum(torch.abs(A[:, x-p:x+p+1, y-p:y+p + 1] - B[:, x + offset[0] - p:x + offset[0] + p + 1,y + offset[1] - p:y + offset[1] + p + 1]))


def random_search(x, y, i=0, alpha=0.5):
    """
    在offset f(x,y)对应的B的patch邻域内进行random search
    在search window里进行random sample得到patch然后比较sample的D和当前的D大小，决定要不要更新
    x: current patch a in A'x
    y: current patch a in A'y
    B: search domain in image B
    i: 窗口大小衰减ratio
    alpha: 窗口每次缩多少（这里是每次缩1/2）
    """
    sh = Bh * alpha ** i  # search window h
    sw = Bw * alpha ** i  # search window w
    bx, by = x + f[x, y][0], y + f[x, y][1]
    while sh > 1 and sw > 1:
        rx, ry = np.random.randint(max(bx - sh, p), min(bx + sh, Bh - p)), np.random.randint(max(by - sh, p),
                                                                                             min(by + sh, Bw - p))
        # random x, random y
        tmp = torch.sum(torch.abs(A[:, x - p:x + p + 1, y - p:y + p + 1] - B[:, rx - p:rx + p + 1, ry - p:ry + p + 1]))
        if tmp < D[x, y]:
            D[x, y] = tmp
            f[x, y] = torch.tensor([rx - x, ry - y],dtype=int)
        sh *= alpha
        sw *= alpha


def PatchMatch(src, trg):
    """
    :param src: source image patch, t frame,A
    :param trg: target image patch, t+1 frame,B
    :param ps: patch_size, default=32*32
    :return:
    """
    transform = transforms.Compose([transforms.ToTensor()])
    global Ah, Aw, Bh, Bw, A, B
    A = transform(Image.open(src))
    B = transform(Image.open(trg))
    Ah, Aw = A.shape[1], A.shape[2]
    Bh, Bw = B.shape[1], B.shape[2]
    global p
    p = patch_size // 2
    global f, D
    f, D = initialize()
    print("initial done!")
    print(f,'\n','-'*20,'\n',D)
    for itr in range(1, ITER + 1):
        if itr % 2 == 1:
            for x in range(p, Ah - p):
                for y in range(p, Aw - p):
                    propagation(x, y, itr)
                    random_search(x, y)
        else:
            for x in range(Ah - p-1, p - 1, -1):
                for y in range(Aw - p-1, p - 1, -1):
                    propagation(x, y, itr)
                    random_search(x, y)
        print(f"Iteration {itr} done.")
    print(f, D)


if __name__ == '__main__':
    PatchMatch(src, trg)
