import pandas as pd
import os
import numpy as np
from dna2vec.multi_k_model import MultiKModel
import matplotlib.pyplot as plt

# workDir = 'D:/Programming/python/PycharmProjects/ProteinDNABinding/'
workDir = 'D:/Programming/python/PycharmProjects/ProteinDNABinding/'
dataDir = workDir + 'data/rawdata/'
modelDir = workDir + 'model/'
w2vrawDir = workDir + 'data/pretrained/dna2vec-20161219-0153-k3to8-100d-10c-29320Mbp-sliding-Xat.w2v'
threemerDir = workDir + 'data/pretrained/3mer.txt'
w2vDir = workDir + 'data/pretrained/DNA2Vec_dict.npy'
threeDir = workDir + 'data/pretrained/3mer.npy'

plotDir = workDir + 'plot/'
id_seq_dict = {}
DNA2Vec = np.load(w2vDir, allow_pickle=True).item()
def create_list(path):
    list = []
    for i, j, k in os.walk(path):
        for item in j:
            list.append(item)
    return list
def readfile(in_file, out_file, label):
    i = 0
    for line in in_file:
        if i % 2 == 0:
            seqID = line.strip("\n")
        elif i % 2 == 1:
            sequence = line.strip("\n")
            if not sequence.__contains__('N'):
                out_file.write(sequence + "\t" + str(label) + "\n")
                id_seq_dict[sequence] = seqID
        i = i + 1


# 训练集测试集自动构造
def make_file(makeDir, name):
    positive_file = open(makeDir + name + 'positive.fa', 'r')
    negative_file = open(makeDir + name + 'negative.fa', 'r')
    out_file = open(makeDir + name + 'all_data.txt', 'w')
    out_file.write('sequence' + '\t' + 'label' + '\n')
    readfile(positive_file, out_file, 1)
    readfile(negative_file, out_file, 0)
    np.save(makeDir + name + 'id_seq_dict.npy', id_seq_dict)
    id_seq_dict.clear()
make_file("D:/Programming/python/PycharmProjects/ProteinDNABinding/data/rawdata","/n2f222/")
def make_test(makeDir, name):
    positive_file_path = makeDir + name + 'test.fa'
    out_file_path = makeDir + name + 'test.txt'
    npy_file_path = makeDir + name + 'id_seq_dict.npy'

    sequence_count = 0

    with open(positive_file_path, 'r') as positive_file, open(out_file_path, 'w') as out_file:
        out_file.write('sequence' + '\t' + 'label' + '\n')
        sequence_count = readfile(positive_file, out_file, 0)

    np.save(npy_file_path, id_seq_dict)
    id_seq_dict.clear()

    print(f"Total sequences processed: {sequence_count}")

# 示例调用
make_test("D:/Programming/python/PycharmProjects/ProteinDNABinding/data/rawdata", "/n2f222/")
def make_DNA2Vec_dict():
    DNA2Vec = {}
    all_data = pd.read_csv(threemerDir, header=None, sep='\t')
    for i in range(0, all_data.shape[0]):
        line = all_data[0][i].split()
        DNA = line[0]
        vec = line[1:]
        a = np.array(vec)
        a[a == ''] = 0.0
        a = a.astype(np.double)
        DNA2Vec[DNA] = a
    np.save(threeDir, DNA2Vec)
make_DNA2Vec_dict()

def k_mer_stride(dna, k, s):
    l = []
    dna_length = len(dna)
    j = 0
    for i in range(dna_length):
        t = dna[j:j + k]
        if (len(t)) == k:
            vec = DNA2Vec[t]
            l.append(vec)
        j += s
    return l


def mkdir(path):
    # 引入模块
    import os
    # 去除首位空格
    path = path.strip()
    # 去除尾部 \ 符号
    path = path.rstrip("\\")
    # 判断路径是否存在
    # 存在     True
    # 不存在   False
    isExists = os.path.exists(path)
    # 判断结果
    if not isExists:
        # 如果不存在则创建目录
        # 创建目录操作函数
        os.makedirs(path)


if __name__ == "__main__":
    pass
    # make_DNA2Vec_dict()
    # file_list = create_list(dataDir)
    # for name in file_list:
    #     print(name)
        # 定义要创建的目录
        # mkpath = modelDir + '/' + name
        # 调用函数
        # mkdir(mkpath)

make_DNA2Vec_dict()