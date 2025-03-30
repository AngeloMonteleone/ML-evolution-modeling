import os
import argparse
import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument("train")
parser.add_argument("valid")
parser.add_argument("test")
parser.add_argument("step")
parser.add_argument("rand")

args = parser.parse_args()

num_train = int(args.train)
num_valid = int(args.valid)
num_test = int(args.test)

string = ""
base_path="data/test/"
num_evols = num_train + num_valid + num_test
num_frames=43
num_train=int(args.train)

step = int(args.step)
rand = int(args.rand)

arr=np.array([])
'''
with open("train_file_rand.txt","w") as f:
    f.write("")

#NOTA BENE: l'implementazione della randomizzazione delle seqeunze può esserre introdotta con una funzione che restituisca un generatore tramite la parola chiave yield. Non è assolutamente indispensabile, ma forse funziona allo stesso modo

for i in range(num_train):
    #print(base_path+str(i))
    if(rand):
        arr = sorted(np.random.choice(num_frames+1,size=rand,replace=False))
    else:
        arr = range(0,num_frames+1,step)
    for j in arr:
        string+= base_path+str(i)+"/s_"+str(j)+".npy "
    with open(base_path+str(i)+"/param.txt") as f:
        string += f.read()
    with open("train_file_rand.txt","a") as f:
        f.write(string)
    #print(string)
    string=""

with open("valid_file_rand.txt","w") as f:
    f.write("")

for i in range(num_train,num_evols+1-num_test):
    #print(base_path+str(i))
    if(rand):
        arr = sorted(np.random.choice(num_frames+1,size=rand,replace=False))
    else:
        arr = range(0,num_frames+1,step)
    for j in arr:
        string+= base_path+str(i)+"/s_"+str(j)+".npy "
    with open(base_path+str(i)+"/param.txt") as f:
        string += f.read()
    with open("valid_file_rand.txt","a") as f:
        f.write(string)
    #print(string)
    string=""
'''
with open("test_file_rand.txt","w") as f:
    f.write("")

for i in range(num_evols+1-num_test,num_evols+1):
    #print(base_path+str(i))
    if(rand):
        arr = sorted(np.random.choice(num_frames+1,size=rand,replace=False))
    else:
        arr = range(0,num_frames+1,step)
    print(arr)
    for j in arr:
        string+= base_path+str(i)+"/s_"+str(j)+".npy "
    with open(base_path+str(i)+"/param.txt") as f:
        string += f.read()
    with open("test_file_rand.txt","a") as f:
        f.write(string)
    #print(string)
    string=""
