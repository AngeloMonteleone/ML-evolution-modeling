from src.classes import ConvGRUClassifier
from src.dataloaders import TabulatedSeries
from torchvision import transforms
import torch
from torch import nn
import numpy as np
import matplotlib.pyplot as plt
import argparse
import os

string = ""
base_path="data/npy_data/"
num_evols = 999
num_frames=43
showreg = False

parser = argparse.ArgumentParser()
parser.add_argument("model_number")
'''
parser.add_argument("dual")
parser.add_argument("sharp")
parser.add_argument("noise")
'''
args = parser.parse_args()
model_number = args.model_number
'''
dual = int(args.dual)
sharp = int(args.sharp)
noise = int(args.noise)
if(dual):
    print("Applying Z2 symmetry to the test set")
if(sharp):
    print("Applying sharpness to the test set")
if(noise):
    print("Applying gaussian noise to the test set")
'''
def apply_dual(feature):
    return torch.ones_like(feature)-feature

def apply_sharpness(feature):
    f = lambda x: np.float32(0) if x<0.5 else np.float32(1)
    #apply operation element_wise
    return torch.tensor(np.vectorize(f)(feature))

def apply_noise(feature,mean=0,sigma=0.01):
    #f = lambda x: x + np.float32(np.random.normal(loc=mean,scale=sigma,size=1))
    correct = lambda x: np.float32(1) if(x>1) else (np.float32(0) if(x<0) else x)
    return torch.tensor(np.vectorize(correct)(feature + sigma*torch.randn(feature.size())))
def regression_plot(true_labels,predicted_labels,loss,epoch,sigma):
    fig = plt.figure()
    ax = plt.subplot()
    ax.scatter(true_labels,predicted_labels)
    ax.plot(true_labels,true_labels)
    ax.set_xlabel("True parameter")
    ax.set_ylabel("Predicted parameter")
    ax.text(0.05,0.005,"Selected epoch: {}\nTest loss: {:.3E}\nInside 1$\sigma: {}$".format(epoch,loss,sigma),color="black")
    plt.show()

def plot_results(x,y1,y2,y3,leg_labels,xlab,ylab):
    fig = plt.figure()
    ax = plt.subplot()
    ax.plot(x,y1,label=leg_labels[0])
    ax.plot(x,y2,label=leg_labels[1])
    ax.plot(x,y3,label=leg_labels[2])
    ax.set_xlabel(xlab)
    ax.set_ylabel(ylab)
    ax.legend()
    plt.grid()
    plt.show()

train = []
valid=[]
x=range(200)

with open("train_logs/param_" + model_number + "/train_loss.txt") as f:
    arr = f.read().split()
    for elem in arr:
        train.append(float(elem))
with open("train_logs/param_" + model_number + "/valid_loss.txt") as f:
    arr = f.read().split()
    for elem in arr:
        valid.append(float(elem))

#print(train,valid)

count = 0
tot = len(valid)

for i in range(0,tot):
    if(valid[i]>train[i-1]):
        count+=1

min_valid = np.argmin(valid)
min_train = np.argmin(train)

model_path = "train_logs/param_" + model_number + "/model/epoch_" + str(min_valid) + ".pt"
print("Chosen model path: " + model_path)

print("The validation loss was higher " + str(count) + "/" + str(tot) +" times")
print("The validation loss has a minimum at epoch " + str(min_valid)+ ", value: " + str(valid[min_valid]))
print("The training loss has a minimum at epoch " + str(min_train)+ ", value: " + str(train[min_train]))

train = np.log(train)
valid = np.log(valid)
#x = np.log(x)

#fig = plt.figure()
#ax = plt.subplot()
#ax.plot(x,train)
#ax.plot(x,valid)
#plt.show()
input("continue")
transform = transforms.Compose(
            [
                transforms.Grayscale( num_output_channels=1 ),
                transforms.Resize(128),
                transforms.ToTensor()
            ]
        )

#Beginning tests

loss_fn = nn.MSELoss()
original_losses = []
dual_losses = []
noise_losses = []
#conf_intervals = [[],[],[]]
original_intervals = []
dual_intervals = []
noise_intervals = []
xdata = range(5,44,5)
#xdata = [5]
testveramentebrutto = []

filename = "test_file_rand.txt"

for rand in xdata:
    #DA CONTROLLARE SE FAR RIFARE IL FILE PER OGNI TEST O SE AVERLO COMUNE QUANDO SI APPLICANO LE OPERAZIONI
    with open(filename,"w") as f:
            f.write("")

    for i in range(900,num_evols+1):
        arr = sorted(np.random.choice(num_frames+1,size=rand,replace=False))
        print(arr)
        for j in arr:
            string+= base_path+str(i)+"/s_"+str(j)+".npy "
        with open(base_path+str(i)+"/param.txt") as f:
            string += f.read()
        with open(filename,"a") as f:
            f.write(string)
        #print(string)
        string=""

    for oper in range(0,1):
        arr=np.array([])

        dataset = TabulatedSeries(
                        table_path          = filename,
                        params_num          = 1,
                        transform           = transform,
                        translation         = False,
                        rotation            = False,
                        rotation_90         = False,
                        reflections         = (False, False),
                        cropkey             = False,
                        crop_lim            = None,
                        bootstrap_loader    = False,
                        twin_image          = False
                        )

        dataloader = torch.utils.data.DataLoader(
                        dataset = dataset,
                        batch_size      = 2,
                        shuffle         = False,
                        num_workers     = 4,
                        pin_memory      = True
                        )

            # Define model and put to device
        model = ConvGRUClassifier(
                hidden_units        = 2,
                input_channels      = 1, # this is hardcoded for the moment... waiting for multidimensional data!
                output_channels     = 1,
                hidden_channels     = 16,
                kernel_size         = 5,
                padding_mode        = "circular",
                separable           = False,
                bias                = True,
                divergence          = False,
                num_params          = 0,
                dropout             = False,
                dropout_prob        = None
                )

        model.load_state_dict(torch.load(model_path))
        model.eval()#IMPORTANT: The model must be set to evaluation mode

        predictions = torch.tensor([])
        labels = torch.tensor([])
        loss = []
        #loss_fn = nn.MSELoss()
        pred = torch.tensor([])
        count = 0
        #torch.no_grad() activates inference mode, deactivating those operations which are not linked to evaluation (for example the computation of gradients and related operations)
        with torch.no_grad():
            for feature,label in dataloader:
                    if(oper==1):
                        #print("Applying Z2 symmetry to the test set")
                        feature = apply_dual(feature)
                    '''
                    if(sharp):
                        feature = apply_sharpness(feature)
                    '''
                    if(oper==2):
                        #print("Applying gaussian noise to the test set")
                        feature = apply_noise(feature)

                    count+=1
                    print(count)

                    pred = model(feature).squeeze()
                    loss.append(loss_fn(pred,label[0]).item())
                    predictions = torch.cat((predictions, pred),0)
                    labels = torch.cat((labels,label[0]),0)

        predictions = predictions.numpy()
        labels = labels.numpy()
        #print("Predicted: " +str(predictions)+ "\nTrue: " + str(labels) + "\nLoss: " + str(loss))

        mloss = np.mean(loss)
        print("Loss mean on the test set: " + str(mloss))

        residui = []

        #CALCOLO SOTTOSTIME
        count = 0
        for i in range(len(predictions)):
            residui.append(predictions[i]-labels[i])
            if(predictions[i]<labels[i]):
                count+=1

        print("UNDERESTIMATED VALUES: {}/{}".format(count,len(predictions)))

        if(oper == 0):
            original_losses.append(mloss)
        elif(oper == 1):
            dual_losses.append(mloss)
        elif(oper == 2):
            noise_losses.append(mloss)

        count = [0,0,0]
        sqmloss = np.sqrt(mloss)
        for i in range(len(predictions)):
            if(abs(predictions[i]-labels[i])<sqmloss):
                count[0]+=1
            if(abs(predictions[i]-labels[i])<2*sqmloss):
                count[1]+=1
            if(abs(predictions[i]-labels[i])<3*sqmloss):
                count[2]+=1

        print("Total values inside one sigma: " + str(count[0]) + "/" + str(len(predictions)))
        if(oper == 0):
            original_intervals.append(count[0])
        elif(oper == 1):
            dual_intervals.append(count[0])
        elif(oper == 2):
            noise_intervals.append(count[0])

        print("Total values inside two sigmas: " + str(count[1]) + "/" + str(len(predictions)))
        #conf_intervals[1].append(count[1])
        print("Total values inside three sigmas: " + str(count[2]) + "/" + str(len(predictions)))
        #conf_intervals[2].append(count[2])
        if(showreg):
            regression_plot(labels,predictions,mloss,min_valid,float(count[0])/len(predictions))

        #ANALISI OUTLYIERS

        count = 0
        print("VALUES OUTSIDE 5 SIGMA")
        for i in range(len(predictions)):
            if(abs(predictions[i]-labels[i])>5*sqmloss):
                count+=1
                print("Examples#{}".format(i))
                print("Prediction: {:.4E}\nLabel: {:.4E}".format(predictions[i],labels[i]))
                print("t-value: {}".format(abs(predictions[i]-labels[i])/sqmloss))
        print("NUMBERS OF OUTLIERS: {}".format(count))

        plt.hist(residui)
        plt.show()
    '''
    testveramentebrutto.append(rand)
    plot_results(testveramentebrutto, original_losses, dual_losses, noise_losses, ["Original data","$Z_2$ symmetry","Gaussian noise, $\sigma$=0.01"], "Sequence length", "Test loss")
    plot_results(testveramentebrutto, original_intervals, dual_intervals, noise_intervals, ["Original data","$Z_2$ symmetry","Gaussian noise, $\sigma$=0.01"], "Sequence length", "Values inside 1$\sigma$")
    '''
#plot_results(list(xdata), original_losses, dual_losses, noise_losses, ["Original data","$Z_2$ symmetry","Gaussian noise, $\sigma$=0.01"], "Sequence length", "Test loss")
#plot_results(list(xdata), original_intervals, dual_intervals, noise_intervals, ["Original data","$Z_2$ symmetry","Gaussian noise, $\sigma$=0.01"], "Sequence length", "Values inside 1$\sigma$")

print(original_losses,dual_losses,noise_losses)
print(original_intervals,dual_intervals,noise_intervals)

#with open("train_logs/param_" + model_number + "/test_loss40.txt","w") as f:
    #f.write("")
#with open("train_logs/param_" + model_number + "/test_loss40.txt","a") as f:
    #string=""
    #for arr in [original_losses,dual_losses,noise_losses]:
        #string=" ".join([str(elem) for elem in arr])
        #f.write("\n" + string)

#with open("train_logs/param_" + model_number + "/test_counts.txt","w") as f:
    #f.write("")
#with open("train_logs/param_" + model_number + "/test_counts.txt","a") as f:
    #string=""
    #for arr in [original_intervals,dual_intervals,noise_intervals]:
        #string=" ".join([str(elem) for elem in arr])
        #f.write("\n" + string)
