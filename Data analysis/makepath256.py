from src.classes import ConvGRUClassifier
from src.dataloaders import TabulatedSeries
from torchvision import transforms
import torch
from torch import nn
import os
import argparse
import numpy as np

string = ""
base_path="data/evos_512x512/"
num_frames=101
filename = "test512.txt"
num_evos = 1

parser = argparse.ArgumentParser()
parser.add_argument("model_number")
args = parser.parse_args()
model_number = args.model_number

def apply_dual(feature):
    return torch.ones_like(feature)-feature

def apply_sharpness(feature):
    f = lambda x: np.float32(0) if x<0.5 else np.float32(1)
    #apply operation element_wise
    return torch.tensor(np.vectorize(f)(feature))

def apply_noise(feature,mean=0,sigma=0.01):
    #f = lambda x: x + np.float32(np.random.normal(loc=mean,scale=sigma,size=1))
    correct = lambda x: np.float32(1) if(x>1) else (np.float32(0) if(x<0) else x)

with open(filename,"w") as f:
    f.write("")

param=0

for i in range(num_evos):
    with open(base_path+"evo_"+str(i)+"/param.txt") as f:
        param = float(f.read())
        print(param)
    for j in range(1,num_frames):
        string+= base_path+"evo_"+str(i)+"/eta_"+str(param)+"_snap_"+str(j)+".npy "
    with open(filename,"a") as f:
        f.write(string+str(param)+"\n")
    #print(string)
    string=""

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

transform = transforms.Compose(
            [
                transforms.Grayscale( num_output_channels=1 ),
                transforms.Resize(128),
                transforms.ToTensor()
            ]
        )

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
                batch_size      = 1,
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
loss_fn = nn.MSELoss()
pred = torch.tensor([])
count = 0
#torch.no_grad() activates inference mode, deactivating those operations which are not linked to evaluation (for example the computation of gradients and related operations)
with torch.no_grad():
    for feature,label in dataloader:
            count+=1

            pred = model(feature).squeeze()
            predictions = torch.cat((predictions, pred),0)
            loss.append(loss_fn(torch.mean(pred),label[0][0]).item())
            labels = torch.cat((labels,label[0]),0)

            print("\nPREDICTION#{}:".format(count))
            print(pred)
            print("MEAN:")
            print(torch.mean(pred))
            print("VARAINCE: ")
            print(torch.var(pred))
            print("LABEL:")
            print(label[0])
            '''
            print("PREDICTION WITH DUAL:")
            pred1 = model(apply_dual(feature)).squeeze()
            print(pred1)
            print("MEAN WITH DUAL:")
            print(torch.mean(pred1))
            print("VARAINCE WITH DUAL: ")
            print(torch.var(pred1))
            '''

predictions = predictions.numpy()
labels = labels.numpy()
#print("Predicted: " +str(predictions)+ "\nTrue: " + str(labels) + "\nLoss: " + str(loss))

mloss =np.mean(loss)
print("Loss mean on the test set: " + str(mloss))
'''
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
print("Total values inside two sigmas: " + str(count[1]) + "/" + str(len(predictions)))
print("Total values inside three sigmas: " + str(count[2]) + "/" + str(len(predictions)))
#regression_plot(labels,predictions,mloss,min_valid,float(count[0])/len(predictions))

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
'''
