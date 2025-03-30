from src.classes import ConvGRU, ConvGRUClassifier
from src.utils import save_vtk
from src.dataloaders import TabulatedSeries
from test_class import ConvGRUTEST
from torchvision import transforms
import torch
from torch import nn
import numpy as np
import matplotlib.pyplot as plt
import argparse
import os
from numba import njit

model_class = ConvGRU
save_evo = True

dt = 0.01

num_frames=200
filename = ""
num_evos = 1

parser = argparse.ArgumentParser()
parser.add_argument("model_number")
#parser.add_argument("filename")
parser.add_argument("dual")
parser.add_argument("sharp")
parser.add_argument("noise")
parser.add_argument("create_file")

args = parser.parse_args()
model_number = args.model_number
#filename = args.filename
dual = int(args.dual)
sharp = int(args.sharp)
noise = int(args.noise)
if(dual):
    print("Applying Z2 symmetry to the test set")
if(sharp):
    print("Applying sharpness to the test set")
if(noise):
    print("Applying gaussian noise to the test set")

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

#train = np.log(train)
#valid = np.log(valid)
#x = np.log(x)

#fig = plt.figure()
#ax = plt.subplot()
#ax.plot(x,train)
#ax.plot(x,valid)
#plt.show()

#TEST ON SINGLE FILE, BECAUSE I AM LAZY
if(args.create_file!="n"):
    num_frames = 0
    base_path = args.create_file
    for file in os.listdir(base_path + "/evo_0"):
        name = os.fsdecode(file)
        if name.endswith(".npy"):
            num_frames+=1

    string = ""
    filename = "generictest.txt"
    with open(filename,"w") as f:
        f.write("")

    param=0

    for i in range(num_evos):
        with open(base_path+"/evo_"+str(i)+"/param.txt") as f:
            param = float(f.read())
            print(param)
        for j in range(1,num_frames+1):
            string+= base_path+"/evo_"+str(i)+"/eta_"+str(param)+"_snap_"+str(j)+".npy "
        with open(filename,"a") as f:
            f.write(string+str(param)+"\n")
        #print(string)
        string=""
else:
    filename="test_file.txt"
    num_frames=44
    num_evos = 100

print("detected frames: {}".format(num_frames))
print("future set to: {}".format(num_frames-2))
print("number of evolutions: {}".format(num_evos))
#print("base path: " + base_path)

transform = transforms.Compose(
            [
                transforms.Grayscale(),
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
model = model_class(
        hidden_units        = 2,
        input_channels      = 1, # this is hardcoded for the moment... waiting for multidimensional data!
        output_channels     = None,
        hidden_channels     = 16,
        kernel_size         = 5,
        padding_mode        = "circular",
        separable           = False,
        bias                = True,
        divergence          = True,
        num_params          = 1,
        dropout             = False,
        dropout_prob        = None
        )

model.load_state_dict(torch.load(model_path))
model.make_div_filters(torch.zeros(1,device="cpu"))
model.eval()#IMPORTANT: The model must be set to evaluation mode
'''
print("Model's state_dict:")
for param_tensor in model.state_dict():
    print(param_tensor, "\t", model.state_dict()[param_tensor].size())

input("\n\npress enter\n\n")
'''

loss = []
loss_fn = nn.MSELoss()
count = 0
#torch.no_grad() activates inference mode, deactivating those operations which are not linked to evaluation (for example the computation of gradients and related operations)
'''
#FOR PARAMETER EXTRACTION
extractor_model = ConvGRUClassifier(
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

extractor_model.load_state_dict(torch.load("train_logs/param_14/model/epoch_145.pt"))
extractor_model.eval()
param_loss = []
'''
@njit#(parallel = True)
def der (func, eps): #takes func as array, returns der as array
	leng = len(func)
	f_der = np.zeros(leng)
	for i in range (leng):
		if i == 0: f_der[i] = (func[1] - func[leng-1])/(2*eps)
		elif i == leng-1: f_der[i] = (func[0] - func[leng-2])/(2*eps)
		else: f_der[i] = (func[i+1] - func[i-1])/(2*eps)
	return list(f_der)

def der_x2D(func,eps):
    ret = []
    for row in func:
        ret.append(der(row,eps))
    return np.array(ret)

def der_y2D(func,eps):
    func = np.array(func).transpose()
    ret = []
    for row in func:
        ret.append(der(row,eps))
    return np.array(ret).transpose()

def delta(lab,pred):
    #print(lab.shape)
    #print(pred.shape)
    ret=[]
    lab = lab[1:,...]
    for i in range(pred.shape[0]):
        #print(i)
        single_lab = lab[i,0,...].numpy()
        single_pred = pred[i,0,...].numpy()
        gradx = der_x2D(single_lab,1)
        grady = der_y2D(single_lab,1)
        power = lambda a: a**2
        num = np.vectorize(power)(single_lab-single_pred)
        den = np.vectorize(power)(gradx) + np.vectorize(power)(gradx)
        den = np.vectorize(power)(den)
        integral_den = np.sum(den)
        integral_num = np.sum(num)
        ret.append((integral_num/integral_den)**.5)
        print(ret[-1])
    return ret

with torch.no_grad():
    for feature,label in dataloader:
            if(dual):
                feature = apply_dual(feature)
            if(sharp):
                feature = apply_sharpness(feature)
            if(noise):
                feature = apply_noise(feature)
            count+=1
            print(count)
            pred = model(feature[:,0:1,:,:,:],future=(num_frames-2),params=label)
            if(save_evo and os.path.exists("temp")):
                if(input("save?")!=''):
                    print("saving single evolution in temp")
                    for i in range(pred.shape[1]):
                        save_vtk(pred[0,i,...].numpy(),"temp/output_{}.vtk".format(i))
            elif(not(os.path.exists("temp")) and save_evo):
                print("temp folder does not exist!")
            else:
                print("not saving")
            '''
            print("PREDICTION:")
            print(pred)
            print("LABEL:")
            print(label)
            '''
            loss.append(loss_fn(pred,feature[:,1:,:,:,:]).item())
            print("MSE LOSS: {:.3E}".format(loss[-1]))
            print("CROSS ENTROPY: {:.3E}".format(nn.functional.cross_entropy(pred,feature[:,1:,:,:,:])))
            del_bar = delta(feature[0,...],pred[0,...])
            int_del_bar = 0
            for el in del_bar:
                int_del_bar += el
            int_del_bar*=dt
            print("Integral delta: {:.3E}".format(int_del_bar))
            #valutazione primi frame
            print(pred.shape)
            print(feature.shape)
            frame_loss = []
            frame_loss_modified = []
            for i in range(pred.shape[1]):
                frame_real = feature[0,i+1,...]
                frame_pred = pred[0,i,...]
                frame_loss.append(loss_fn(frame_real,frame_pred))
                if(i<10):
                    frame_loss_modified.append(frame_loss[-1]*10)
                else:
                    frame_loss_modified.append(frame_loss[-1])
                print("Loss on single frame: {:.3E}, above mean: {}, modified loss: {:.3E}".format(frame_loss[-1],int(frame_loss[-1]>loss[-1]),frame_loss_modified[-1]))
            print("mean frame loss: {:.3E}".format(np.mean(frame_loss)))
            print("mean frame modified loss: {:.3E}".format(np.mean(frame_loss_modified)))
            #input("enter")
'''
            parameter_pred = extractor_model(pred).squeeze()
            #print(parameter_pred.shape)
            #print(label[0].squeeze().shape)#I have to squeeze the true label tensor for the dimensions to match
            param_loss.append(loss_fn(parameter_pred,label[0].squeeze()).item())
            print("PREDICTED ETA")
            print(parameter_pred)
            print("TRUE ETA:")
            print(label[0].squeeze())
            print("LOSS ON PARAMETER")
            print(param_loss[-1])
            #input("enter")
'''
#print("Predicted: " +str(predictions)+ "\nTrue: " + str(labels) + "\nLoss: " + str(loss))

mloss =np.mean(loss)
print("Loss mean on the test set: " + str(mloss))
'''
mparam_loss = np.mean(param_loss)
print("Loss mean on the parameter test: " + str(mparam_loss))
'''
