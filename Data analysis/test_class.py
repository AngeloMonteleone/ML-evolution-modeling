from src.classes import ConvGRUClassifier, ConvGRU
from src.utils import save_vtk
import torch
from torch import nn
import numpy as np

class ConvGRUClassifierTEST(ConvGRUClassifier):
    '''
    This class subclasses ConvGRUClassifier to print the output of every layer
    '''
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def forward(self, in_sequence, future=0, params=None, noise_reg=0.0):
        in_sequence = self.cat_params(in_sequence, params)
        GRU_result = self.forward_old(in_sequence, future, params=params, noise_reg=noise_reg) # non conservative dynamics
        print("GRU result: ")
        print(GRU_result.shape)
        x = GRU_result[:,-1,:,:,:]
        for name, module in self.toOut.named_children():
            if(not name.startswith('params')):
                print("Model name: " + name)
                x = module(x)
                print(x)

        return x

class ConvGRUTEST(ConvGRU):
    def __init__(self, *args, **kwargs):
        super().__init__(*args,**kwargs)
        #self.arr_output = []
        self.arr_Jx = []
        self.arr_Jy = []
        self.arr_gradx = []
        self.arr_grady = []
        self.arr_div = []

    def divergence(self,x):
        #save_vtk(x.numpy()[0,...],"temp/output.vtk")
        #self.arr_output.append(x.numpy()[0,...])
        #print(x.numpy()[0,...])
        Jx, Jy = torch.split(x, 1, dim=1)

        Jx = Jx - torch.mean(Jx, dim=(-1,-2), keepdim=True)
        Jy = Jy - torch.mean(Jy, dim=(-1,-2), keepdim=True)

        #print("J_x: " + str(Jx.shape))
        #print(Jx.numpy()[0,...])
        self.arr_Jx.append(Jx.numpy()[0,...])
        #save_vtk(Jx.numpy()[0,...],"temp/J_x.vtk")
        #print("J_y " + str(Jy.shape))
        #print(Jy)
        self.arr_Jy.append(Jy.numpy()[0,...])
        #save_vtk(Jy.numpy()[0,...],"temp/J_y.vtk")

        gradx = self.divergence_filters[0](Jx)
        grady = self.divergence_filters[1](Jy)

        #print("gradx " + str(gradx.shape))
        #print(gradx)
        self.arr_gradx.append(gradx.numpy()[0,...])
        #save_vtk(gradx.numpy()[0,...],"temp/gradx.vtk")
        #print("grady " + str(grady.shape))
        #print(grady)
        self.arr_grady.append(grady.numpy()[0,...])
        #save_vtk(grady.numpy()[0,...],"temp/grady.vtk")

        divergence = gradx+grady

        #print("divergence")
        #print(divergence)
        self.arr_div.append(divergence.numpy()[0,...])
        #save_vtk(divergence.numpy()[0,...],"temp/div.vtk")
        print("frame {}".format(len(self.arr_div)))
        if(len(self.arr_div)==43):
            print("saving")
            self.savedata()
            #self.arr_output = []
            self.arr_Jx = []
            self.arr_Jy = []
            self.arr_gradx = []
            self.arr_grady = []
            self.arr_div = []
            input("finished saving, press enter to proceed")
        return divergence

    def savedata(self):
        names=["Jx","Jy","gradx","grady","div"]
        arrs = [self.arr_Jx,self.arr_Jy,self.arr_gradx,self.arr_grady,self.arr_div]
        for ind in range(len(arrs)):
            arr = arrs[ind]
            print("saving: " + names[ind])
            for i in range(len(arr)):
                #print(arr[i])
                #print("saving \'temp/" + names[ind] + "_{}.vtk\'".format(i))
                save_vtk(arr[i],"temp/" + names[ind] + "_{}.vtk".format(i))
