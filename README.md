# ML-evolution-modeling
This repository is meant to showcase my Bachelor's Thesis _Machine Learning Modeling of the Microstructural Evolution of Strained Materials_. Machine Learning (ML) methods were employed to achieve first and foremost the extraction of a parameter characterizing the evolution of a system from a sequence of snapshots and here I present a synthesis of my work. The relevant code I wrote, is included in the folder _Data analysis_ and is based on the **CRANE code**, which was developed my co-supervisor. It can be found [here](https://github.com/dlanzo/CRANE). 

Every time that I refer to the "code", the "model" or similar appelatives I intend the models which are produced by the use of the CRANE code on a suitable dataset.

## Physics of Spinodal Decomposition
A binary alloy at constant pressure and temperature may exhibit spontaneous phase separation through spinodal decomposition under a certain threshold temperature. This happens because the system tends to minimize Gibbs free energy, which is constituted by two terms: an entropic term, which favours the mixing and an enthalpic term, which may push towards separation, depending on the components of the system. When this happens some regions of space will be characterized by the prevalence of one of the two atomic species, whereas others by the lack thereof.
This system may be further analysed by the Cahn-Hilliard equation, which also includes the cost of the interface between the phases. It permits to treat the dynamics through the evolution of an order parameter, a convenient way to map the concentration fields into a single scalar field. The displacement of material depends on the gradient of the chemical potential, obtained by the minimization of a functional which encloses the different energy contributions to the system. Strain effects arise here through an elastic energy term.
When the two components have the same crystal structure the difference between the lattice parameters $$l_{\alpha}$$ and $$l_{\beta}$$ , represented by the reticular mismatch $$\eta = \frac{l_{\alpha}-l_{\beta}}{l_{\beta}}$$, may influence spinodal decomposition. If $$\eta$$ is sufficiently high, because of the anisotropy of the elastic constants, more rectangular shaped interfaces form, which can be easily distinguished by those coming from a low parameter (Fig. 1).

<p align="center">
<img src="images\confronto strain.png" width="250" height="300">
  <p align="center">
  Fig.1
  </p>
</p>

## ML modeling
Once the elastic constants are fixed, the parameter which determines the microstructure is  and the aim of this study is to extract its value from a video of the evolution of the system through Machine Learning methods. The starting point is a randomly initialized model, depending on a certain number of parameters. These are updated as more examples of evolutions are observed. The criterion for parameter update is the minimization of the mean squared error on the prediction of the parameter.
The scalar field describing the system depends on two spatial dimensions, therefore at every instant it can be represented by an image. For such data Convolutional Networks are handy, because they can capture local features of data organized in a grid-like structure. Moreover, since a single microstructure is not sufficient for parameter extractions, also Recurrent Networks are employed, because they can handle sequences of arbitrary length.
In this way the model can map a tensor with three indices (the sequence) to a scalar (the parameter), learning that the latter must be something like the reticular mismatch encountered during training (Fig. 2)

<p align="center">
<img src="images\example1.png" width="450" height="200">
  <p align="center">
  Fig.2
  </p>
</p>

The strength of Machine Learning methods is their versatility and ability to encode biases leading the model to learn to discriminate data based on certain features. For instance, with a slight change in the network structure for parameter extraction the model may be used to predict the evolution of the system based on the initial conditions (Fig. 3) and to do so an additional layer, is implemented. This layer is inspired from the continuity equation, making it possible to produce an output sequence which respects a local conservation law of the order parameter.

<p align="center">
<img src="images\generalization test_page-0001.jpg" width="430" height="250">
  <p align="center">
  Fig.3
  </p>
</p>

## Quick summary of data analysis
The file `makepath.py` is a simple script providing a .txt file which containes the patterns to the database on which the ML model was tested. The database is not present in this repository, but in its most used form it consisted of 1000 sequences of 44 frames with a resolution of 128x128. Every sequence was coupled with a numerical value, which was the true value of the eigenstrain parameter $$\eta$$.
The sequences were split in different subsets for training and valiation. Most of the times 700 were used for training, 200 for validation and 100 were left out to constitute a test set to be used at a later stage of the analysis.

The file `makepath256.py` does a similar thing to the previous one, but it includes the analysis of the performance of the model. For instance, one of the things it does is checking the values of the training and validation loss (which the model saves upon training) to see in which epoch of training the lowest loss was achieved and how many times the validation loss was higher than the training loss, as indicated in the following three print statements
```python
print("The validation loss was higher " + str(count) + "/" + str(tot) +" times")
print("The validation loss has a minimum at epoch " + str(min_valid)+ ", value: " + str(valid[min_valid]))
print("The training loss has a minimum at epoch " + str(min_train)+ ", value: " + str(train[min_train]))
```
Moreover, this files tests the model on hold out data, on which some operations can be applied, such as the dual transformation which changes the value of every pixel from $$x$$ to $$1-x$$, as each snapshot of each sequence is in grayscale and the system as a $$\mathbb{Z}_2$$ symmetry, which the model should learn upon training and reproduce on test data. This is implemented by the simple function:
```python
def apply_dual(feature):
    return torch.ones_like(feature)-feature
```
This file in particular was made also to test sequences in which the snapshots were of a greater resolution of the original ones used upon training.

The file `autotest.py` repeats the similar analyses of the previous one but repeats the tests multiple times, each time changing the length of the sequence. If the sequences were 44 frames long this program tests the models by selecting the first time only 5 frames of the sequence, then 10, 15, all the way up to the maximum. This was meant as a possible generalization test, to see how well the model could perform while reducing the amount of test data that was fed to it. Results indicated a good performance often for sequences long just 15 or 10 frames.  

The file `test_class.py` is a module which modifies some classes of the CRANE code by using inheritance and adding them some additional data analysis features, like the possibility of priting some information about the numerical results of the inner layers (the "ConvGRUClassifierTEST" class). Another modified class is the "ConvGRUTEST" class, which inherits the properties of a model meant to predict the evolution of the system. In this setting the model was given some initial frames and the eigenstrain $$\eta$$ and its scope was to predict the evolution of the system for a given number of frames.
These modified classes are tested in the files `test_script.py` and `test_script_evo.py` respectively.
