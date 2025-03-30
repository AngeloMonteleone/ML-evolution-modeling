# ML-evolution-modeling
This repository is meant to showcase my Bachelor's Thesis _Machine Learning Modeling of the Microstructural Evolution of Strained Materials_. Machine Learning (ML) methods were employed to achieve first and foremost the extraction of a parameter characterizing the evolution of a system from a sequence of snapshots and here I present a synthesis of my work.

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


<!--## Phase separation in binary alloys
A binary alloy at constant pressure and temperature may exhibit spontaneous phase separation through spinodal decomposition under a certain threshold temperature. This happens because the system tends to minimize Gibbs free energy:

$G^{\text{mixed}}=G^{\text{pure}}+\Delta H -T\Delta S$

which is constituted by two terms: an entropic term, which favours the mixing and an enthalpic term, which may push towards separation, depending on the components of the system. When this happens some regions of space will be characterized by the prevalence of one of the two atomic species, whereas others by the lack thereof.

This system may be further analysed by the Cahn-Hilliard equation, which also includes the cost of the interface between the phases. It permits to treat the dynamics through the evolution of an order parameter, a convenient way to map the concentration fields into a single scalar field. The equation is the following:

$\frac{\partial\varphi}{\partial t} =\nabla\cdot\left(M\nabla\left(\frac{\delta G}{\delta \varphi}\right)\right)$

where the unknown $$\varphi$$ is a scalar field representing the local concentration: the more an atomic species prevails, the closer will $$\varphi$$ be to 1. It depends on the functional derivative of the free-energy functional $$G[\varphi]$$:

$G[\varphi] = \int_{\Omega}[k|\nabla\varphi|^2+g_B(\varphi)+\rho(\varphi)]d^3x$

where the first term represents the energy cost of having an interface with a certain shape, the second is the bulk energy of the mixture and the third is the elastic energy density, and it is nonzero when the system is subject to strain and stresses. This last term, for a simple volumetric compression is equal to:

$$\rho(\varphi)=\frac{1}{2}C_{ijkl}(\varphi)(\varepsilon_{ij}-\varepsilon^0_{ij}(\varphi))(\varepsilon_{kl}-\varepsilon^0_{kl}(\varphi))$$

where the eig
-->
