<img src="docs/images/SSMLearnLogo.png" width="350" align="right">

<h1 style="font-family:Helvetica;" align="left">
    SSMLearnPy
</h1>

## Data-driven Reduced Order Models for Nonlinear Dynamical Systems

This package perform data-driven identification of reduced order model based on spectral submanifolds (SSMs). The required input consists of trajectory data of generic system observables close to an SSM, the SSM dimension and the methods for manifold learning and reduced dynamics fitting.

The computational steps for achieving a reduced-order model are:

1. Embedding of the simulation/measurement data in a suitable observable space;
2. Computation of the invariant manifold parametrization and its reduced order or latent coordinates;
3. Identification of the reduced dynamics, with initial implementation of normal form transformation.

The code can run analytics and predictions on the reduced-order model, as shown in our examples. There are no constraints on the types of masurements, on the kind of nonlinearities or on the problem dimensions.

We have included a demonstration of SSM computation over the following examples.

- Oscillator chain: n degrees of freedom with trajectories on or off specific SSMs;
- Brake-Reuss beam: benchmark system for the dynamics of jointed structures, data from experimental measurements (DIC and accelerometers);
- Vortex Shedding behind a cylinder: data from CFD simulations, projected on a low-dimensional linear subspace of the phase space;
- Couette flow: data from CFD simulations, projected on a low-dimensional linear subspace of the phase space, for different regimes;
- Liquid sloshing of a water tank: data from experimental measurements.
- One degree of freedom oscillator, as a benchmark problem;

## Installation
1. Clone the project with 
    ```sh
    git clone git@github.com:mattiacenedese/SSMLearnPy.git
    ```
2. Enter in the project folder 
    ```sh
    cd SSMLearnPy/
    ```
3. (optional) Create a new conda environment and activate it:
    ```sh
    conda create -n ssmlearn python=3.9
    conda activate ssmlearn
    ```
4. Install dependencies:
    ```sh
    pip install -r requirements.txt
    ```
5. Install the package:
    ```sh
    pip install -e .
    ```
6. You are ready.

## References
Please consider to cite this article when using this code:

- M. Cenedese, J. Axås, B. Bäuerlein, K. Avila and G. Haller. Data-driven modeling and prediction of non-linearizable dynamics via spectral submanifolds. [*Nature Communications*](https://doi.org/10.1038/s41467-022-28518-y), **13** (2022) 872. [[PDF]](https://www.nature.com/articles/s41467-022-28518-y.pdf) [[Supplementary Information]](https://static-content.springer.com/esm/art%3A10.1038%2Fs41467-022-28518-y/MediaObjects/41467_2022_28518_MOESM1_ESM.pdf)

Additional works appear in the references:

- M. Cenedese, J. Axås, H. Yang, M. Eriten and G. Haller. Data-driven nonlinear model reduction to spectral submanifolds in mechanical systems, [*Philosophical Transactions of the Royal Society A: Mathematical, Physical and Engineering Sciences*](https://doi.org/10.1038/s41467-022-28518-y) **380** (2022) 20210194. [[PDF]](http://www.georgehaller.com/reprints/Cenedeseetal_DataDrivenNonlinearModelReduction.pdf) 

- G. Haller, S. Jain and M. Cenedese. Dynamics-based machine learning for nonlinearizable phenomena. Data-driven reduced models on spectral submanifolds, [*SIAM News*](https://sinews.siam.org/Details-Page/dynamics-based-machine-learning-for-nonlinearizable-phenomena) **55/5** (2022) 20210194. [[PDF]](http://www.georgehaller.com/reprints/HallerJainCenedese_dynamics_based_machine_learning.pdf) 

- B. Kaszás, M. Cenedese & G. Haller, Dynamics-based machine learning of transitions in Couette flow, [*Physical Review Fluids*](https://link.aps.org/doi/10.1103/PhysRevFluids.7.L082402), **7** (2022) L082402. [[PDF]](http://www.georgehaller.com/reprints/dynamicsbasedmachinelearning.pdf) [[Supplemental Material]](http://www.georgehaller.com/reprints/dynamicsbasedmachinelearning_supp.pdf)

- J.I. Alora, M. Cenedese, E. Schmerling, G. Haller & M. Pavone, Data-driven spectral submanifold reduction for nonlinear optimal control of high-dimensional robots (2022) [*arxiv.org/2209.0571*](https://arxiv.org/2209.0571). [[PDF]](https://arxiv.org/pdf/2209.0571.pdf) 
