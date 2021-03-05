# Machine Learning of PDEs
This repository contains the code of my Master thesis with the title "Physics Informed Machine Learning of Nonlinear Partial Differential Equations".

The class definitions for the numerical and the machine learning solver are found in [numerical_solvers] and [machine_learning_solvers].

All results of the work can be recreated by running the files in the [scripts] folder. Note that for some of the numerical results you will need the exact solution to the Burgers' equation at a certain granularity. Check out the [burgers_exact] folder for more information on how to generate those.

[scripts]: https://github.com/sch0ngut/machine-learning-of-pdes/tree/main/scripts
[burgers_exact]: https://github.com/sch0ngut/machine-learning-of-pdes/tree/main/burgers_exact
[numerical_solvers]: https://github.com/sch0ngut/machine-learning-of-pdes/tree/main/numerical_solvers
[machine_learning_solvers]: https://github.com/sch0ngut/machine-learning-of-pdes/blob/main/machine_learning_solver/PINN.py

## Abstract
Finding approximate solutions to nonlinear partial differential equations given some initial and boundary conditions is a well studied task within the field of numerical analysis. Nevertheless, numerical methods face several limitations with respect to the complexity of the underlying problem and the related computational effort.
In this work we aim to investigate whether methods from another discipline, namely machine learning, can solve such tasks and overcome the limitations of the numerical approaches. Specifically we consider physics informed neural networks, a recently discovered method that allows the encoding of the underlying partial differential equation directly into the loss function of a neural network by applying automatic differentiation with respect to the network's inputs. Our investigations suggest that these networks are indeed able to detect the imposed dynamics and overcome numerical issues related to stability and discontinuities.
The main result of this work is the discovery of a strong dependence between the overall performance of the method and the performance on the initial condition. We encountered that this dependence makes physics informed neural networks prone to produce wrong solutions if a small change in the initial condition can lead to a substantially different evolution of the system. This observation suggests that machine learning solvers are not yet ready to fully replace numerical approaches. Nevertheless, the achieved performance is very impressive and gives rise to the hope that this is just the beginning of an entirely new branch in scientific computing, i.e. solving partial differential equations using machine learning.
