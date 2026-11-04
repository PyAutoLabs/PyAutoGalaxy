# Chapter 3: Pixelizations

In chapter 3, we reconstruct complex galaxy light on pixelized grids. We use Bayesian evidence to choose the smoothing strength, investigate fits that remain inadequate, and adapt the mesh and regularization to the light distribution.

The chapter contains the following tutorials:

[Tutorial 1: Pixelizations, Mappers and Inversions](https://colab.research.google.com/github/PyAutoLabs/HowToGalaxy/blob/main/notebooks/chapter_3_pixelizations/tutorial_1_pixelizations.ipynb)
\- Connect mesh points to image pixels and reconstruct complex galaxy light with a linear inversion.

[Tutorial 2: Bayesian Regularization](https://colab.research.google.com/github/PyAutoLabs/HowToGalaxy/blob/main/notebooks/chapter_3_pixelizations/tutorial_2_bayesian_regularization.ipynb)
\- Compare log evidences to choose the regularization strength and examine its dependence on mesh resolution.

[Tutorial 3: Fit Problems](https://colab.research.google.com/github/PyAutoLabs/HowToGalaxy/blob/main/notebooks/chapter_3_pixelizations/tutorial_3_fit_problems.ipynb)
\- See why compact galaxy light challenges mesh resolution and constant regularization, even after choosing the coefficient using evidence.

[Tutorial 4: Adaptive Reconstruction](https://colab.research.google.com/github/PyAutoLabs/HowToGalaxy/blob/main/notebooks/chapter_3_pixelizations/tutorial_4_adaptive_reconstruction.ipynb)
\- Combine Delaunay meshes, Hilbert brightness adaptation and adaptive Split regularization for directly observed galaxies.

[Tutorial 5: Model Fit](https://colab.research.google.com/github/PyAutoLabs/HowToGalaxy/blob/main/notebooks/chapter_3_pixelizations/tutorial_5_model_fit.ipynb)
\- Combine a smooth analytic light component with a pixelization in a model fit.
