(references)=

# Citations & References

The bibtex entries for **PyAutoGalaxy** and its affiliated software packages can be found
[here](https://github.com/PyAutoLabs/PyAutoGalaxy/blob/main/files/citations.bib), with example text for citing **PyAutoGalaxy**
in [.tex format here](https://github.com/PyAutoLabs/PyAutoGalaxy/blob/main/files/citations.tex) format here and
[.md format here](https://github.com/PyAutoLabs/PyAutoGalaxy/blob/main/files/citations.md).

**PyAutoGalaxy** is published in the [Journal of Open Source Software](https://joss.theoj.org/papers/10.21105/joss.04475#) and its
entry in the above .bib file is under the citation key `pyautogalaxy`.

As shown in the examples, we would greatly appreciate it if you mention **PyAutoGalaxy** by name and include a link to
our GitHub page!

You should also specify the non-linear search(es) you use in your analysis (e.g. Dynesty, Emcee, PySwarms, etc) in
the main body of text, and delete as appropriate any packages your analysis did not use. The citations.bib file includes
the citation key for all of these projects.

## Jax-Zero-Contour

If you use the zero-contour method for critical curve and caustic computation (the default in
`visualize/general.yaml` via `critical_curves_method: zero_contour`), please cite the
`Jax-Zero-Contour` package by Coleman Krawczyk:

```bibtex
@software{coleman_krawczyk_2025_15730415,
  author       = {Coleman Krawczyk},
  title        = {CKrawczyk/Jax-Zero-Contour: Version 2.0.0},
  month        = jun,
  year         = 2025,
  publisher    = {Zenodo},
  version      = {v2.0.0},
  doi          = {10.5281/zenodo.15730415},
  url          = {https://doi.org/10.5281/zenodo.15730415},
}
```

The package is available at <https://github.com/CKrawczyk/Jax-Zero-Contour> and archived at
<https://doi.org/10.5281/zenodo.15730415>.

## NUFFTax

If you fit interferometer datasets on the JAX path, the non-uniform FFT is performed by
`nufftax`, a pure-JAX NUFFT implementation by the GragasLab team. Please cite the
package:

```bibtex
@software{nufftax,
  author = {Gragas and Oudoumanessah, Geoffroy and Iollo, Jacopo},
  title  = {nufftax: Pure JAX implementation of the Non-Uniform Fast Fourier Transform},
  url    = {https://github.com/GragasLab/nufftax},
  year   = {2026},
}
```

`nufftax`'s algorithm is based on FINUFFT (Flatiron Institute); the upstream paper should
also be cited:

```bibtex
@article{finufft,
  author  = {Barnett, Alexander H. and Magland, Jeremy F. and af Klinteberg, Ludvig},
  title   = {A parallel non-uniform fast Fourier transform library based on an
             'exponential of semicircle' kernel},
  journal = {SIAM J. Sci. Comput.},
  volume  = {41},
  number  = {5},
  pages   = {C479--C504},
  year    = {2019},
}
```

The package is available at <https://github.com/GragasLab/nufftax>.

## Rectangular Mesh (Pixelized Reconstructions)

If you reconstruct a galaxy using the adaptive rectangular meshes (`RectangularAdaptDensity` or
`RectangularAdaptImage`), you **must cite** the following paper <https://arxiv.org/abs/2606.30620> under citation
key `Enzi2026`. The mesh's adaptive coordinate transform implements the ray-guided transformed uniform (RTU)
grid formulation this paper introduces, which is what makes the pixelization adaptive and fully
auto-differentiable:

```bibtex
@article{Enzi2026,
  author        = {Enzi, Wolfgang J. R. and Krawczyk, Coleman M. and Li, Tian and Collett, Thomas E.},
  title         = {Gaussian processes on ray-guided transformed uniform grids for fast, flexible, and auto-differentiable adaptive source reconstruction in lens modelling},
  journal       = {MNRAS, submitted},
  eprint        = {2606.30620},
  archivePrefix = {arXiv},
  primaryClass  = {astro-ph.GA},
  year          = {2026},
  url           = {https://arxiv.org/abs/2606.30620},
}
```

Note that **PyAutoGalaxy** pairs the RTU grid with its own regularization schemes (e.g. `reg.Constant`,
`reg.Adapt`) rather than the Gaussian-process prior used in the paper, so quantitative results are not
directly comparable between the two implementations. The `RectangularUniform` mesh performs no RTU transform,
so this citation is not required when only the uniform mesh is used.

## Dynesty

If you used the nested sampling algorithm Dynesty, please follow the citation instructions [on the dynesty readthedocs](https://dynesty.readthedocs.io/en/latest/references.html).
