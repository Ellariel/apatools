# Software and what to cite

apatools wraps other software: the numbers in a table come from an *engine*, and the engine is what a reader needs to know
about (and what you should cite) next to apatools itself. The references below were checked against Crossref; check the
year and the version of the software you used. The literature behind the *methods* is at the end of the page of each module.

## What to cite for which engine

| module | engine | cite |
|---|---|---|
| `lm` | `statsmodels` | statsmodels |
| `slm` | `spreg` | PySAL (spreg, libpysal) |
| `plm` (`backend=`) | `spreg` | PySAL (spreg, libpysal) |
| `plm` (`backend=`) | `linearmodels` | PySAL (spreg, libpysal), linearmodels |
| `plm` (`backend=`) | `splm` | R, plm (R), splm (R), spdep (R) |
| `sem` / `sem_plot` (`engine=`) | `semopy` | semopy |
| `sem` / `sem_plot` (`engine=`) | `lavaan` | R, lavaan (R) |
| `sem` / `sem_plot` (`engine=`) | `semPlot` | R, semPlot (R) |
| `sem` / `sem_plot` (`engine=`) | `networkx` | NetworkX, matplotlib |
| `sem` / `sem_plot` (`engine=`) | `graphviz` | Graphviz |
| `psm` | `psmpy` | psmpy |

The default engine of a module is the first one listed for it. Cite every engine you used, for fitting and for drawing alike.

## References

**statsmodels**

Seabold, S., & Perktold, J. (2010). Statsmodels: Econometric and statistical modeling with Python. In S. van der Walt & J. Millman (Eds.), Proceedings of the 9th Python in Science Conference (pp. 92–96). https://doi.org/10.25080/Majora-92bf1922-011

**PySAL (spreg, libpysal)**

Rey, S. J., & Anselin, L. (2007). PySAL: A Python library of spatial analytical methods. The Review of Regional Studies, 37(1), 5–27. https://doi.org/10.52324/001c.8285

**linearmodels**

Sheppard, K., & contributors. (2026). linearmodels (v7.1) [Computer software]. Zenodo. https://doi.org/10.5281/zenodo.2591973

**semopy**

Igolkina, A. A., & Meshcheryakov, G. (2020). semopy: A Python package for structural equation modeling. Structural Equation Modeling: A Multidisciplinary Journal, 27(6), 952–963. https://doi.org/10.1080/10705511.2019.1704289

**lavaan (R)**

Rosseel, Y. (2012). lavaan: An R package for structural equation modeling. Journal of Statistical Software, 48(2), 1–36. https://doi.org/10.18637/jss.v048.i02

**semPlot (R)**

Epskamp, S. (2015). semPlot: Unified visualizations of structural equation models. Structural Equation Modeling: A Multidisciplinary Journal, 22(3), 474–483. https://doi.org/10.1080/10705511.2014.937847

**R**

R Core Team. (2025). R: A language and environment for statistical computing [Computer software]. R Foundation for Statistical Computing. https://www.R-project.org/

**plm (R)**

Croissant, Y., & Millo, G. (2008). Panel data econometrics in R: The plm package. Journal of Statistical Software, 27(2), 1–43. https://doi.org/10.18637/jss.v027.i02

**splm (R)**

Millo, G., & Piras, G. (2012). splm: Spatial panel data models in R. Journal of Statistical Software, 47(1), 1–38. https://doi.org/10.18637/jss.v047.i01

**spdep (R)**

Bivand, R. S., & Wong, D. W. S. (2018). Comparing implementations of global and local indicators of spatial association. TEST, 27(3), 716–748. https://doi.org/10.1007/s11749-018-0599-x

**psmpy**

Kline, A., & Luo, Y. (2022). PsmPy: A package for retrospective cohort matching in Python. In 2022 44th Annual International Conference of the IEEE Engineering in Medicine & Biology Society (EMBC) (pp. 1354–1357). https://doi.org/10.1109/EMBC48229.2022.9871333

**NetworkX**

Hagberg, A. A., Schult, D. A., & Swart, P. J. (2008). Exploring network structure, dynamics, and function using NetworkX. In G. Varoquaux, T. Vaught, & J. Millman (Eds.), Proceedings of the 7th Python in Science Conference (pp. 11–15). https://doi.org/10.25080/TCWV9851

**matplotlib**

Hunter, J. D. (2007). Matplotlib: A 2D graphics environment. Computing in Science & Engineering, 9(3), 90–95. https://doi.org/10.1109/MCSE.2007.55

**Graphviz**

Gansner, E. R., & North, S. C. (2000). An open graph visualization system and its applications to software engineering. Software: Practice and Experience, 30(11), 1203–1233. https://doi.org/10.1002/1097-024X(200009)30:11<1203::AID-SPE338>3.0.CO;2-N

Other software used without being an engine: numpy, scipy and pandas (see their citation pages), and patsy and
scikit-learn where a module needs them. The licenses of the dependencies are those of their packages.
