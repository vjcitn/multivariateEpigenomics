# multivariateEpigenomics

This package is intended to help with exploration of high-dimensional
epigenomic data such as the Illumina EPIC array.

In version 0.0.1, we include Julia code implementing
an NMF algorithm and have a vignette comparing PCA and
two NMF implementations.

To install the code, the following would succeed, but
other approaches are possible.  You would need
`devtools` installed and must use R 4.3 or later, and
must also have a recent version of julia.

```
devtools::install_github("vjcitn/multivariateEpigenomics", dependencies=TRUE)
```
