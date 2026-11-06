"""Secondary-vertex (SV) reconstruction with MaskFormers on CMS simulation: dataset, prediction writer, plots and entry points.

Reproduces the setup of "Secondary vertex reconstruction with MaskFormers" (Van Stroud et al., Eur. Phys. J. C 84 (2024)
1020, arXiv:2312.12272) on CMS ttbar simulation: the reconstructed tracks of an event are the input constituents, the
heavy-flavour decay vertices (the `GV` collection of the BTV ntuples) are the objects the model finds, assigns tracks
to, classifies (b / c / other) and fits (pT, direction, flight distance). Imports `maskformer` only, never `hgcal`.
"""
