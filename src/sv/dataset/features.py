"""Track input features and vertex regression targets of the SV MaskFormer (torch-free, numpy).

One place defines how every input feature and every regression target is DERIVED from the raw branches, so that the
dataset (sv.dataset.dataset_sv), the normalization fit (sv.scripts.dataset.compute_norm_par) and the evaluation plots
cannot drift apart. The scale dicts (params/sv_*_var_transform.yaml, hepattn `FeatureScaler` format) then only shift
and scale the derived values (`type: std` or `min_max_sym`), except for the vertex targets, where the dict may carry a
`fn` (`log`, `log1p`) because the prediction writer has to invert the transform: those `fn`s are applied by hepattn's
`VarTransform`, not here.

Track features (TRACK_FEATURES). The paper feeds per-track impact parameters, the pT fraction and the angles relative
to the jet axis; without jets the global kinematics are used instead. Heavy-tailed quantities are compressed with
asinh (linear below a knee, logarithmic above, sign preserving): the impact parameters and dz in units of 100 um
(IP_SCALE_CM), the significances and chi2 as they are. The -9999 "not available" markers of the NanoAOD impact
parameters (SENTINEL) are mapped to 0 before the transform, i.e. to "no measurable displacement".

Vertex targets (VERTEX_TARGETS), following the paper: the hadron pT, its direction (eta, sin/cos phi) and the flight
distance. Flight distances are measured from the generated primary vertex (PV_x/y/z of the event, see
sv.dataset.collections): `lxy` in the transverse plane, `lz` along the beam, `l3d` in space. The decay-vertex
coordinates x, y, z are available as well. A target is listed under `targets.vertex` of a config and regressed by the
model as `particle_<name>`.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np

#: values at or below this are the NanoAOD "not available" markers (-9999) of the impact parameters
SENTINEL = -9998.0
#: knee of the asinh compression of the impact parameters and dz: 100 um
IP_SCALE_CM = 0.01


def clean(x) -> np.ndarray:
    """Float copy of `x` with the -9999 markers replaced by 0."""
    x = np.asarray(x, dtype=np.float64)
    return np.where(x <= SENTINEL, 0.0, x)


def asinh_scaled(x, scale: float = 1.0) -> np.ndarray:
    """Sign-preserving compression: linear below `scale`, logarithmic above (markers cleaned first)."""
    return np.arcsinh(clean(x) / scale)


@dataclass(frozen=True)
class Feature:
    """How one feature is derived and scaled.

    name     : key of the feature (config lists, scale dicts, `particle_<name>` targets)
    branches : raw branches the derivation reads
    derive   : callable(table) -> float array, `table` a dict of flat numpy arrays keyed by branch name (vertex
               targets also receive the per-vertex primary-vertex coordinates under the PV_FIELDS keys)
    scaling  : "std" (shift by the mean, divide by the std), "min_max_sym" (map [min, max] onto [-1, 1]) or
               "identity" (already in [-1, 1]: written with mean 0 / std 1)
    fn       : transform hepattn's VarTransform applies before the shift/scale (vertex targets only: "log",
               "log1p"); the normalization fit measures the statistics of fn(derived)
    comment  : written into the scale dict
    """

    name: str
    branches: tuple[str, ...]
    derive: Callable[[dict], np.ndarray]
    scaling: str = "std"
    fn: str | None = None
    comment: str = ""


def _f(table: dict, key: str) -> np.ndarray:
    return np.asarray(table[key], dtype=np.float64)


TRACK_FEATURES: dict[str, Feature] = {
    f.name: f
    for f in (
        Feature("pt", ("track_pt",), lambda t: np.log(_f(t, "track_pt")), "std", comment="log(pT [GeV])"),
        Feature("eta", ("track_eta",), lambda t: _f(t, "track_eta"), "min_max_sym", comment="tracker acceptance"),
        Feature("sinphi", ("track_phi",), lambda t: np.sin(_f(t, "track_phi")), "identity", comment="already in [-1, 1]"),
        Feature("cosphi", ("track_phi",), lambda t: np.cos(_f(t, "track_phi")), "identity", comment="already in [-1, 1]"),
        Feature("ip2d", ("track_ip2dValue",), lambda t: asinh_scaled(t["track_ip2dValue"], IP_SCALE_CM), "std",
                comment="asinh(|d_xy| / 100 um)"),
        Feature("ip2dsig", ("track_ip2dSignificance",), lambda t: asinh_scaled(t["track_ip2dSignificance"]), "std",
                comment="asinh(|d_xy| / sigma)"),
        Feature("ip3d", ("track_ip3dValue",), lambda t: asinh_scaled(t["track_ip3dValue"], IP_SCALE_CM), "std",
                comment="asinh(|d_3D| / 100 um)"),
        Feature("ip3dsig", ("track_ip3dSignificance",), lambda t: asinh_scaled(t["track_ip3dSignificance"]), "std",
                comment="asinh(|d_3D| / sigma); -9999 markers -> 0"),
        Feature("dz", ("track_dz",), lambda t: asinh_scaled(t["track_dz"], IP_SCALE_CM), "std",
                comment="asinh(dz / 100 um), signed"),
        Feature("dzsig", ("track_dzSignificance",), lambda t: asinh_scaled(t["track_dzSignificance"]), "std",
                comment="asinh(dz / sigma), signed"),
        Feature("charge", ("track_charge",), lambda t: _f(t, "track_charge"), "identity", comment="+-1"),
        Feature("nhits", ("track_nHits",), lambda t: _f(t, "track_nHits"), "std", comment="tracker hits"),
        Feature("npixhits", ("track_numberOfPixelHits",), lambda t: _f(t, "track_numberOfPixelHits"), "std", comment="pixel hits"),
        Feature("nstriphits", ("track_numberOfStripHits",), lambda t: _f(t, "track_numberOfStripHits"), "std", comment="strip hits"),
        Feature("chi2", ("track_chi2",), lambda t: asinh_scaled(t["track_chi2"]), "std",
                comment="asinh(chi2); integer-valued in the ntuples, 60% zeros"),
        Feature("qoverp", ("track_qoverp",), lambda t: _f(t, "track_qoverp"), "std", comment="q/p [1/GeV]"),
        Feature("highpurity", ("track_isHighPurity",), lambda t: _f(t, "track_isHighPurity"), "identity", comment="0/1 flag"),
        Feature("passseed", ("track_passSeed",), lambda t: _f(t, "track_passSeed"), "identity", comment="0/1 flag (IVF seed selection)"),
    )
}

#: the per-vertex primary-vertex coordinates handed to the vertex derivations (broadcast per vertex by the caller)
PV_KEYS = ("PV_x", "PV_y", "PV_z")


def _lxy(t):
    return np.hypot(_f(t, "GV_x") - _f(t, "PV_x"), _f(t, "GV_y") - _f(t, "PV_y"))


def _lz(t):
    return _f(t, "GV_z") - _f(t, "PV_z")


VERTEX_TARGETS: dict[str, Feature] = {
    f.name: f
    for f in (
        Feature("pt", ("GV_pt",), lambda t: _f(t, "GV_pt"), "std", fn="log", comment="hadron pT [GeV] (> 10 by selection)"),
        Feature("eta", ("GV_eta",), lambda t: _f(t, "GV_eta"), "min_max_sym", comment="hadron eta, |eta| < 2.5 by selection"),
        Feature("sinphi", ("GV_phi",), lambda t: np.sin(_f(t, "GV_phi")), "identity", comment="already in [-1, 1]"),
        Feature("cosphi", ("GV_phi",), lambda t: np.cos(_f(t, "GV_phi")), "identity", comment="already in [-1, 1]"),
        Feature("lxy", ("GV_x", "GV_y", *PV_KEYS), _lxy, "std", fn="log1p",
                comment="transverse flight distance from the primary vertex [cm]"),
        Feature("lz", ("GV_z", *PV_KEYS), _lz, "std", comment="flight distance along z from the primary vertex [cm], signed"),
        Feature("l3d", ("GV_x", "GV_y", "GV_z", *PV_KEYS), lambda t: np.sqrt(_lxy(t) ** 2 + _lz(t) ** 2), "std", fn="log1p",
                comment="3D flight distance from the primary vertex [cm]"),
        Feature("x", ("GV_x",), lambda t: _f(t, "GV_x"), "std", comment="decay vertex x [cm]"),
        Feature("y", ("GV_y",), lambda t: _f(t, "GV_y"), "std", comment="decay vertex y [cm]"),
        Feature("z", ("GV_z",), lambda t: _f(t, "GV_z"), "std", comment="decay vertex z [cm]"),
    )
}


def derive(features: dict[str, Feature], names, table: dict) -> dict[str, np.ndarray]:
    """Derived values of the features `names` (keys of `features`) from `table` (flat numpy arrays per branch)."""
    unknown = [n for n in names if n not in features]
    if unknown:
        raise ValueError(f"unknown features {unknown}; known: {sorted(features)}")
    return {name: np.asarray(features[name].derive(table), dtype=np.float64) for name in names}


def branches_needed(features: dict[str, Feature], names) -> list[str]:
    """Raw branches the features `names` read, in first-use order."""
    seen: dict[str, None] = {}
    for name in names:
        for b in features[name].branches:
            seen.setdefault(b, None)
    return list(seen)


def apply_fn(fn: str | None, x: np.ndarray) -> np.ndarray:
    """The `fn` of a scale-dict entry, as hepattn's VarTransform applies it (numpy mirror, for the fit and the plots)."""
    if fn is None:
        return x
    if fn == "log":
        return np.log(x)
    if fn == "log1p":
        return np.log1p(x)
    if fn == "sqrt":
        return np.sqrt(x)
    raise ValueError(f"unknown transform fn {fn!r}")


def invert_fn(fn: str | None, x: np.ndarray) -> np.ndarray:
    if fn is None:
        return x
    if fn == "log":
        return np.exp(x)
    if fn == "log1p":
        return np.expm1(x)
    if fn == "sqrt":
        return np.square(x)
    raise ValueError(f"unknown transform fn {fn!r}")
