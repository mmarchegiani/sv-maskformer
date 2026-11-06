"""Collection knowledge of the SV (secondary-vertex) dataset, shared by the converter, the dataset, the normalization
fit, the prediction writer and the plots. Torch-free on purpose (the plots run in an LCG view).

The input files are the BTV secondary-vertex ntuples (a NanoAOD-like `Events` tree, e.g.
/eos/user/m/mmarcheg/BTV/sv_reconstruction/ttbar_BTV_1.root, 5000 ttbar events), converted to parquet by
`maskformer-sv convert` (sv.scripts.dataset.convert_root) into one record per event with the layout of the HGCAL
files: one top-level struct column per collection, each field a jagged array with the full NanoAOD branch name.

Collections (as measured on ttbar_BTV_1.root):

`track`  -- every reconstructed track of the event (mean 390, max 847 per event; pT > 0.8 GeV for 99% of them,
            no jet association). Kinematics (pt, eta, phi, charge, qoverp), hit content (nHits, pixel/strip hits,
            chi2), the impact parameters with respect to the primary vertex (`ip2dValue`/`ip3dValue` in cm,
            unsigned, and their significances, `dz` in cm and `dzSignificance`) and two quality flags
            (`isHighPurity`, 99%; `passSeed`, 89%, the IVF seed-track selection). `timeSig` is -9999 everywhere.
            The "not available" marker of the impact parameters is -9999 (1.3% of the tracks for
            ip3dSignificance, never a track from a heavy-flavour vertex); see sv.dataset.features.
            Truth: `track_hadidx` is the index, within the event's `GV` collection, of the heavy-flavour vertex
            the track comes from (-1 for 98% of the tracks); `track_label` summarises it: 0 and 1 are tracks from
            no GV (0: tracks with a wide dz spread, 1: mostly pileup-like tracks, |dz| > 2 cm for 69%; their
            exact definition is not documented in the file), 2 from a b-hadron decay, 3 from a c-hadron that
            comes from a b-hadron, 4 from a prompt c-hadron.

`GV`     -- the gen-level heavy-flavour decay vertices (mean 2.7, max 11 per event; 4% of the events have none):
            one per b- or c-hadron (and a handful of 3-prong taus). The candidate vertices are selected by the
            converter (VERTEX_SELECTION / DAUGHTER_SELECTION, `select_vertices`): pT > 10 GeV, |eta| < 2.5 and at
            least two daughters that are charged, stable (a generator-stable charged species, STABLE_CHARGED_PDGIDS:
            the file carries no status), with pT > 0.8 GeV and |eta| < 2.5, counted from the `GVDaughters`
            collection (`GV_nSelectedDaughters`; `GV_nDaughters` is the producer's own count). The ntuple producer
            already applies this selection to the `GV` collection: on ttbar_BTV_1.root all 13,628 vertices pass and
            `GV_nDaughters` is exactly the number of stored daughters, all of them charged, stable, pT > 0.8 GeV,
            |eta| < 2.5. Vertices failing it are dropped and the `track_hadidx` of their tracks reset to -1
            (`track_label` to 0). Position of the decay vertex
            (`GV_x/y/z`, cm) and of the hadron's production vertex (`GV_x_i/y_i/z_i`: the primary vertex for a
            prompt hadron, the b-hadron decay vertex for a c-hadron of a b -> c chain), the hadron kinematics
            (`GV_pt/eta/phi`), its identity (`GV_Hadron_pdgId`; `GV_pdgClass` holds the same value) and the
            flags `GV_isB`, `GV_isD` (b-/c-hadron), `GV_fromHF` (the hadron is a decay product of another
            heavy-flavour hadron) and `GV_toHF` (it decays to one). On ttbar_BTV_1.root 91% of the vertices
            have at least two reconstructed tracks pointing to them (`track_hadidx`), 99% at least one.
            The decay vertex matches `GVDaughters_vx/vy/vz` and the tracks match the `GVDaughters` of the
            same `hadronIndex` in pT, eta, phi and charge (checked); `trackPair_edgeLabel` (the edge-classifier
            baseline) is exactly "same `track_hadidx` >= 0".

`PV`     -- the generated primary vertex of the hard scatter (written by the converter from the vertex of the
            first status-21 GenPart; it coincides with `GV_x_i` of the prompt hadrons). The flight distances of
            the vertices (`lxy`, `lz`) are measured from it, see sv.dataset.features.

The vertex classes of the model (hepattn's `ObjectClassificationTask`, null class at index NUM_CLASSES):
0 B (`GV_isB`), 1 BtoC (`GV_isD` and `GV_fromHF`: a c-hadron from a b-hadron decay), 2 C (`GV_isD`, not from a
b-hadron), 3 Other (taus and the rest). On the train split: 32% / 45% / 23% / 0.4% of the vertices.
"""

from __future__ import annotations

import numpy as np

#: name of the track (input constituent) and vertex (target object) collections in the parquet files
TRACK_COLLECTION = "track"
VERTEX_COLLECTION = "GV"

#: per-event scalar branches kept by the converter
EVENT_BRANCHES = ("run", "luminosityBlock", "event")
#: the generated primary vertex (per-event scalars written by the converter)
PV_FIELDS = ("PV_x", "PV_y", "PV_z")

#: raw NanoAOD branches of the track collection read by the converter (the dataset reads the subset its features need)
TRACK_BRANCHES = (
    "track_pt", "track_eta", "track_phi", "track_energy", "track_px", "track_py", "track_pz",
    "track_charge", "track_qoverp", "track_qdotp",
    "track_nHits", "track_numberOfPixelHits", "track_numberOfStripHits", "track_chi2",
    "track_isHighPurity", "track_passSeed",
    "track_dz", "track_dzSignificance",
    "track_ip2dValue", "track_ip2dSignificance", "track_ip3dValue", "track_ip3dSignificance",
    "track_timeSig",
    "track_hadidx", "track_label",
)
#: per-track index of the vertex it comes from (-1: none) and the summary label
TRACK_TRUTH_IDX = "track_hadidx"
TRACK_TRUTH_LABEL = "track_label"

#: raw NanoAOD branches of the vertex collection
VERTEX_BRANCHES = (
    "GV_Hadron_pdgId", "GV_isB", "GV_isD", "GV_fromHF", "GV_toHF", "GV_pdgClass", "GV_nDaughters",
    "GV_pt", "GV_eta", "GV_phi",
    "GV_x", "GV_y", "GV_z", "GV_x_i", "GV_y_i", "GV_z_i",
    "GV_maxDaughterPairDeltaR",
)
#: truth fields the converter adds to the vertex collection (torch-free derivations, see `vertex_truth_fields`)
VERTEX_DERIVED = ("GV_class", "GV_nTracks", "GV_nSelectedDaughters",
                  "GV_lxy", "GV_lz", "GV_l3d", "GV_flight_xy", "GV_flight_3d")

# ---- candidate vertex selection -------------------------------------------------------------------------------

#: selection of the candidate vertices (applied by the converter, checked by the dataset): hadron pT and |eta|, and
#: the minimum number of daughters passing DAUGHTER_SELECTION
VERTEX_SELECTION = {"min_pt": 10.0, "max_abs_eta": 2.5, "min_daughters": 2}
#: a daughter counts when charged, stable (STABLE_CHARGED_PDGIDS), with this pT and |eta|
DAUGHTER_SELECTION = {"min_pt": 0.8, "max_abs_eta": 2.5}
#: generator-stable charged particles (Pythia 8 leaves particles with c*tau > 10 mm undecayed: e, mu, pi, K, p,
#: Sigma+-, Xi-, Omega-); the daughter collection carries no status, so "stable" is this species list
STABLE_CHARGED_PDGIDS = frozenset({11, 13, 211, 321, 2212, 3112, 3222, 3312, 3334})
#: branches of the daughter collection the converter needs for the selection (always read)
DAUGHTER_COLLECTION = "GVDaughters"
DAUGHTER_BRANCHES = ("GVDaughters_hadronIndex", "GVDaughters_charge", "GVDaughters_pt", "GVDaughters_eta", "GVDaughters_pdgId")


def select_daughters(charge, pt, eta, pdg_id) -> np.ndarray:
    """Mask of the daughters that count for the vertex selection: charged, stable species, pT and |eta| cuts
    (DAUGHTER_SELECTION)."""
    charge = np.asarray(charge)
    pdg_id = np.abs(np.asarray(pdg_id).astype(np.int64))
    stable = np.isin(pdg_id, np.fromiter(STABLE_CHARGED_PDGIDS, dtype=np.int64))
    return (charge != 0) & stable & (np.asarray(pt) > DAUGHTER_SELECTION["min_pt"]) & (np.abs(np.asarray(eta)) < DAUGHTER_SELECTION["max_abs_eta"])


def vertex_selection_flags(pt, eta, n_selected_daughters) -> dict[str, np.ndarray]:
    """The three requirements of VERTEX_SELECTION per vertex (name -> mask), in cutflow order."""
    return {
        f"pt > {VERTEX_SELECTION['min_pt']:g}": np.asarray(pt) > VERTEX_SELECTION["min_pt"],
        f"|eta| < {VERTEX_SELECTION['max_abs_eta']:g}": np.abs(np.asarray(eta)) < VERTEX_SELECTION["max_abs_eta"],
        f"selected daughters >= {VERTEX_SELECTION['min_daughters']}": np.asarray(n_selected_daughters) >= VERTEX_SELECTION["min_daughters"],
    }


def select_vertices(pt, eta, n_selected_daughters) -> np.ndarray:
    """Mask of the candidate vertices (VERTEX_SELECTION): hadron pT, |eta| and the number of selected daughters."""
    flags = vertex_selection_flags(pt, eta, n_selected_daughters)
    return np.logical_and.reduce(list(flags.values()))

#: optional collections the converter can keep (--keep): read only when asked for
OPTIONAL_COLLECTIONS = {
    "GVDaughters": ("GVDaughters_charge", "GVDaughters_pdgId", "GVDaughters_statusFlags", "GVDaughters_hadronIndex",
                    "GVDaughters_pt", "GVDaughters_eta", "GVDaughters_phi", "GVDaughters_vx", "GVDaughters_vy", "GVDaughters_vz"),
    "RejectedGV": ("RejectedGV_pt", "RejectedGV_eta", "RejectedGV_phi", "RejectedGV_x", "RejectedGV_y", "RejectedGV_z",
                   "RejectedGV_x_i", "RejectedGV_y_i", "RejectedGV_z_i"),
    "trackPair": ("trackPair_trk1Idx", "trackPair_trk2Idx", "trackPair_deltaR", "trackPair_dca", "trackPair_dcaSig",
                  "trackPair_cptopv", "trackPair_pvtoPCA1", "trackPair_pvtoPCA2", "trackPair_dotprod1", "trackPair_dotprod2",
                  "trackPair_pairMom", "trackPair_pairInvMass", "trackPair_pt", "trackPair_eta", "trackPair_edgeLabel"),
}
#: GenPart branches the converter reads to locate the generated primary vertex
GENPART_PV_BRANCHES = ("GenPart_status", "GenPart_vx", "GenPart_vy", "GenPart_vz")
#: status of the GenParts produced at the hard-scatter vertex (Pythia 8: incoming partons of the hardest subprocess)
GENPART_HARD_STATUS = 21

# ---- vertex classes --------------------------------------------------------------------------------------------

#: the object classes: B (b-hadron), BtoC (c-hadron from a b-hadron decay), C (c-hadron not from a b-hadron), Other
#: (taus and the rest). hepattn's ObjectClassificationTask puts the null class at index NUM_CLASSES, which is
#: therefore the residual (not a target / fake) class, as in the HGCAL dataset.
CLASS_NAMES = ("B", "BtoC", "C", "Other")
NUM_CLASSES = len(CLASS_NAMES)
RESIDUAL_CLASS = NUM_CLASSES
CLASS_B, CLASS_BTOC, CLASS_C, CLASS_OTHER = range(NUM_CLASSES)


def vertex_class(is_b, is_d, from_hf) -> np.ndarray:
    """Class index of each vertex from the `GV_isB` / `GV_isD` / `GV_fromHF` flags: B (0), BtoC (1), C (2),
    Other (3)."""
    is_b = np.asarray(is_b).astype(bool)
    is_d = np.asarray(is_d).astype(bool)
    from_hf = np.asarray(from_hf).astype(bool)
    return np.where(is_b, CLASS_B, np.where(is_d & from_hf, CLASS_BTOC, np.where(is_d, CLASS_C, CLASS_OTHER))).astype(np.int64)


def track_counts_per_vertex(hadidx: np.ndarray, n_vertices: int) -> np.ndarray:
    """Number of tracks pointing to each of the `n_vertices` vertices of one event (`hadidx`: the event's
    `track_hadidx` values, -1 for none)."""
    hadidx = np.asarray(hadidx)
    valid = hadidx >= 0
    if n_vertices == 0:
        return np.zeros(0, dtype=np.int64)
    if valid.any() and int(hadidx[valid].max()) >= n_vertices:
        raise ValueError(f"track_hadidx {int(hadidx[valid].max())} >= number of vertices {n_vertices}")
    return np.bincount(hadidx[valid], minlength=n_vertices).astype(np.int64)
