# Secondary-vertex reconstruction (`sv`): tracks -> heavy-flavour decay vertices

MaskFormer reconstruction of the heavy-flavour decay vertices of an event from its reconstructed tracks, after
[Secondary vertex reconstruction with MaskFormers](https://link.springer.com/article/10.1140/epjc/s10052-024-13374-5)
(Van Stroud et al., EPJC 84 (2024) 1020, [arXiv:2312.12272](https://arxiv.org/abs/2312.12272)), on CMS ttbar
simulation. The code lives in `src/sv/` (reusing `src/maskformer/`, never `src/hgcal/`), the command is
`maskformer-sv` (`python -m sv` without the installed package).

| config | model | purpose |
|---|---|---|
| `gv_tracks.yaml` | dim 256, 4 encoder + 3 decoder layers, 8 heads, 16 queries, AdamW/OneCycle, 50 epochs (the paper's model) | training on GPU |
| `gv_tracks_small.yaml` | dim 64, 2 + 2 layers, torch attention, 200 train events, CPU | smoke tests of the pipeline |

## The problem, mapped onto the MaskFormer

| | paper (ATLAS-like Delphes, per jet) | here (CMS ttbar ntuples, per event) |
|---|---|---|
| input constituents | the 50 highest-pT tracks of the jet | all reconstructed tracks of the event (`track` collection: mean 390, max 847) |
| track features | d0, z0, pT fraction, eta/phi relative to the jet, jet pT/eta | global kinematics, impact parameters and significances, dz, hit content, quality flags (`sv.dataset.features.TRACK_FEATURES`) |
| objects (queries) | b- and c-hadron decay vertices, 5 queries | the `GV` collection: b-/c-hadron (and 3-prong tau) decay vertices with pT > 10 GeV, |eta| < 2.5, >= 2 charged daughters; mean 2.7, max 11 per event; 16 queries |
| truth track-vertex link | truth association | `track_hadidx`, the index of the vertex each track comes from (-1 for 98% of the tracks) |
| vertex classes | b, c, null | b, c, other (taus, rare baryons), null |
| regression | hadron pT, Lxy (from the beam line), dR to the jet axis | hadron pT, eta, sin/cos phi, lxy and lz (flight distance from the generated primary vertex); also available: l3d, x, y, z |
| losses | 0.5 CE + 2 L1 + 12 mask BCE + 8 mask Dice, Hungarian matching | the same weights; the matching cost is the same combination |
| evaluation | perfect / loose vertex-finding efficiency vs Lxy, fake rate, fit residuals | the same, plus the track assignment and the class confusion (`maskformer-sv plot`) |

The data and the truth definitions are documented in `src/sv/dataset/collections.py`, the feature derivations in
`src/sv/dataset/features.py`. On `ttbar_BTV_1.root` 91% of the vertices have at least two reconstructed tracks and
99% at least one; a vertex with fewer than `min_tracks_for_valid` tracks is not a target of the model.

## Workflow

```bash
# 1. ROOT ntuples -> parquet (torch-free; one record per event: track, GV, PV_x/y/z, event ids), 80/10/10 split
maskformer-sv convert /eos/user/m/mmarcheg/BTV/sv_reconstruction/ttbar_BTV_1.root \
    --outdir /eos/user/m/mmarcheg/BTV/sv_reconstruction/parquet --split 0.8 0.1 0.1

# 2. normalization parameters of the track features and the vertex targets (params/sv_*_var_transform.yaml)
maskformer-sv norm-par --input /eos/user/m/mmarcheg/BTV/sv_reconstruction/parquet/train

# 3. training
maskformer-sv train --config configs/sv/gv_tracks.yaml
maskformer-sv train -c configs/sv/gv_tracks_small.yaml --debug --max-epochs 1 --trainer.limit_train_batches 5   # smoke test

# 4. evaluation: predictions next to the best checkpoint + the plot chain
maskformer-sv evaluate --config logs/<run>/config.yaml --test-path /eos/user/m/mmarcheg/BTV/sv_reconstruction/parquet/test
maskformer-sv plot --input <ckpt>__test_test.parquet --outdir <dir>        # the plots again, torch-free
```

Any Lightning argument can be appended to `train`/`evaluate` (`--trainer.max_epochs 5`, `--data.num_workers 0`,
`--data.track_require_pass_seed true`, ...); `--attn-type torch` at evaluation on GPUs without flash attention.

## Structure of the configs

```yaml
data:
  inputs:
    track: [pt, eta, sinphi, cosphi, ip2d, ip2dsig, ip3d, ip3dsig, dz, dzsig, charge, nhits, npixhits, nstriphits, chi2, qoverp, highpurity, passseed]
  targets:
    vertex: [pt, eta, sinphi, cosphi, lxy, lz]
  num_objects: 16            # object slots = decoder queries
  max_tracks: 1024           # padding of the tracks
  min_tracks_for_valid: 1    # a vertex is a target with at least this many selected tracks
  track_min_pt: 0.0          # the track selection (all off by default): track_max_abs_eta, track_require_pass_seed,
                             # track_require_high_purity, track_max_per_event (the highest-pT tracks)
  scale_dict_path: params/sv_track_var_transform.yaml           # scales the input features
  vertex_scale_dict_path: params/sv_vertex_var_transform.yaml   # scales the targets; the prediction writer inverts it
model:
  constituent: track         # the wrapper's metrics read pflow_track_valid / particle_track_valid
  num_classes: 3
```

The vertex is the MaskFormer's object, called `particle` in the task names and the tensors (hepattn's convention,
shared with the HGCAL pipeline: `particle_valid`, `particle_class`, `particle_track_valid`, `particle_<target>`);
the track is the input constituent (`track_features`, `track_valid`, `track_pt/eta/phi` for the positional
encoding). The tasks are hepattn's `ObjectClassificationTask`, the repository's `RowwiseObjectHitMaskTask` (mask
losses correct for batches of several events) and `ConfigurableObjectRegressionTask` (a configurable head on the
query embedding, with an L1 matching cost), see `src/maskformer/models/tasks.py`. The regression task takes no scale
dict: it works in the scaled space of the vertex dict and the prediction writer un-scales its outputs
(`sv.callbacks.prediction_writer.SVPredictionWriter`).

The inputs are derived from the NanoAOD branches by `sv.dataset.features` (log pT, asinh-compressed impact
parameters with the -9999 "not available" markers mapped to 0, ...) and scaled by the track dict, whose entries are
pure shifts/scales; the vertex dict may carry a `fn` (`log` for pT, `log1p` for lxy) applied by hepattn's
`VarTransform` so that the writer can invert it. `maskformer-sv norm-par` refits both from a converted file.

## Outputs of the evaluation

`<ckpt>__test[_<suff>].parquet` holds the test events with the prediction branches (one prediction per truth
vertex from the Hungarian-matched query, the track assignment, the fake vertices; the writer's module doc lists
them) and `<ckpt>__test[_<suff>]_plots/`:

- `efficiency/`: perfect (track recall = purity = 1) and loose (both >= 0.5) vertex-finding efficiency vs lxy, pT,
  truth track multiplicity, vertices per event, per category (b, c from b, prompt c, other); recall/purity;
- `fakes/`: predicted vertices without a truth vertex, per event and vs the number of truth vertices;
- `classification/`: confusion matrix and class probabilities;
- `regression/`: residuals, predicted vs true and resolution vs true for every regressed property;
- `resolution/`: for every variable the model predicts (pT, eta, phi, sin/cos phi, lxy, lz, ...) the absolute residual
  pred - true and the relative one (pred - true) / true, with median, sigma68, mean and standard deviation, plus
  `overview.png` with all of them (`src/sv/scripts/plot/plot_sv_resolution.py`, also runnable on its own:
  `--selection loose|perfect|found|targets` picks the vertices, `--range-sigmas` the histogram window);
- `tracks/`: where the heavy-flavour tracks end up (right vertex / other / fake / unassigned), background tracks
  claimed by a vertex;
- `metrics.json`, `summary.txt`.

Vertices enter the efficiencies with at least `EVAL_MIN_TRACKS = 2` selected truth tracks (the paper's evaluation
selection, `src/sv/scripts/plot/plot_sv_eval.py`); single-track vertices are reported separately. The chain is
`src/sv/scripts/plot/run_sv_plot_chain.py`, run by the evaluation as a subprocess and by `maskformer-sv plot`.
