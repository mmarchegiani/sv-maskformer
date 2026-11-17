# Secondary-vertex reconstruction (`sv`): tracks -> heavy-flavour decay vertices

MaskFormer reconstruction of the heavy-flavour decay vertices of an event from its reconstructed tracks, after
[Secondary vertex reconstruction with MaskFormers](https://link.springer.com/article/10.1140/epjc/s10052-024-13374-5)
(Van Stroud et al., EPJC 84 (2024) 1020, [arXiv:2312.12272](https://arxiv.org/abs/2312.12272)), on CMS ttbar
simulation. The code lives in `src/sv/` (reusing `src/maskformer/`, never `src/hgcal/`), the command is
`maskformer-sv` (`python -m sv` without the installed package).

| config | model | purpose |
|---|---|---|
| `gv_tracks.yaml` | dim 256, 4 encoder + 3 decoder layers, 8 heads, 16 queries, mask attention, AdamW/OneCycle, 50 epochs (the paper's model) | training on GPU |
| `gv_tracks_small.yaml` | dim 64, 2 + 2 layers, torch attention, 200 train events, CPU | smoke tests of the pipeline |
| `gv_tracks_dynamic_queries.yaml` | `gv_tracks.yaml` with hepattn's dynamic queries: a per-track `query_init` task (`maskformer.models.tasks.QuerySeedTask`, target `track_seed` = the highest-pT track of each target vertex) seeds the queries from the track embeddings, at most `num_objects`; batch size 1 | training on GPU, fewer empty query slots |
| `gv_tracks_dynamic_queries_null0p3.yaml` | `gv_tracks_dynamic_queries.yaml` with classification `null_weight` 0.3 (was 0.1) and `query_init` `max_pos_weight` 10 (was the default 1000) | fewer fake vertices: a less vertex-biased classifier and fewer seeds |

## The problem, mapped onto the MaskFormer

| | paper (ATLAS-like Delphes, per jet) | here (CMS ttbar ntuples, per event) |
|---|---|---|
| input constituents | the 50 highest-pT tracks of the jet | all reconstructed tracks of the event (`track` collection: mean 390, max 847) |
| track features | d0, z0, pT fraction, eta/phi relative to the jet, jet pT/eta | global kinematics, impact parameters and significances, dz, hit content, quality flags (`sv.dataset.features.TRACK_FEATURES`) |
| objects (queries) | b- and c-hadron decay vertices, 5 queries | the `GV` collection after the candidate selection below: b-/c-hadron (and 3-prong tau) decay vertices; mean 2.7, max 11 per event; 16 queries |
| truth track-vertex link | truth association | `track_hadidx`, the index of the vertex each track comes from (-1 for 98% of the tracks) |
| vertex classes | b, c, null | B, BtoC (c-hadron from a b-hadron), C (c-hadron not from a b-hadron), Other (taus, the rest), null; from `GV_isB`, `GV_isD`, `GV_fromHF` |
| regression | hadron pT, Lxy (from the beam line), dR to the jet axis | hadron pT, eta, sin/cos phi, lxy and lz (flight distance from the generated primary vertex); also available: l3d, x, y, z |
| decoder cross-attention | mask attention (each query attends to the tracks of its predicted mask) | the same; hepattn's local strided window (a fixed window of tracks per query over the phi-sorted tracks) is available, see below |
| losses | 0.5 CE + 2 L1 + 12 mask BCE + 8 mask Dice, Hungarian matching | the same weights; the matching cost is the same combination |
| evaluation | perfect / loose vertex-finding efficiency vs Lxy, fake rate, fit residuals | the same, plus the track assignment and the class confusion (`maskformer-sv plot`) |

The data and the truth definitions are documented in `src/sv/dataset/collections.py`, the feature derivations in
`src/sv/dataset/features.py`.

**Candidate vertices.** `maskformer-sv convert` keeps a `GV` vertex when (`VERTEX_SELECTION`, `DAUGHTER_SELECTION`
in `collections.py`):

- hadron pT > 10 GeV and |eta| < 2.5;
- at least 2 daughters (from the `GVDaughters` collection) that are charged, stable (a generator-stable charged
  species: e, mu, pi, K, p, Sigma, Xi, Omega; the daughters carry no status), with pT > 0.8 GeV and |eta| < 2.5.
  Their number is written as `GV_nSelectedDaughters`.

The vertices failing it are dropped, the remaining ones renumbered and the `track_hadidx` of the tracks of a
dropped vertex reset to -1 (`track_label` 0); the converter prints the cutflow. The ntuple producer already applies
this selection: on `ttbar_BTV_1.root` all 13,628 vertices pass. The dataset refuses a file whose vertices fail it
and reports a file converted before the selection existed (no `GV_nSelectedDaughters`).

**Targets.** Among the candidate vertices, 91% have at least two reconstructed tracks on `ttbar_BTV_1.root` and
99% at least one; a vertex with fewer than `min_tracks_for_valid` tracks is not a target of the model (null class,
its tracks are background).

## Workflow

```bash
# 1. ROOT ntuples -> parquet (torch-free; one record per event: track, GV, PV_x/y/z, event ids), 80/10/10 split
maskformer-sv convert '/eos/cms/store/group/phys_btag/mmarcheg/sv_reconstruction/training_datasets/TTto4Q_tracks_GenVertex_50k/TTto4Q_track_*.root' \
    --outdir /eos/cms/store/group/phys_btag/mmarcheg/sv_reconstruction/training_datasets/TTto4Q_tracks_GenVertex_50k/parquet --split 0.8 0.1 0.1

# 2. normalization parameters of the track features and the vertex targets (params/sv_*_var_transform.yaml)
maskformer-sv norm-par --input /eos/cms/store/group/phys_btag/mmarcheg/sv_reconstruction/training_datasets/TTto4Q_tracks_GenVertex_50k/parquet/train

# 3. training
maskformer-sv train --config configs/sv/gv_tracks.yaml
maskformer-sv train -c configs/sv/gv_tracks_small.yaml --debug --max-epochs 1 --trainer.limit_train_batches 5   # smoke test
condor_submit jobs/train_sv_condor.sub            # unattended on the CERN GPU batch: training + evaluation + plots on
                                                  # EOS (jobs/train_sv_condor.sh; smoke=1 flavour=longlunch to validate)

# 4. evaluation: predictions next to the best checkpoint + the plot chain
maskformer-sv evaluate --config logs/<run>/config.yaml --test-path /eos/cms/store/group/phys_btag/mmarcheg/sv_reconstruction/training_datasets/TTto4Q_tracks_GenVertex_50k/parquet/test
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
  max_tracks: 1024           # events with more tracks are dropped (and the padding, see pad_to_max_tracks)
  sort_tracks_by: null       # order of the tracks of an event (null: file order; phi, eta ascending; pt descending)
  pad_to_max_tracks: true    # pad every event to max_tracks; false keeps each event's own length (batches are
                             # padded to their longest event)
  min_tracks_for_valid: 1    # a vertex is a target with at least this many selected tracks
  track_min_pt: 0.0          # the track selection (all off by default): track_max_abs_eta, track_require_pass_seed,
                             # track_require_high_purity, track_max_per_event (the highest-pT tracks)
  scale_dict_path: params/sv_track_var_transform.yaml           # scales the input features
  vertex_scale_dict_path: params/sv_vertex_var_transform.yaml   # scales the targets; the prediction writer inverts it
  batch_size: 64
model:
  constituent: track         # the wrapper's metrics read pflow_track_valid / particle_track_valid
  num_classes: 4            # B, BtoC, C, Other (null = index 4)
  model:
    init_args:
      decoder:
        mask_attention: true   # the paper's masked cross-attention
```

**Alternative decoder cross-attention.** hepattn's local strided window attention (`MaskFormerDecoder` with
`local_strided_attn: true`, `window_size: 512` (even), `window_wrap: true`) gives query i a fixed window of
`window_size` tracks around position i x n_tracks / num_queries of the track list instead of the predicted mask.
hepattn requires `mask_attention: false`, batch size 1, unpadded events (`data.pad_to_max_tracks: false`) and the
torch or flex attention backend in the decoder layers; the window runs over the dataset's track order, so
`data.sort_tracks_by: phi` (the window wraps around +-pi; `track_local_idx` maps the tracks back to the file for the
prediction writer). The encoder keeps its full flash-varlen self-attention either way (hepattn's encoder window
would exclude the register tokens). With 16 queries and 512 tracks the window covers the whole event for 70% of the
ttbar events, so it is close to unrestricted cross-attention there.

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
  truth track multiplicity, vertices per event, per class (B, BtoC, C, Other); recall/purity;
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
