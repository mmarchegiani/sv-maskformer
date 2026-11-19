# Normalization parameters (scale dicts)

The files here hold the normalization parameters of the model inputs and targets: for every variable the
transform applied first (`fn`: none, `sqrt`, `log` or `log1p`), the kind of scaling (`type: std` shifts by
`mean` and divides by `std`; `type: min_max_sym` maps [`min`, `max`] onto [-1, 1]) and the fitted statistics.
hepattn's `FeatureScaler` reads them, `maskformer norm-par` writes them.

Every training config names **two** of them (see [configs/README.md](../configs/README.md)):

| key in `data:` | fitted on | used by |
|---|---|---|
| `scale_dict_path` | the topocluster collection (`hgcal_rechit_var_transform.yaml`, `hgcal_layercluster_var_transform.yaml` or `hgcal_trackster_clue3dhigh_var_transform.yaml`) | the node input features in `HGCDataset` |
| `particle_scale_dict_path` | the particle collection (`hgcal_mergedsimcluster_var_transform.yaml`, `hgcal_simcluster_var_transform.yaml` or `hgcal_simtrackster_var_transform.yaml`) | the regression targets in `HGCDataset`, the regression task (`scale_dict_path: *particle_scale_dict_path`) and the prediction writer |

Both dicts use the keys `e`/`eta`/`r`, but for populations whose spreads differ by orders of magnitude (a rechit
carries ~0.01 GeV, a SimCluster ~10 GeV), so they must stay separate files. The rationale and the history of the
fix that introduced the split are in [docs/NORMALIZATION_FIX.md](../docs/NORMALIZATION_FIX.md).

## Files

| file | collection | fitted on | notes |
|---|---|---|---|
| `hgcal_rechit_var_transform.yaml` | RecHitHGC | `100tau_train.parquet`, 344M hits | `e: log`; `t`/`t_filt` fitted on valid times |
| `hgcal_layercluster_var_transform.yaml` | LayerCluster | `100tau_train.parquet`, 3.2M layer clusters | |
| `hgcal_simcluster_var_transform.yaml` | SimCluster | `100tau_train.parquet`, 108k SimClusters | `e: log1p` |
| `hgcal_trackster_clue3dhigh_var_transform.yaml` | ticlTrackstersCLUE3DHigh | `taus_fineCalo_split.parquet` (TICLv5, CMSSW_20_0_0_pre1), 62,533 tracksters | fitted 2026-09-18 |
| `hgcal_simtrackster_var_transform.yaml` | ticlSimTracksters | same file, 579,188 SimTracksters | `e` fitted on `regressed_energy`; boundary targets added 2026-10-16; fitted with `--allow-outliers` (3 SimTracksters with the barycenter in the opposite endcap, kept out of the fit by the quantile bounds) |
| `toy_rechit_var_transform.yaml` | toy-calorimeter RecHitHGC | toy `pufree` sample (5 files, 105M rechits) | fitted 2026-10-21; node dict of `configs/toycalo/rechit_pufree_a3.yaml` |
| `toy_simcluster_var_transform_sec.yaml` | toy-calorimeter SimCluster, per-sector targets | toy `pufree_sec` sample (5 files, 385k SimClusters) | fitted 2026-10-21; particle dict of the sectored-decoder config |
| `sv_track_var_transform.yaml` | `track` (SV ntuples), the derived input features of `sv.dataset.features` | train split of `ttbar_BTV_1.root` (4000 events, 1.56M tracks) | fitted 2026-11-06 with `maskformer-sv norm-par`; entries are pure shifts/scales (the derivations hold the log/asinh) |
| `sv_vertex_var_transform.yaml` | `GV` (SV ntuples), the vertex regression targets | same split, 10,938 vertices | fitted 2026-11-06; `pt: log`, `lxy`/`l3d: log1p`; the SV prediction writer inverts it |

The header of each file records the date, the input file and the bounds settings it was fitted with. `type: std`
is used for energies, `min_max_sym` with detector-envelope extremes for the geometry; the particle `eta`/`r` use
quantile bounds because SimCluster impact points reach |eta| ~ 5.9 while the bulk sits within 3.1.

### Referenced by configs but not fitted yet

| file | collection | needed by | how to produce it |
|---|---|---|---|
| `hgcal_mergedsimcluster_var_transform.yaml` | MergedSimCluster | `configs/rechit/mergedsimcluster.yaml`, `configs/test/rechit_mergedsimcluster.yaml` | `sbatch jobs/norm_par_job.sh` on falcon (its defaults are RecHitHGC / MergedSimCluster on the tau sample); the local sample the other dicts were fitted on has no `MergedSimCluster` collection |

Until it exists, a config that needs it fails at dataset construction with a `FileNotFoundError` naming the
file. `tests/test_configs.py` keeps this list in sync with the configs.

The SV MaskFormer (`configs/sv/`, `src/sv/`) names its two dicts `scale_dict_path` (track features) and
`vertex_scale_dict_path` (vertex targets); they are fitted with `maskformer-sv norm-par --input <converted parquet>`
(`src/sv/scripts/dataset/compute_norm_par.py`, torch-free) and keyed by the feature names of
`src/sv/dataset/features.py` (`pt`, not `e`).

## Fitting: `maskformer norm-par`

The full fits run on falcon through Slurm (the gateway kills processes above 60% CPU):

```bash
sbatch jobs/norm_par_job.sh                                            # RecHitHGC -> MergedSimCluster (configs/rechit/mergedsimcluster.yaml)
TOPO=RecHitHGC    PART=SimCluster sbatch jobs/norm_par_job.sh          # configs/rechit/simcluster.yaml
TOPO=LayerCluster PART=SimCluster sbatch jobs/norm_par_job.sh          # configs/layercluster/simcluster.yaml
TOPO=ticlTrackstersCLUE3DHigh PART=ticlSimTracksters DATA_DIR=<dir> sbatch jobs/norm_par_job.sh   # configs/trackster_clue3dhigh/*.yaml
```

The job runs `maskformer norm-par` with the node dict written to `params/<topocluster dict>` and the particle
dict to `params/<particle dict>` (the names follow `DEFAULT_OUTPUT` in `src/hgcal/scripts/dataset/compute_norm_par.py`).
Each variable is first transformed (`sqrt`/`log`/`log1p`, set per variable in `TOPO_VAR_SPECS` /
`PARTICLE_VAR_SPECS` inside that script), and only then are its `mean`/`std`/`min`/`max` measured and written to
the YAML. That order matters, because hepattn's `VarTransform` applies the transform *before* shifting and scaling.
The job also writes before/after histograms to `plots/norm_par/` (`<Collection>_<var>_{before,after}.png`) and
fails if a variable's extreme lies further than its allowed number of standard deviations from the mean
(`NSIGMA_FAIL` in the script), so a broken variable is caught without tripping on physical tails. The boundary
eta/phi of the particles are derived from `boundary_{x,y,z}` (the dumped `boundaryPhi` branch is broken).

A quick check on a small subset:

```bash
maskformer norm-par --max-files 1 --max-events 20 --dry-run
maskformer norm-par --topocluster-collection LayerCluster --particle-collection SimCluster --input-dir <dir> --max-files 1 --max-events 20 --dry-run
```

Options: `--input-dir` (a directory of parquet files or one file), `--topocluster-collection`,
`--particle-collection`, `--output`, `--particle-output`, `--max-files`, `--max-events`, `--threads` (keep 1 on
the gateway), `--plot-dir`, `--no-plots`, `--allow-outliers`, `--dry-run`; the script's other options
(`--batch-events`, `--sample-frac`, `--sample-cap`, `--clip-quantile`, `--bounds`, `--warn-nsigma`, `--seed`,
`--no-particles`, `--no-alternatives`) are forwarded as given.

## Checking a config: `maskformer verify-norm`

```bash
maskformer verify-norm --config configs/layercluster/simcluster.yaml --parquet <file.parquet> --num-events 200
```

builds the dataset of the config on the file exactly as training would and reports the scaled range of every
node input feature and regression target, then checks the failure modes the normalization fix addressed: real
dynamic range of the node energy, a scaled `r` input, targets scaled with the particle dict, the regression task
wired to the particle dict, and finiteness. Exit code 0 when every check passes.

## How the paths are resolved

Configs reference the files as `params/<name>.yaml`, relative to the repository root. `maskformer.utils.paths.resolve_scale_dict_path`
(torch-free) finds them from any directory: as given, under the repository root (`$HGCAL_REPO_ROOT`, the checkout
holding `src/`, the current directory, or `/app` inside the container image), and maps the `configs/<name>.yaml`
of runs trained before November 2026 onto `params/<name>.yaml` with one `[paths] ... using params/...` line. The
Lightning CLI applies it to the data module's dicts and to the regression task's dict before anything is
instantiated, so the config saved in the run folder carries the resolved `params/` paths; `maskformer plot
--scale-dict` goes through the same function.
