# Notebook versions in Git

Each version branch tracks one notebook directory, shared research documents,
scripts and existing reports. Folder names are preserved so notebook paths keep working.

| Branch | Notebook directory |
|---|---|
| `codex/notebook-v1` | `notebooks v1` |
| `codex/notebook-v2-beta` | `Notebook_2 beta` |
| `codex/notebook-v2-release` | `Notebook_2 release` |
| `codex/notebook-v3-beta` | `Notebook_3 beta` |
| `codex/notebook-v3-beta-e7` | `Notebook_3 beta e7` |
| `codex/notebook-v3-beta-class200` | `Notebook_3 beta class 200` |
| `codex/notebook-v3-release` | `Notebook_3 release` |
| `codex/notebook-v4-release` | `Notebook_3 release` (protocol v4; same folder, Drive paths and source fingerprints) |
| `codex/notebook-v4.1-release` | `Notebook_3 release` (v4 protocol unchanged; sessions can share one seed's conditions) |

v4 keeps the `Notebook_3 release` folder because Colab paths (`Notebook_3_release`) and
the running `runs/release_v4` provenance depend on unchanged bytes. v4 changes the CNN
protocol (early stopping, train-only geometric augmentation, validation Youden threshold)
and writes to `runs/release_v4`. Tag `v3-release-run1` marks the exact v3 code of the
cancelled `runs/release_full_dae_b064_gpu` run; its file hashes match that run's provenance.
v4.1 changes only `tools/`, notebooks and paste-in cells, so root/shared source hashes and
`config_hash` equal v4 and runs in `runs/release_v4` resume across the two versions.

Save or stash local code edits before changing versions:

```powershell
git switch codex/notebook-v2-release
git switch codex/notebook-v3-beta-class200
```

The original `main` and `gradcam` branches are unchanged. The local backup
`codex/archive-before-version-split-20261012` at `c926cc4bca9fff6bcfa406609154a04caadabcb1` includes all six versions and the previously uncommitted
class200 edits, including `check_overfit.py` and `check_overfit.ipynb`.
New version branches retain this history. No existing history was rewritten.
The new branches have been created locally; they have not been pushed.

## Large files

Generated ZIP packages, data ZIPs, model weights and runtime outputs are ignored.
Existing committed ZIPs and weights were copied with SHA-256 verification to:

```text
.local-artifacts/version-split-20261012/<original relative path>
```

A local `manifest.json` records artifact sizes and hashes. The active class200 ZIPs
are also restored to their original paths for local/Colab use. This backup directory
is local only. To use another version's package, copy its files from this backup
to the original directory after switching, or retrieve them from the archive branch.
For new clones, provide dataset ZIPs/weights separately and use each version's
existing builder (`build_notebook.py` or `build_notebooks.py`) to create packages.
Builders may require local data, manifests or previous experiment results.

Branch separation reduces tracked checkout size, but a full clone still downloads
the old history and its binaries. For a small source-only clone of one version:

```powershell
git clone --single-branch --depth 1 --branch codex/notebook-v3-beta-class200 https://github.com/Catsack423/Enhancing-Chest-X-RAY-Infiltration-Detection-Using-Noise-Reduction-and-Explainable-AI.git
```

This clone command works after the new branch has been pushed to origin.
Publishing the six new branches without changing existing remote branches:

```powershell
git push origin codex/notebook-v1 codex/notebook-v2-beta codex/notebook-v2-release codex/notebook-v3-beta codex/notebook-v3-beta-e7 codex/notebook-v3-beta-class200
```

Packing local Git objects reduces loose-object overhead. Purging historical binaries
would require a separate history rewrite and coordination with other clones.
