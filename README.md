# Massachusetts-Building-Detection-From-Aerial-Imagery

> [!NOTE]
> **GitHub may show "Invalid Notebook" when previewing the `.ipynb` files.** This is a rendering issue, not a problem with the file. To view the notebooks, click the **Download raw file** button (or download the repo) and open them locally in Jupyter / VS Code, or paste the GitHub URL into [nbviewer.org](https://nbviewer.org/).

A convolutional neural network that classifies whether a piece of land is built on, trained on Massachusetts NAIP aerial imagery and MassGIS building footprints. Final project for **BA865 — Neural Networks for Business Applications** (Boston University, Spring 2026).

We built and benchmarked two formulations of the same underlying task:

1. **Parcel-scoped** — given a NAIP crop masked to a single tax parcel, is the parcel built? *(Team approach.)*
2. **Tile-scoped** — given an arbitrary 256m × 256m NAIP tile, does it contain any building? *(Individual extension; closer to how production systems like Microsoft Building Footprints actually operate.)*

The parcel formulation has cleaner labels and better per-input accuracy. The tile formulation gives up some accuracy to gain operational generality (no parcel database required, fixed grid for change detection, composes into downstream segmentation).

---

## Headline Results

| Formulation | Architecture | Test accuracy | ROC-AUC | Test set |
|---|---|---|---|---|
| Parcel-scoped | Custom 4-block CNN | **94.8%** | **0.99** | 599 parcels |
| Parcel-scoped | Frozen MobileNetV2 | 92.2% | 0.98 | 599 parcels |
| Tile-scoped | Custom 4-block CNN | ~88% | ~0.94 | 2,693 tiles |
| Tile-scoped | Frozen MobileNetV2 | ~90% | ~0.95 | 2,693 tiles |

The ~5-point accuracy gap between the two formulations is decomposed in the writeup into label-noise vs. structural sources (image–question alignment, fixed-grid zoom).

## Quick Start

Tested on Google Colab (CPU and GPU runtimes). Recommended path: open the notebook directly in Colab.

```bash
pip install -r requirements.txt
jupyter lab notebooks/tile_pipeline.ipynb
```

Run cells top-to-bottom. The pipeline:

1. Pulls NAIP imagery from the Microsoft Planetary Computer STAC API for three Massachusetts study areas (Newton/Brookline, Framingham MetroWest, Worcester Edge).
2. Downloads MassGIS building footprints for the same bboxes.
3. Tiles each NAIP raster on a fixed 256m grid.
4. Labels each tile via an argmax + 25%-secondary-floor rule against the footprints, with deep-interior-zero and parcel-style cleanup filters.
5. Trains a baseline CNN and a frozen MobileNetV2 transfer-learning model.
6. Evaluates on a stratified 70/15/15 split with class weighting.

End-to-end runtime: ~30 min on a single T4 GPU; ~2 hr on CPU.

---

## Data Sources

| Source | Use | Access |
|---|---|---|
| MassGIS Property Tax Parcels | Parcel polygons (parcel-scoped pipeline) | ArcGIS FeatureServer, public |
| MassGIS Building Structures 2-D | Building footprints, used to derive labels | ArcGIS FeatureServer, public domain |
| NAIP Aerial Imagery | RGB inputs at 0.6–1.0m resolution | Microsoft Planetary Computer STAC API |

All three are free and require no authentication. NAIP imagery used: 2018–2023 acquisition windows.

---

## Methodology Notes

**Coordinate Reference Systems.** All label math is done in EPSG:26986 (Mass State Plane, meters). NAIP tiles are kept in their native UTM CRS for raster reads. CRS validation happens before every spatial join.

**Tile labeling.** Each building footprint is assigned to the tile it has the largest intersection area with (argmax) plus any tile holding ≥25% of its total area (secondary floor). This avoids two failure modes of a single coverage threshold: slivers in neighboring tiles (suppressed by argmax) and buildings genuinely shared across two tiles (caught by the secondary floor).

**Cleanup filter.** Mirrors the parcel pipeline's `BUILT_LABEL_OVERLAP_M2 / CLEAN_NOT_BUILT_BUFFER_M` logic. Drops positives with <50 m² total building area (sliver artifacts) and negatives with a footprint within 5 m of the tile boundary (likely registration noise).

**Architectures.** Both formulations use the same two architectures for direct comparison:

- **Custom baseline CNN.** Four Conv2D blocks (32 → 64 → 128 → 256), GlobalAveragePooling, Dense(128), Dropout(0.30), sigmoid. Augmentation: RandomFlip (horizontal+vertical for aerial), RandomRotation(0.05).
- **MobileNetV2 transfer learning.** Frozen ImageNet backbone, custom head (GAP → Dropout → Dense(64) → sigmoid). Trainable parameters: ~82K out of ~2.3M.

Both trained with Adam(1e-3), binary cross-entropy, batch size 32, EarlyStopping on val_auc with patience 5.

---

## Error Analysis

Three dominant failure modes identified by manual inspection of high-confidence model–record disagreements:

1. **Neighboring-context bleed** — model over-predicts "built" when surrounding tiles/parcels are built up.
2. **Roof-pattern leakage** — paved or vegetated parcels picking up shared roof patterns from neighbors despite the parcel mask.
3. **Tree-canopy occlusion** — small or partially-occluded buildings missed by the model from above.

Critically, on the parcel pipeline, **all** inspected high-confidence disagreements were *model* errors, not *label* errors. This invalidates the originally-pitched use case of using disagreements as a label-error screening queue without further validation.

---

## Limitations

- **Proxy labels.** All labels derive from MassGIS building footprint overlap; not manually verified ground truth.
- **Geographic narrowness.** Three Massachusetts study areas. Generalization to other states or rural geographies is unproven.
- **Imagery distribution.** Trained on summer NAIP at ~1 m/px. Performance on winter, Sentinel-2, or sub-meter commercial imagery is unknown.
- **Capture-date mismatch.** NAIP and MassGIS footprints are not co-acquired; some "errors" may reflect real changes between capture dates.
- **Screener use case unvalidated.** The proposed label-error screening queue requires a manually verified evaluation set before it can be deployed.

---

## Future Work

A phased 12-month roadmap is detailed in the presentation. Highlights:

- **Months 0–3** — Collect a manually verified evaluation set; run by-municipality holdout to measure geographic generalization.
- **Months 3–6** — Fine-tune MobileNetV2's upper blocks; add NAIP near-infrared (NIR) as a 4th input channel; replace mean-color fill with a small context ring around the parcel.
- **Months 6–12** — Train an inconsistency-flagging model on human-reviewed queue outputs; embed inside the assessor's existing GIS workflow.

---

## Acknowledgments

- Reference paper: *Sentinel-2 building/road detection at scale* (arXiv:2310.11622) — informed labeling decisions and discussion of registration noise.
- MassGIS for open parcel and building footprint data.
- Microsoft Planetary Computer for NAIP STAC access.
- BA865 instruction team and classmates.

---

## License

Code: MIT.
Data: subject to the licenses of the upstream sources (MassGIS public domain; NAIP via Microsoft Planetary Computer terms).
