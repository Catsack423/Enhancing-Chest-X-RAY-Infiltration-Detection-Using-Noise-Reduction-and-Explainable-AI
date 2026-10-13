# V3 release — measured results

Run mode: full; training seed: 42. Smoke scores verify the pipeline only.

Classification uses Infiltration-only vs No Finding-only. XAI includes all bbox cases, wrong predictions and empty maps. PR-AUC uses average precision. Patient-bootstrap intervals and paired deltas use matched patient draws within this seed; between-seed mean/SD is reported separately. No clinical significance is inferred from a delta interval.

Scores from weighted BCE are not assumed to be calibrated population probabilities; each model's threshold maximizes Youden J on its own validation predictions, and paired deltas keep both models' thresholds. SHAP explains processed spatial views; signed RGB/view arrays are retained and localization uses positive evidence only. Frequency fusion is a separate representation experiment.

| condition_id | status | n | best_epoch | epochs_run | threshold | accuracy | balanced_accuracy | precision | recall | specificity | f1 | roc_auc | pr_auc |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| baseline | complete | 12081 | 6 | 9 | 0.4952 | 0.4279 | 0.5857 | 0.2207 | 0.8351 | 0.3362 | 0.3492 | 0.6677 | 0.3099 |
| median_L1 | complete | 12081 | 6 | 9 | 0.4907 | 0.4527 | 0.5869 | 0.2234 | 0.7991 | 0.3747 | 0.3492 | 0.6670 | 0.3138 |
| median_L2 | complete | 12081 | 6 | 9 | 0.4547 | 0.4170 | 0.5769 | 0.2165 | 0.8297 | 0.3241 | 0.3434 | 0.6561 | 0.3000 |
| median_L3 | complete | 12081 | 6 | 9 | 0.4167 | 0.3674 | 0.5657 | 0.2093 | 0.8793 | 0.2521 | 0.3381 | 0.6616 | 0.3042 |
| clahe_dwt_strength_L1 | complete | 12081 | 6 | 9 | 0.5128 | 0.4510 | 0.5890 | 0.2241 | 0.8072 | 0.3708 | 0.3508 | 0.6679 | 0.3181 |
| clahe_dwt_strength_L2 | complete | 12081 | 9 | 12 | 0.4830 | 0.4261 | 0.5792 | 0.2181 | 0.8212 | 0.3372 | 0.3446 | 0.6529 | 0.3010 |
| clahe_dwt_strength_L3 | complete | 12081 | 9 | 12 | 0.5246 | 0.4664 | 0.5842 | 0.2236 | 0.7703 | 0.3980 | 0.3466 | 0.6470 | 0.2963 |
| dae_clahe_strength_L1 | complete | 12081 | 6 | 9 | 0.5072 | 0.4271 | 0.5826 | 0.2195 | 0.8284 | 0.3368 | 0.3470 | 0.6685 | 0.3141 |
| dae_clahe_strength_L2 | complete | 12081 | 6 | 9 | 0.4718 | 0.3909 | 0.5738 | 0.2136 | 0.8631 | 0.2846 | 0.3424 | 0.6531 | 0.3021 |
| dae_clahe_strength_L3 | complete | 12081 | 6 | 9 | 0.4753 | 0.4209 | 0.5629 | 0.2113 | 0.7874 | 0.3384 | 0.3332 | 0.6177 | 0.2724 |
| dwt_L1_L4_fusion | complete | 12081 | 13 | 16 | 0.5106 | 0.4089 | 0.5742 | 0.2149 | 0.8356 | 0.3128 | 0.3419 | 0.6590 | 0.3062 |
| dft_low_mid_high_fusion | complete | 12081 | 6 | 9 | 0.4746 | 0.4221 | 0.5811 | 0.2185 | 0.8324 | 0.3297 | 0.3461 | 0.6681 | 0.3140 |
