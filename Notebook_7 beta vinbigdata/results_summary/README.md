# v7 beta pilot (VinBigData 512) — seed 42

Dataset: `awsaf49/vinbigdata-512-image-dataset`.
Classes: `normal` (0, No finding only) vs `infiltration` (1, any Infiltration annotation, including co-occurring findings).
12 CNNs, one shared DAE; 320 distinct train images (160+160), 80 validation (40+40), 100 test (50+50).
Test is evaluated independently for each of 16 fixed corruption states. Bootstrap 95% CIs use 2,000 draws in full mode.
