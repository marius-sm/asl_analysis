# ICM multi-PLD ASL analysis pipeline

This page details the different steps of our analysis. We assume that we have
- a T1 image, used for the segmentation of different anatomical structures
- an ASL/Proton Density image pair for each Post Labelling Delay. The PD image is used for computing the Cerebral Blood Flow (CBF) in absolute units (ml/100g/min).

The different steps are:
- Conversion of DICOM data to NIfTI
- Plexus choroid segmentation on T1 image
- FastSurfer segmentation on T1 image
- Rigid co-registration of all ASL/PD images
- Computation of CBF from ASL images
- Calibration of CBF into absolute units (ml/100g/min) using PD images
- Rigid registration of T1 onto ASL/CBF
- Computation of statistics for each label in the segmentation
