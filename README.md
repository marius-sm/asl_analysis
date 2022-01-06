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

## Conversion of DICOM to NIfTI
Use dcm2niix
```bash
module load dcm2niix
mkdir nifti_data
dcm2niix -o nifti_data -f '%p_serie_%s' dicom_data_directory
```
where `dicom_data_directory` is the directory containing all the DICOM data of our patient.
The NIfTI files should appear in the folder `nifti_data`.
For example, we may obtain
````
nifti_data
├── 3D_Ax_ASL_1025_serie_5.nii
├── 3D_Ax_ASL_1025_serie_5.json
├── 3D_Ax_ASL_1025_serie_5a.nii
├── 3D_Ax_ASL_1025_serie_5a.json
├── 3D_Ax_ASL_1025_serie_6.nii
├── 3D_Ax_ASL_1025_serie_6.json
├── 3D_Ax_ASL_1025_serie_6a.nii
├── 3D_Ax_ASL_1025_serie_6a.json
...
├── 3D_Ax_T1_MPRAGE_F_serie_2.nii
└── 3D_Ax_T1_MPRAGE_F_serie_2.json
````
