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

## Requirements
### FastSurfer
FastSurfer must be cloned from GitHub.
```
git clone https://github.com/Deep-MI/FastSurfer
```
I recommend to create a conda environment for FastSurfer in order to install the necessary packages (https://github.com/Deep-MI/FastSurfer/blob/master/requirements.txt). FreeSurfer does not need to be installed.
A GPU enables fast segmentation (under 1 minute).

### FSL
FSL can be loaded on the cluster
```
module load FSL/6.0.3
```

## Conversion of DICOM to NIfTI
Use dcm2niix
```bash
module load dcm2niix
mkdir nifti_data
dcm2niix -o nifti_data -f '%p_serie_%s' dicom_data_directory
```
where `dicom_data_directory` is the directory containing all the DICOM data of our patient.
The NIfTI files should appear in the directory `nifti_data`.
For example, we may obtain
```
nifti_data
├── 3D_Ax_ASL_1025.nii
├── 3D_Ax_ASL_1025.json
├── 3D_Ax_ASL_1025a.nii
├── 3D_Ax_ASL_1025a.json
├── 3D_Ax_ASL_1525.nii
├── 3D_Ax_ASL_1525.json
├── 3D_Ax_ASL_1525a.nii
├── 3D_Ax_ASL_1525a.json
...
├── 3D_Ax_T1_MPRAGE_F.nii
└── 3D_Ax_T1_MPRAGE_F.json
```
Each `.nii` file is accompanied by a `.json` file containing the meta-data (acquision parameters, etc).

## Plexus choroid segmentation on T1

```
python plexus_segmentation.py -i nifti_data/3D_Ax_T1_MPRAGE_F.nii
```

This will add the file `plexus_mask_3D_Ax_T1_MPRAGE_F.nii` into the directory `nifti_data`.
This mask contains continuous values between 0 and 1 and is in the native space of the T1 image.

## FastSurfer segmentation on T1

Create a directory for the outputs:
```
mkdir nifti_data/fastsurfer_outputs
```

In order to run FastSurfer, we need to go to the directory into which we have cloned FastSurfer
```
cd fastsurfer_directory
```

Now we can run FastSurfer
```
./run_fastsurfer.sh --t1 .../nifti_data/3D_Ax_T1_MPRAGE_F.nii --sd .../nifti_data --sid .../nifti_data/fastsurfer_outputs --seg_only
```

This should create a file
```
nifti_data
└── fastsurfer_outputs
    └── mri
        └── aparc.DKTatlas+aseg.deep.mgz
```

The file `aparc.DKTatlas+aseg.deep.mgz` is in a FastSurfer conformed space. We will now move it back to the native space of the T1 image
```
mri_vol2vol \
--mov nifti_data/fastsurfer_outputs/mri/aparc.DKTatlas+aseg.deep.mgz \
--targ nifti_data/3D_Ax_T1_MPRAGE_F.nii \
--o nifti_data/fastsurfer_outputs/mri/aparc.DKTatlas+aseg.deep_native_space.nii.gz \ 
--regheader --interp nearest
```
