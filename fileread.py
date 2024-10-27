import SimpleITK as sitk
import numpy as np
import os

print("Reading Dicom directory:", )
reader = sitk.ImageSeriesReader()

dicom_names = reader.GetGDCMSeriesFileNames("/Users/mayunagupta/Downloads/Non-contrast Cardiac CT Images Dataset with Coronary Artery Calcium Scoring/Dataset/Patient/Patient_18/SE0001")
reader.SetFileNames(dicom_names)

print(dicom_names)

image = reader.Execute()

size = image.GetSize()
print("Image size:", size[0], size[1], size[2])

# print("Writing image:", sys.argv[2])

sitk.WriteImage(image, "test.nii.gz")

# if "SITK_NOSHOW" not in os.environ:
#     sitk.Show(image, "Dicom Series")
