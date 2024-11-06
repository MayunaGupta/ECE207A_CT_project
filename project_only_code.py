# !pip install matplotlib
import SimpleITK as sitk
import numpy as np
import os
import matplotlib.pyplot as plt
import math
import matplotlib.animation as animation
import nibabel
import nibabel.processing
import glob
# image
# Image = sitk.ReadImage("test.nii.gz")
# sitk.Show(Image)
# space = Image.GetSpacing()
# voxel_volume = space[0]*space[1]*space[2]

''' Reading folder structure to automate and create outputs'''

input_image_path = r'C:\Users\Chandan\Downloads\CAC_cases_2500_vs_625_group1_originals\CAC_cases_2500_vs_625_group1_originals\ID27649_CAC\img-nii\Cal Score bone .625 - 304.nii.gz'
input_mask_path = r'C:\Users\Chandan\Downloads\CAC_cases_2500_vs_625_group1_originals\CAC_cases_2500_vs_625_group1_originals\ID27649_CAC\img-nii\Segmentation-Cal_Score_Bone-label.nii.gz'
img1 = sitk.ReadImage(input_image_path)
mask1 = sitk.ReadImage(input_mask_path)
intr_blur_img_path = r'C:\Users\Chandan\Downloads\intr_blur.nii.gz'
intr_dw_img_path = r'C:\Users\Chandan\Downloads\intr_down_blur.nii.gz'
intr_dw_mask_path = r'C:\Users\Chandan\Downloads\intr_dw_mask.nii.gz'

intr_dw_mask_path_1 = r'C:\Users\Chandan\Downloads\intr_dw_inpmask.nii.gz'

space = img1.GetSpacing()
voxel_volume = space[0]*space[1]*space[2]
pixelID = img1.GetPixelID()

gaussian = sitk.SmoothingRecursiveGaussianImageFilter()
gaussian.SetSigma(0.8)
blur_image = gaussian.Execute(img1)

caster = sitk.CastImageFilter()
caster.SetOutputPixelType(pixelID)
blur_image = caster.Execute(blur_image)
sitk.WriteImage(blur_image, intr_blur_img_path)

#Downsampling
downsample = 1
downsample_dim = [2,2,2]

if (downsample):
    input_img = nibabel.load(intr_blur_img_path)
    input_img_mask = nibabel.load(input_mask_path)
    resampled_img = nibabel.processing.resample_to_output(input_img, downsample_dim)
    resampled_mask = nibabel.processing.resample_to_output(input_img_mask, downsample_dim)
    nibabel.save(resampled_img, intr_dw_img_path)
    nibabel.save(resampled_mask, intr_dw_mask_path)
    nibabel.save(input_img_mask, intr_dw_mask_path_1)
    blur_image = sitk.ReadImage(intr_dw_img_path)
    mask1 = sitk.ReadImage(intr_dw_mask_path)
    
    print(blur_image.GetSpacing())
    print(mask1.GetSpacing())


# sitk.GetImageFromArray(sitk.GetArrayViewFromImage(img1)[0,:,:,:])
image_array = sitk.GetArrayViewFromImage(blur_image)
mask_array = sitk.GetArrayFromImage(mask1)

cropped_array = image_array * mask_array
# Display the image using Matplotlib
plt.imshow(cropped_array[80,:,:], cmap="gray")
plt.show()  

# signal = np.mean(cropped_array)
signal = np.mean(cropped_array[cropped_array>0])

noise = np.std(cropped_array[cropped_array>0])
snr = signal / noise
print(signal, noise, snr)


# # Display the image using Matplotlib
# plt.imshow(mask_array[40,130:450,20:450], cmap="gray")
# plt.show()
# def dist(l1, l2,):
#     return [np.sqrt((l1[0]-l2[0])**2) , np.sqrt((l1[1]-l2[1])**2) , np.sqrt((l1[2]-l2[2])**2 )]

# mask = np.zeros(image_array.shape)
# mask.shape
# # heart_array = image_array[:,130:450,20:450]
# # circular kernel 
# centre = [mask.shape[0]//2,mask.shape[1]//2+30, mask.shape[2]//2]
# radius = [50, 150, 200]
# for i in range(mask.shape[0]):
#     for j in range(mask.shape[1]):
#         for k in range(mask.shape[2]):

#             a,b,c = dist([i,j,k], centre) 
#             if a< radius[0] and b<radius[1] and c<radius[2]:
#                 mask[i,j,k]=1

# heart_array = image_array * mask


# # calcim = heart_array > 130

# # Display the image using Matplotlib
# plt.imshow(heart_array[54,:,:], cmap="gray")
# plt.show()  



def make_movie(heart_array, filename):
    # Create a 3D numpy array

    # Create a figure and axes
    fig, ax = plt.subplots()

    # Initialize an empty list to store the frames
    ims = []

    # Iterate over the slices
    for i in range(heart_array.shape[0]):
        im = ax.imshow(heart_array[i, :,:], animated=True, cmap="gray")
        ims.append([im])

    # Create the animation
    ani = animation.ArtistAnimation(fig, ims, interval=50, blit=True, repeat_delay=1000)

    # Save the animation as a video file (e.g., MP4)
    ani.save(f'{filename}.mp4')

    # Display the animation
    plt.show()

#make_movie(cropped_array, "heart_original_ID27649_temporal")
calcium_mask =cropped_array[cropped_array > 130]
# plt.imshow(calcium_mask[55], cmap='gray')
# plt.show()
# Image.GetSpacing()[1]
def calculate_agatston_score_aorta(ct_scan, voxel_volume, calcium_mask_location="calcium_mask.nii.gz"):
    """
    Calculates the Agatston score of a cropped CT scan specifically for the aorta region.

    Args:
        ct_scan: A 3D numpy array representing the CT scan intensity values of the aorta region.
        voxel_volume: The volume in centimetres cubed (cm^3) of one voxel.

    Returns:
        A tuple containing:
            - The Agatston score: An integer representing the total Agatston score.
            - The Calcium Volume: A float representing the total volume (cm^3) of calcium detected.
    """

    # Thresholds for calcium detection at different attenuation levels
    # global calcium_threshold  # This variable should likely be defined elsewhere
    calcium_threshold = 130 ; 

    thresh = [130, 200, 300, 400]  # Thresholds for classifying calcium based on attenuation

    # Identify calcium pixels using SimpleITK for potential future conversion
    # This line is currently commented out as the code uses numpy for efficiency
    # filter = sitk.BinaryThresholdImageFilter()
    # calcium_mask = ct_scan
    # binaryFilter = sitk.BinaryThresholdImageFilter()
    # binaryFilter.SetInsideValue(1)
    # binaryFilter.SetLowerThreshold(calcium_threshold)
    # binaryFilter.SetOutsideValue(0)
    # ct_image_fitered = binaryFilter.Execute(calcium_mask)
    #
    # # Save the calcium mask for visualization purposes (optional)
    # sitk.WriteImage(ct_image_fitered, calcium_mask_location)

    # Efficiently identify calcium pixels using numpy
    calcium_mask = np.where(ct_scan > calcium_threshold, 1, 0)
    print(np.sum(calcium_mask))
    # Convert the 3D CT scan to a flat array for easier processing
    ct_scan_2 = ct_scan.flat  # Flatten the 3D array

    # Calculate area of each calcium lesion and assign weights based on attenuation
    # Assign weights (labels) based on the intensity of the calcium pixels
    labeled_calcium = np.where(ct_scan_2 >= thresh[3], 4,  # Highest attenuation gets weight 4
                       np.where(ct_scan_2 >= thresh[2], 3,  # Second highest gets weight 3
                            np.where(ct_scan_2 >= thresh[1], 2,  # Middle gets weight 2
                                 np.where(ct_scan_2 >= thresh[0], 1, 0))))  # Lowest gets weight 1
    
    calcium_image = ct_scan * calcium_mask
    # Count the number of pixels in each calcium class and multiply by the weight (area)
    # This efficiently calculates the total area and volume of calcium at each level
    calcium_areas = np.bincount(labeled_calcium, weights=labeled_calcium)
    calcium_volumes = np.bincount(labeled_calcium.flat)[1:] * voxel_volume  # Extract volumes (excluding background)

    # Print calcium areas for debugging purposes (optional)
    # print("Areas", calcium_areas)

    # Calculate the Agatston score and calcium volume

    
    agaston_score = np.sum(calcium_areas) * voxel_volume  # Total area times voxel volume
    calcium_volume = np.sum(calcium_volumes)  # Total volume of calcium
    # sitk.WriteImage(sitk.GetImageFromArray(calcium_mask),calcium_mask_location )
    return agaston_score, calcium_volume, calcium_image
ag_score , volume, calcium_image = calculate_agatston_score_aorta(cropped_array, voxel_volume, calcium_mask_location='calcium_mask_image_2')

print(ag_score , volume)

from PIL import Image

def display_sitk_image_as_movie(image_array, name_img, duration=100):
    """
    Displays a 3D SimpleITK image as a movie using Pillow by iterating through each slice.

    Parameters:
        sitk_image (sitk.Image): The 3D SimpleITK image to display.
        duration (int): Duration to show each frame in milliseconds (default: 100ms).
    """
    # Convert SITK image to a NumPy array
    #image_array = sitk.GetArrayFromImage(sitk_image)  # Shape: (depth, height, width)

    # Create list to store frames
    frames = []

    # Convert each slice to a PIL Image and add to frames
    for i in range(image_array.shape[0]):
        slice_array = image_array[i, :, :]  # Extract each 2D slice
        normalized_slice = np.uint8(255 * (slice_array - np.min(slice_array)) / (np.ptp(slice_array)))  # Normalize
        frame = Image.fromarray(normalized_slice)
        frames.append(frame)

    # Display as animated movie
    frames[0].show()  # Show the first frame to initiate
    frames[0].save(name_img + "_movie.gif", save_all=True, append_images=frames[1:], duration=duration, loop=0)

    print("Movie saved as '3D_image_movie_0.2Gaussian.gif'")

# Usage example

display_sitk_image_as_movie(calcium_image, 'calcium_image', duration=50)
blur_image_1 = sitk.GetArrayFromImage(blur_image)
display_sitk_image_as_movie(blur_image_1, 'ct_image', duration=50)
#make_movie(calcium_image, "calcium_mask_image_ID27649")

# Plotting a basic histogram
# plt.hist(calcium_image.flat, color='skyblue', edgecolor='black')
 
# Get the array from the image
calcium_image_array_mask = np.where(cropped_array > 130, 1,0) #sitk.GetArrayFromImage(calcium_image)
calcium_image_array = calcium_image_array_mask * cropped_array
# Flatten the 3D array to 1D for histogram
flattened_array = calcium_image_array.flatten()


flattened_array.mean()
flattened_array.max()

#Counting lesions based on Agatston bins
lesion_count = {}

thresh = [130, 99999999]
for i in range(0, len(thresh)-1):
    crp_image = sitk.GetImageFromArray(cropped_array)
    thresholded_image = sitk.BinaryThreshold(crp_image, lowerThreshold=thresh[i], upperThreshold=thresh[i+1], insideValue=1, outsideValue=0)
    connected_components = sitk.ConnectedComponent(thresholded_image)
    label_image = sitk.RelabelComponent(connected_components)
    num_components = sitk.GetArrayFromImage(label_image).max()
    lesion_count[str(thresh[i]) + '_to_' + str(thresh[i+1]) ] = num_components
    

print(lesion_count)
# Plot the histogram
# plt.figsize(16,16)
plt.figure(figsize=(16,16))
plt.subplot(2, 2, 1,)
plt.grid()
plt.hist(flattened_array[(flattened_array>130)&(flattened_array<200)], bins=30)
plt.xlabel("Intensity in HU (Hounsfield Unit)")
plt.ylabel("Frequency")
plt.title("Histogram of Intensities class 1 (130 HU - 199HU)")
# plt.show()

plt.subplot(2, 2, 2)
# Plot the histogram
plt.grid()
plt.hist(flattened_array[(flattened_array>200)&(flattened_array<300)], bins=30)
plt.xlabel("Intensity in HU (Hounsfield Unit)")
plt.ylabel("Frequency")
plt.title("Histogram of Intensities class 2 (200HU - 299HU)")
# plt.show()

plt.subplot(2, 2, 3)
# Plot the histogram
plt.grid()
plt.hist(flattened_array[(flattened_array>300)&(flattened_array<400)], bins=30)
plt.xlabel("Intensity in HU (Hounsfield Unit)")
plt.ylabel("Frequency")
plt.title("Histogram of Intensities class 3 (300HU - 399HU)")
# plt.show()

plt.subplot(2, 2, 4)
# Plot the histogram
plt.grid()
plt.hist(flattened_array[(flattened_array>400)], bins=100)
plt.xlabel("Intensity in HU (Hounsfield Unit)")
plt.ylabel("Frequency")
plt.title("Histogram of Intensities class 3 (400HU and above)")
plt.savefig('histogram.png')
plt.show()