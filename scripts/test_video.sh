#!/bin/bash
cd ..

# Define the root path and folder for DAVIS dataset -- MODIFY TO MATCH YOUR NEEDS.
Davis_root="./DAVIS/DAVIS-2017-trainval-480p/DAVIS"
root_folder="$Davis_root/Annotations/480p/"

# Counter for video folders
counter=1

# Loop through all directories in the root folder
for folder in "$root_folder"/*/; do
    # Extract the video name from the folder path
    video_name=$(basename "$folder")

    # Print the current video folder being processed
    echo "Processing Video Folder $counter: $video_name"
    
    ## Test 
    # Set the paths for RGB images and masks
    rgb_root="$Davis_root/JPEGImages/480p/$video_name"
    mask_root="$Davis_root/Annotations/480p/$video_name"

    # Set the result path for the current video
    result_path="./Results/testvideo/$video_name"

    # Run Attenuation Model
    python test_video.py --mask_root "$mask_root" --rgb_root "$rgb_root" --result_path "$result_path" --init_parameternet_weights "bestmodels/editnet_attenuate.pth" --result_for_decrease 1 --batch_size 1

    # Run Amplification Model
    python test_video.py --mask_root "$mask_root" --rgb_root "$rgb_root" --result_path "$result_path" --init_parameternet_weights "bestmodels/editnet_amplify.pth" --result_for_decrease 0 --batch_size 1

    # Increment the counter
    counter=$((counter + 1))

done
