#!/bin/bash

# Define variables
WEIGHTS=""
MODEL="small" # Option for model size
EXP_ID=""
GPUS=(0 1)  # List of GPU indices, adjust as per available GPUs

# Google Cloud destination for rclone uploads
GDRIVE_DEST=""

# Enable or disable rclone upload
ENABLE_RCLONE_UPLOAD=false

# List of datasets  "d17-test-dev" "y19-val" "mose-val"
datasets=("d17-val" "d17-test-dev")

# Check if the number of GPUs matches the number of datasets
if [ "${#GPUS[@]}" -ne "${#datasets[@]}" ]; then
    echo "Error: The number of GPUs (${#GPUS[@]}) does not match the number of datasets (${#datasets[@]})."
    exit 1
fi

# Loop through each dataset and assign a GPU, running each in the background
for i in "${!datasets[@]}"; do
    dataset="${datasets[$i]}"
    gpu="${GPUS[i % ${#GPUS[@]}]}"
    export CUDA_VISIBLE_DEVICES="$gpu"
    export OMP_NUM_THREADS=1
    echo "Running evaluation on $dataset with GPU $gpu"

    python oasis/eval_vos.py \
        dataset="$dataset" \
        weights="$WEIGHTS" \
        model="$MODEL" \
        exp_id="$EXP_ID" &

    # Wait for the evaluation to complete before proceeding

    # Skip upload if the dataset is "d17-val"
    if [ "$dataset" == "d17-val" ]; then
        echo "Skipping upload for $dataset as it does not generate a .zip file."
        continue
    fi

    # Upload to Google Cloud using rclone if ENABLE_RCLONE_UPLOAD is true
    if [ "$ENABLE_RCLONE_UPLOAD" = true ]; then
        ANNOTATION_FILE="results/${EXP_ID}/${dataset}/${EXP_ID}_${dataset}.zip"
        echo "Uploading $ANNOTATION_FILE to Google Cloud at $GDRIVE_DEST"
        rclone copy "$ANNOTATION_FILE" "$GDRIVE_DEST"
    else
        echo "Skipping upload for $ANNOTATION_FILE as ENABLE_RCLONE_UPLOAD is set to false."
    fi
done

# Wait for all background jobs to complete
wait