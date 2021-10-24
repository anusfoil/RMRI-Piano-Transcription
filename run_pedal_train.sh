# ============ Train piano transcription system from scratch ============
# MAESTRO dataset directory. Users need to download MAESTRO dataset into this folder.
DATASET_DIR="/import/c4dm-datasets/maestro-v2.0.0"

# Modify to your workspace
WORKSPACE="./workspace"

# --- 1. Train note transcription system ---
# python pytorch/main.py train --workspace=$WORKSPACE --model_type='Regress_onset_offset_frame_velocity_CRNN' --loss_type='regress_onset_offset_frame_velocity_bce' --augmentation='none' --max_note_shift=0 --batch_size=12 --learning_rate=5e-4 --reduce_iteration=10000 --resume_iteration=0 --early_stop=300000 --cuda

# --- 2. Train pedal transcription system ---
python pytorch/main.py train --workspace=$WORKSPACE --model_type='Regress_pedal_velocity_CRNN' --loss_type='regress_pedal_velocity' --augmentation='none' --max_note_shift=0 --batch_size=12 --learning_rate=5e-4 --reduce_iteration=10000 --resume_iteration=0 --early_stop=300000 --cuda
