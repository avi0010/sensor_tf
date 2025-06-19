# File and directory paths
DATA_GENERATED = data_generated
DATA_PROCESSED = data_processed
RESULTS = results
MODEL_SAVE_DIR = training

# Default goal: `make` will invoke this by default if no target is provided
.DEFAULT_GOAL := train

# Rule to generate Tensor dataset (for example, if raw data is missing)
$(DATA_GENERATED):
	python3 -m sensors.scripts.preprocess

$(DATA_PROCESSED): $(DATA_GENERATED)
	python3 -m sensors.scripts.save_dataset_tfRecord

# Rule to train the model with specific parameters
train: $(DATA_PROCESSED)
	python3 -m sensors.scripts.train \
		--learning_rate 0.0001 \
		--feature_length 27 \
		--epochs 50 \
		--threshold 0.5 \
		--hidden_layers 32 \
		--pos_weight 0.1 \
		--batch_size 1024 \
		--heads 2 \
		--linformer_dim 64\
		--gamma 0.975
