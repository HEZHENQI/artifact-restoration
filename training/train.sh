export OPENAI_LOGDIR="XXX"

MODEL_FLAGS="--image_size 256 --num_channels 128 --num_res_blocks 3 --learn_sigma True"
DIFFUSION_FLAGS="--diffusion_steps 4000 --noise_schedule linear"
TRAIN_FLAGS="--lr 1e-4 --batch_size 8"
python scripts/image_train.py --data_dir /.../XXX $MODEL_FLAGS $DIFFUSION_FLAGS $TRAIN_FLAGS
