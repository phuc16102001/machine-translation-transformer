export CUDA_VISIBLE_DEVICES=0
python3 inference.py --model_folder "../models" \
    --prompt "Dân số Việt Nam hiện đang là khoảng 90 triệu người, và nó vẫn có xu hướng tăng đều." 