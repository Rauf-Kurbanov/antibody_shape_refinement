#!/bin/sh
python3 main.py --model simple_coordinates \
--model_input_size=100 \
--model_output_size=9 \
--model_hidden_dim=256 \
--learning_rate=0.001 \
--n_layers=1 \
--batch_size=32 \
--epochs=5000 \
--save_prefix=test