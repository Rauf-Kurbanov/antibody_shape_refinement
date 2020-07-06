Predicting test data: 
python predict_test_data.py \
--model simple_coordinates \
--input_size 100 \
--output_size 9 \
--hidden_dim 256 \
--n_layers 1 \
--saved_model_path ../../results/simplemodel_coord_backup_copy \
--test_data_path_seq ../../results/seq_data_emb \
--test_data_path_coord ../../results/coord_data_emb