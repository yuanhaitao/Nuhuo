# THINK

### Environment Requirement

please run the command 'conda install --yes --file requirements.txt' to install the environment

### Datasets

#### main data
We have preprocessed these data into the format of our model's input, where the sampling data (all data is too big, 200G+😁) is provided in the data folder. In particular, the training data are split into small files in the folder 'train_output_dataset_shuffle_sim_detail', and the validation data and test data are named in the file 'val_output_dataset_shuffle_sim_detail.txt' and 'test_output_dataset_shuffle_sim_detail.txt', each line corresponds to a sample.

#### auxiliary data
We also need some auxiliary data, such as the average speed and weights for computing metrics, and these files are named 'linkid_avg_speed_8.txt' and 'linkid_fre_chengdu_15.txt'
In addition, we also need the link and graph partition data, named 'new_edge_dict.pk' and 'subgraphs_128.pk'.



### Run the code

The executed command is as follows:

python ./code/pipeline.py --city=chengdu --device=cuda:0 --training_file=./data/chengdu/train_output_dataset_shuffle_sim_detail --validation_file=./data/chengdu/val_output_dataset_shuffle_sim_detail.txt --test_file=./data/chengdu/test_output_dataset_shuffle_sim_detail.txt --avg_speed_file=./data/chengdu/linkid_avg_speed_8.txt --weight_file=./data/chengdu/linkid_fre_chengdu_15.txt --edge_file=./data/chengdu/new_edge_dict.pk --graph_file=./data/chengdu/subgraphs_128.pk --train_batch_size=100 --num_workers=0  --test_batch_size=500 --epochs=500 --batch_per_ep=500 --lr=0.0001 --early_stop_epoch=30 --out_spatial_dim=100 --out_temporal_dim=5  --graph_layer=1 --spatial_context_dim=10  --temporal_context_dim=20 --hidden_size=150 --lambda_weight=1.0 --interval_size=15 --data_ratio=1 --hist_size=8 --time_slot_num=12 --mask_rate=0.2

- city: chengdu or xian, the default value is chengdu.
- device: cpu/cuda(0,1,2..), where cuda(0,1,...) means using GPU for training the model
- training_file: the training data folder, is required, such as './data/chengdu/train_output_dataset_shuffle_sim_detail'
- validation_file: the validation data file, is required, such as './data/chengdu/val_output_dataset_shuffle_sim_detail.txt'
- test_file: the test data file, is required, such as './data/chengdu/test_output_dataset_shuffle_sim_detail.txt'
- avg_speed_file: the average data file, is required, such as './data/chengdu/linkid_avg_speed_8.txt'
- weight_file: the weight data file, is required, such as './data/chengdu/linkid_fre_chengdu_15.txt'
- edge_file: the edge data file, is required, such as './data/chengdu/new_edge_dict.pk'
- graph_file: the graph partition data file, is required, such as './data/chengdu/subgraphs_128.pk', 128 means the partitoin numbers
- train_batch_size: the batch size for training the model
- num_workers: the number of workders for parallel data loading
- test_batch_size: the batch size when evaluating the model using test data
- epochs: the maximum learning epochs
- batch_per_ep: the number of batch for each epoch
- lr: the learning ratio
- early_stop_epoch: the maximum number of epochs for early stopping
- out_spatial_dim: the hyper-parameter $d_{s}$
- out_temporal_dim: the hyper-parameter $d_{t}$
- graph_layer: the hyper-parameter $N$
- spatial_context_dim: the hyper-parameter $d_g$
- temporal_context_dim: the hyper-parameter $d_w$
- hidden_size: the hyper-parameter $d_h$
- lambda_weight: the hyper-parameter $\lambda$
- interval_size: the data parameter $\Delta t$
- data_ratio: the ratio for training data
- hist_size: the data parameter $m$
- time_slot_num: the data parameter $T$, where $12,72,144,288$ means $1,6,12,24$ hours.
- mask_rate: the data parameter $\rho$