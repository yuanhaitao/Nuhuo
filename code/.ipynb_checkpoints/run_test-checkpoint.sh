device="cuda:$1"
use_meta=$2
use_sim=$3
# device="cuda:0"
data_path="/home/hatim/data/DataCompletion/data"
city="chengdu"
mask_rate="0.2"
# training_file="${data_path}/${city}/train_output_dataset_shuffle_${mask_rate}.txt"
# validation_file="${data_path}/${city}/val_output_dataset_shuffle_${mask_rate}.txt"
# test_file="${data_path}/${city}/test_output_dataset_shuffle_${mask_rate}.txt"
training_file="${data_path}/${city}/train_output_dataset_shuffle_sim_${mask_rate}"
validation_file="${data_path}/${city}/val_output_dataset_shuffle_sim_${mask_rate}.txt"
test_file="${data_path}/${city}/test_output_dataset_shuffle_sim_${mask_rate}.txt"
edge_file="${data_path}/${city}/new_edge_dict.pk"
graph_file="${data_path}/${city}/subgraphs_128.pk"
# training parameters
train_batch_size=100
num_workers=4
test_batch_size=500
epochs=300
batch_per_ep=500
lr=0.001
early_stop_epoch=30
grad_clamp=10
weight_decay=0.1
# model parameters
# use_meta=0
out_spatial_dim=50
out_temporal_dim=50
graph_layer=2
rnn_layer=2
spatial_context_dim=50
temporal_context_dim=50
hidden_size=100
threshold=0.1
rep_scale_weight=10.0
time_flag=$4
mkdir -p run_log
python pipeline_test.py --device=${device} --test_file=${test_file} \
                --edge_file=${edge_file} --graph_file=${graph_file} --num_workers=${num_workers} \
                --test_batch_size=${test_batch_size} --out_spatial_dim=${out_spatial_dim} --out_temporal_dim=${out_temporal_dim} \
                --graph_layer=${graph_layer} --rnn_layer=${rnn_layer} --spatial_context_dim=${spatial_context_dim} \
                --temporal_context_dim=${temporal_context_dim} --hidden_size=${hidden_size} --use_meta=${use_meta} \
                --use_sim=${use_sim} --time_flag=${time_flag} > run_log/test.${time_flag}.log