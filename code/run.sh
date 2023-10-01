device="cuda:$1"
use_meta=$2
use_sim=$3
use_global=$4
use_local=$5
use_fusion=$6
test=0
resort_data=0
if [ $# -eq 4 ];
then
    test=$4
fi

# device="cuda:0"
# data_path="/home/hatim/data/DataCompletion/data"
data_path="/home_nfs/haitao/data/yht/DataCompletion/data"
city="chengdu"
interval_size=15
min_num=1
mask_rate="0.2"
hist_size=8
max_speed=40
# training_file="${data_path}/${city}/train_output_dataset_shuffle_${mask_rate}.txt"
# validation_file="${data_path}/${city}/val_output_dataset_shuffle_${mask_rate}.txt"
# test_file="${data_path}/${city}/test_output_dataset_shuffle_${mask_rate}.txt"
# training_file="${data_path}/${city}/train_output_dataset_shuffle_sim_detail_${mask_rate}"
# validation_file="${data_path}/${city}/val_output_dataset_shuffle_sim_detail_${mask_rate}.txt"
# test_file="${data_path}/${city}/test_output_dataset_shuffle_sim_detail_${mask_rate}.txt"
training_file="${data_path}/${city}/train_output_dataset_shuffle_sim_detail_${interval_size}_${min_num}_${mask_rate}_${hist_size}_${max_speed}"
validation_file="${data_path}/${city}/val_output_dataset_shuffle_sim_detail_${interval_size}_${min_num}_${mask_rate}_${hist_size}_${max_speed}.txt"
test_file="${data_path}/${city}/test_output_dataset_shuffle_sim_detail_${interval_size}_${min_num}_${mask_rate}_${hist_size}_${max_speed}.txt"
new_test_file="${data_path}/${city}/new_test_output_dataset_shuffle_sim_detail_${interval_size}_${min_num}_${mask_rate}_${hist_size}_${max_speed}.txt"
avg_speed_file="${data_path}/${city}/linkid_avg_speed.txt"
edge_file="${data_path}/${city}/new_edge_dict.pk"
graph_file="${data_path}/${city}/subgraphs_128.pk"
# training parameters
train_batch_size=100
num_workers=5
# num_workers=0
test_batch_size=500
epochs=300
batch_per_ep=500
lr=0.0001
early_stop_epoch=100
grad_clamp=10
weight_decay=0.1
# model parameters
use_meta=0
# out_spatial_dim=50
# out_temporal_dim=50
# graph_layer=2
# rnn_layer=2
# spatial_context_dim=50
# temporal_context_dim=50
# hidden_size=100

out_spatial_dim=100
out_temporal_dim=5
graph_layer=1
rnn_layer=2
spatial_context_dim=10
temporal_context_dim=20
hidden_size=150

threshold=0.1
rep_scale_weight=10000.0
mkdir -p run_log
python pipeline.py --city=${city} --device=${device} --training_file=${training_file} --validation_file=${validation_file} --test_file=${test_file} --new_test_file=${new_test_file} --avg_speed_file=${avg_speed_file} \
                --edge_file=${edge_file} --graph_file=${graph_file} --train_batch_size=${train_batch_size} --num_workers=${num_workers} \
                --test_batch_size=${test_batch_size} --epochs=${epochs} --batch_per_ep=${batch_per_ep} --lr=${lr} --early_stop_epoch=${early_stop_epoch} --grad_clamp=${grad_clamp} \
                --weight_decay=${weight_decay} --out_spatial_dim=${out_spatial_dim} --out_temporal_dim=${out_temporal_dim} \
                --graph_layer=${graph_layer} --rnn_layer=${rnn_layer} --spatial_context_dim=${spatial_context_dim} \
                --temporal_context_dim=${temporal_context_dim} --hidden_size=${hidden_size} --use_meta=${use_meta} \
                --use_sim=${use_sim} --use_global=${use_global} --use_local=${use_local} --use_fusion=${use_fusion} \
                --threshold=$threshold --rep_scale_weight=${rep_scale_weight} --test=${test} --resort_data=${resort_data}
                #> run_log/${use_meta}_${use_sim}.log