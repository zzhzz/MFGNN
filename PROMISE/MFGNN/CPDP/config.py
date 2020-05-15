import json,sys
EPOCH = 80
cuda_devices = '7' # choose cuda device to run
with_df = True
test_mode = True
edge_type = 7
train_prob = sys.argv[1]
test_prob = sys.argv[2]
token_size=3120
feature_size=50
hidden_size=80
label_size = 1
heads = 3
data_dir = '../../../MFGNNPreparation/datas/promise/'

