EPOCH = 200
cuda_devices = '3' # choose cuda device to run
with_df = True
test_mode = True
edge_type = 5
prob = '721C'
token_size=200
feature_sizes={'1062C':70,'721C':50,'731C':50,'742C':50,'822C':50}
feature_size=feature_sizes[prob]
hidden_size=200
label_size= 5
heads = 2
data_dir = '../MFGNNPreparation/codes/codeforces/'

