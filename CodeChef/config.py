EPOCH = 200
cuda_devices = '3' # choose cuda device to run
with_df = True
test_mode = True
edge_type = 5
prob = 'sumtrian'
token_size=200
feature_sizes = {'mnmx':50, 'subinc':50, 'flow016':70, 'sumtrian':50}
hidden_sizes = {'mnmx':100, 'subinc':200, 'flow016':200, 'sumtrian':200}
feature_size=feature_sizes[prob]
hidden_size=hidden_sizes[prob]
label_size= 4
heads = 2
data_dir = '../MFGNNPreparation/codes/codechef/'

