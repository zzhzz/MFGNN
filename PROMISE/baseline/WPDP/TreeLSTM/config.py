import sys
EPOCH = 40

cuda_devices = '6' # choose cuda device to run
train_proj = sys.argv[1]
test_proj = sys.argv[2]
with_df = True
token_size=53000
feature_size=70
hidden_size=100
label_size=1

