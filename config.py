

# model_names, select_chan_way, dataformat_list,chans,chans_num 都具有相同的长度
is_plot_model = True # 在1080上，改为 False
model_names = ['Smaller_DeepConvNet','Smaller_DeepConvNet','Smaller_DeepConvNet']  # 先改只有一个模型融合的情况
select_chan_way = ['P_left','P_mid','P_right'] # 每个分支对应的输入数据
input_way = 'branch' # branch together 一起 或者 分支
sample_points = 200
total_times=1
epochs = 3
batch_size = 32
total_sub_num = 8

area_to_elecs = {
    'EOG':[1, 6],
    'F':[3, 4, 7, 8, 9, 10, 11, 12, 13, 14, 15],
    'C':[17, 18, 19, 20, 21, 22, 23, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42],
    'P':[2, 5, 16, 24, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61],
    '9':[16, 24, 54, 55, 57, 58, 59, 60, 61],
    'P_left': [16, 44, 45, 46, 47, 48, 54, 55, 56, 59, 60],
    'P_mid':  [48, 56, 60, 47, 55, 59, 49, 57, 61],
    'P_right':[24, 48, 49, 50, 51, 52, 56, 57, 58, 60, 61]
}

model_to_dataformat = {
    'EEGNet':'2D', 
    'ShallowConvNet':'2D',
    'DeepConvNet':'2D',
    'Smaller_DeepConvNet':'2D',
    'Transpose_Net':'2D',
    'Transfer_Proposed_Conv_R':'2D',
    'JNE_CNN':'3D',
    'Proposed_Conv':'3D',
    'Proposed_Conv_R':'3D',
    'generate_lstmfcn':'true_2D'
}

dataformat_list = []
for name in model_names:
    dataformat_list += [model_to_dataformat[name]]
if len(dataformat_list)!=len(select_chan_way):
    dataformat_list = dataformat_list*len(select_chan_way)

# 若是单个模型  chans_num: 数字  chans:列表
# 若是融合模型  chans_num: 列表  chans:二维列表

chans = [0]*len(select_chan_way)
chans_num = []
for i,key in enumerate(select_chan_way):
    chans[i] = area_to_elecs[key]
    chans_num += [len(chans[i])]
