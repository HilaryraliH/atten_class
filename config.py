
model_to_dataformat = {
    'Convention_2D':'2D',
    'EEGNet':'2D', 
    'ShallowConvNet':'2D',
    'DeepConvNet':'2D',
    'Smaller_DeepConvNet':'2D',
    'Transpose_Net':'2D',
    'Transfer_Proposed_Conv_R':'2D',
    'JNE_CNN':'3D',
    'JNE_CNN_SEBlock':'3D',
    'Proposed_Conv':'3D',
    'Proposed_Conv_R':'3D',
    'generate_lstmfcn':'true_2D'
}

optimal_9 = [16, 24, 54, 55, 57, 58, 59, 60, 61]
area_to_elecs = {
    'EOG':[1, 6],
    'F':[3, 4, 7, 8, 9, 10, 11, 12, 13, 14, 15],
    'C':[17, 18, 19, 20, 21, 22, 23, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42],
    'P':[2, 5, 16, 24, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61],
    '9':optimal_9,
    '9_1': [optimal_9[0]],
    '9_2': [optimal_9[1]],
    '9_3': [optimal_9[2]],
    '9_4': [optimal_9[3]],
    '9_5': [optimal_9[4]],
    '9_6': [optimal_9[5]],
    '9_7': [optimal_9[6]],
    '9_8': [optimal_9[7]],
    '9_9': [optimal_9[8]],
    'P_left': [16, 44, 45, 46, 47, 48, 54, 55, 59], # [16, 44, 45, 46, 47, 48, 54, 55, 56, 59, 60],
    'P_mid':  [48, 56, 60, 47, 55, 59, 49, 57, 61],
    'P_right':[24, 49, 50, 51, 52, 57, 58, 60, 61] # [24, 48, 49, 50, 51, 52, 56, 57, 58, 60, 61]
}

area_to_elecs['F_9'] = area_to_elecs['F']+area_to_elecs['9']
area_to_elecs['F_C_9'] = area_to_elecs['F']+area_to_elecs['C']+area_to_elecs['9']


'''
原数据
1.单个模型，单个输入
2.单个模型，多个通道一起输入
3.一种模型，多个分支输入
4.多种模型，多个分支输入

bandpass：
1.一个模型，多个通道一起输入
2.一个模型，多个分支输入（数据分支，并不是通道分支）
'''

is_plot_model = False # 在1080上，改为 False
model_names = ['EEGNet']*3
select_chan_way = ['P_left','P_mid','P_right']
# select_chan_way = ['9_1','9_2','9_3','9_4','9_5','9_6','9_7','9_8','9_9'] # 每个分支对应的输入数据;
# 当bandpass= True 时，若一起输入，对5文件都提取相同的通道，也需要 用五个，如['9']*5
band_pass = False
attention_mechanism = False
share_model = True # 当建立参数共享的模型时，为true，此时model_names也需要有“分支数量”个，但建立真的模型时，只建立一个
mak_dir_other_info = '三个输入共享参数模型，基于4_40的'


band_pass_num = 5 # 滤波的数量
sample_points = 200
total_times=1
epochs = 20
batch_size = 32
total_sub_num = 8
data_dir = '.\\new_data\\TestDataCell_'
data_file_list = [data_dir+'4_40.mat']
if band_pass:
    # data_file_list = [data_dir+'05_4.mat',data_dir+'4_8.mat',data_dir+'8_12.mat',data_dir+'12_30.mat',data_dir+'30_40.mat']
    data_file_list = [data_dir+'05_40.mat',data_dir+'4_40.mat',data_dir+'8_40.mat',data_dir+'12_40.mat',data_dir+'30_40.mat']

dataformat_list = [] 
for name in model_names: # 种类
    dataformat_list += [model_to_dataformat[name]]
if len(dataformat_list)!=len(select_chan_way): # 数量，如果一个模型不分枝，但要融合多个电极的时候
    dataformat_list = dataformat_list*len(select_chan_way)
# chans和chans-num都是根据select-chan-way的长度来，都是长度和select-chan-way一样的列表
chans_index = [None]*len(select_chan_way)
chans_num = []
for i,key in enumerate(select_chan_way):
    chans_index[i] = area_to_elecs[key]
    chans_num += [len(chans_index[i])]

