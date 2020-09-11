# coding=utf-8

from time import time
from data_pre import mk_save_dir, load_sub
from train_model import fit_model, evaluate_model
from keras.utils import plot_model
from keras.models import load_model
from save_info import *
from model import *
from atten_layer import AttentionLayer

os.environ["PATH"] += os.pathsep + 'C:/C1_Install_package/Graphviz/Graphviz 2.44.1/bin'

root_dir = None
total_confu_matrix = None
for once in range(total_times):

    # 定义变量
    start = time()

    root_dir, save_dir = mk_save_dir(once)
    acc_list = []
    confu_matrix = None  # use to save_to_csv
    confu_matri = None  # use to calculate aver_confu_mat
    numParams = None
    save_model_dir = None
    info_dict = {}

    for sub in range(total_sub_num):

        # 定义每次循环一个sub的变量
        save_model_dir = save_dir + str(sub) + '.h5'
        confu_mat = None
        acc = None
        model = None

        # erect model
        if os.path.exists(save_model_dir):
            model = load_model(save_model_dir, custom_objects={'AttentionLayer': AttentionLayer})
        else:
            if len(model_names) == 1:
                model = erect_single_model()
            else:
                if share_model:
                    model = erect_share_model()  # 现在的share-model代码只能基于EEGNet，还没有扩展到所有模型
                elif attention_mechanism:  # 融合时，加不加注意力机制
                    model = erect_n_branch_model_with_attention()
                else:
                    model = erect_n_branch_model()

        # show model
        if sub == 0 and once == 0:
            model.summary()
        if is_plot_model:
            plot_model(model, to_file=root_dir + 'model_structure.png', show_shapes=True)
            print('\n===================finish plot model image===================\n')
        print('\n===================== start training {}th fold ==========================\n'.format(sub))

        # load data
        print('loading sub {} data'.format(sub))
        (X_train, Y_train), (X_test, Y_test) = load_sub(sub)  # load data

        # fit model
        if os.path.exists(save_model_dir):
            acc, confu_mat = evaluate_model(model, X_test, Y_test)
        else:
            hist = fit_model(model, X_train, Y_train, X_test, Y_test, save_model_dir)
            save_training_pic(sub, hist, save_dir)  # save the training trend
            acc, confu_mat = evaluate_model(model, X_test, Y_test)  # evaluate model

        # save info
        numParams = model.count_params()
        info_dict['numParams'] = numParams
        confu_matrix = my_append_row(confu_matrix, confu_mat)  # save confu_matrix to save_to_csv
        acc_list = np.append(acc_list, acc)  # save each sub's acc
        print("Classification accuracy: %f " % (acc))

        del X_train, Y_train, X_test, Y_test
        import gc

        gc.collect()

    # save to file
    end = time()
    training_time = (end - start) / 3600
    total_confu_matrix = my_append_col(total_confu_matrix, confu_matrix)
    info_dict['training_time'] = str(training_time) + ' hours'
    save_acc_pic(acc_list, save_dir)
    save_csv(confu_matrix, save_dir)
    save_txt(info_dict, save_dir)
save_total_csv(total_confu_matrix, root_dir)
