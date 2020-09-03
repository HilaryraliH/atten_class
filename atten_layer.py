
from keras import backend as Bk
from keras.engine.topology import Layer
from keras import initializers


class self_attention(Layer):  # 输入：(samples, 46, 128) : 46个词，每一个128维 即一行是一个输入向量

    def __init__(self, d2, **kwargs):
        self.init = initializers.get('normal')
        self.d2 = d2
        self.d3 = 40  # 可以改变
        self.N = 0  # 在build函数中改变其值，但总有点不放心似的，总觉得会出bug
        super(self_attention, self).__init__()

    def get_config(self):
        config = {
            'd2': self.d2
        }
        base_config = super(self_attention, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

    def build(self, input_shape):
        self.N = input_shape[1]
        assert len(input_shape) == 3
        print('################################# in bulid func: ###############################')
        print('inputshape: ', input_shape)
        length = input_shape[2]
        self.wq = Bk.variable(self.init((length, self.d3)))
        self.wk = Bk.variable(self.init((length, self.d3)))
        self.wv = Bk.variable(self.init((length, self.d2)))
        super(self_attention, self).build(input_shape)

    def call(self, x, mask=None): # 这里，若去掉了mask=None，则会出错！！！为什么？
        print('##################in func call:###############\n')
        print('x.shape: ', x.shape)
        print(x.shape[2])
        '''3维的计算'''
        Q = Bk.dot(x, self.wq)  # samples，N,d1     d1,d3  ->    N,d3 # batch_dot 也可以换成 dot
        K = Bk.dot(x, self.wk)  # samples，N,d1     d1,d3  ->    N,d3
        V = Bk.dot(x, self.wv)  # samples，N,d1     d1,d2  ->    N,d2
        print('Q,K,V.shape: ', Q.shape, K.shape, V.shape)
        alpha = Bk.softmax(Bk.batch_dot(K, Bk.permute_dimensions(Q, (0, 2, 1))), axis=1)  # samples，N,N
        alpha = Bk.permute_dimensions(alpha, (0, 2, 1))  # samples，N,N
        print('alpha.shape: ', alpha.shape) 
        H = Bk.batch_dot(alpha, V) # 每一行是一个注意力权重向量，与V的各行加权和，得到H各行
        print('H.shape: ',H.shape)
        return H # samples,N,d2

    def compute_output_shape(self, input_shape):
        return input_shape[0], input_shape[1], self.d2


class alpha_attention(Layer):  

    def __init__(self, **kwargs):
        self.init = initializers.get('normal')
        self.d = 1  # 可以改变
        super(alpha_attention, self).__init__()

    def get_config(self):
        config = {
            'd': self.d
        }
        base_config = super(alpha_attention, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

    def build(self, input_shape):
        assert len(input_shape) == 3
        print('################################# in bulid func: ###############################')
        print('inputshape: ', input_shape)
        self.w = Bk.variable(self.init((1, input_shape[1])))
        super(alpha_attention, self).build(input_shape)

    def call(self, x, mask=None): # 这里，若去掉了mask=None，则会出错！！！为什么？
        print('##################in func call:###############\n')
        print('x.shape: ', x.shape)
        print(x.shape[2])
        '''3维的计算'''
        Q = Bk.dot(self.w,x)  # samples，N,d1     d1,d3  ->    N,d3 # batch_dot 也可以换成 dot
        print('Q,K,V.shape: ', Q.shape)
        return Q # samples,N,d2

    def compute_output_shape(self, input_shape):
        return input_shape[0], 1,input_shape[-1]


# JNE文章中的
class AttentionLayer(Layer):
    def __init__(self, **kwargs):
        super(AttentionLayer, self).__init__(** kwargs)
    
    def build(self, input_shape):
        assert len(input_shape)==3
        # W.shape = (time_steps, time_steps)
        self.W = self.add_weight(name='att_weight', 
                                 shape=(input_shape[1], input_shape[1]),
                                 initializer='uniform',
                                 trainable=True)
        super(AttentionLayer, self).build(input_shape)

    def call(self, inputs, mask=None):
        # inputs.shape = (batch_size, time_steps, seq_len)
        x = Bk.permute_dimensions(inputs, (0, 2, 1))
        # x.shape = (batch_size, seq_len, time_steps)
        # general
        a = Bk.softmax(Bk.tanh(Bk.dot(x, self.W)))
        a = Bk.permute_dimensions(a, (0, 2, 1))
        outputs = a * inputs
        outputs = Bk.sum(outputs, axis=1)
        return outputs

    def compute_output_shape(self, input_shape):
        return input_shape[0], input_shape[2]



# 网上的idmb数据集的，没有跑通
# class Attention(Layer):

#     def __init__(self, nb_head, size_per_head, **kwargs):
#         self.nb_head = nb_head
#         self.size_per_head = size_per_head
#         self.output_dim = nb_head * size_per_head
#         super(Attention, self).__init__(**kwargs)

#     def build(self, input_shape):
#         self.WQ = self.add_weight(name='WQ',
#                                   shape=(input_shape[0][-1], self.output_dim),
#                                   initializer='glorot_uniform',
#                                   trainable=True)
#         self.WK = self.add_weight(name='WK',
#                                   shape=(input_shape[1][-1], self.output_dim),
#                                   initializer='glorot_uniform',
#                                   trainable=True)
#         self.WV = self.add_weight(name='WV',
#                                   shape=(input_shape[2][-1], self.output_dim),
#                                   initializer='glorot_uniform',
#                                   trainable=True)
#         super(Attention, self).build(input_shape)

#     def Mask(self, inputs, seq_len, mode='mul'):
#         if seq_len == None:
#             return inputs
#         else:
#             mask = K.one_hot(seq_len[:, 0], K.shape(inputs)[1])
#             mask = 1 - K.cumsum(mask, 1)
#             for _ in range(len(inputs.shape) - 2):
#                 mask = K.expand_dims(mask, 2)
#             if mode == 'mul':
#                 return inputs * mask
#             if mode == 'add':
#                 return inputs - (1 - mask) * 1e12

#     def call(self, x):
#         # 如果只传入Q_seq,K_seq,V_seq，那么就不做Mask
#         # 如果同时传入Q_seq,K_seq,V_seq,Q_len,V_len，那么对多余部分做Mask
#         if len(x) == 3:
#             Q_seq, K_seq, V_seq = x
#             Q_len, V_len = None, None
#         elif len(x) == 5:
#             Q_seq, K_seq, V_seq, Q_len, V_len = x
#         # 对Q、K、V做线性变换
#         Q_seq = K.dot(Q_seq, self.WQ)
#         Q_seq = K.reshape(Q_seq, (-1, K.shape(Q_seq)[1], self.nb_head, self.size_per_head))
#         Q_seq = K.permute_dimensions(Q_seq, (0, 2, 1, 3))
#         K_seq = K.dot(K_seq, self.WK)
#         K_seq = K.reshape(K_seq, (-1, K.shape(K_seq)[1], self.nb_head, self.size_per_head))
#         K_seq = K.permute_dimensions(K_seq, (0, 2, 1, 3))
#         V_seq = K.dot(V_seq, self.WV)
#         V_seq = K.reshape(V_seq, (-1, K.shape(V_seq)[1], self.nb_head, self.size_per_head))
#         V_seq = K.permute_dimensions(V_seq, (0, 2, 1, 3))
#         # 计算内积，然后mask，然后softmax
#         A = K.batch_dot(Q_seq, K_seq, axes=[3, 3]) / self.size_per_head ** 0.5
#         A = K.permute_dimensions(A, (0, 3, 2, 1))
#         A = self.Mask(A, V_len, 'add')
#         A = K.permute_dimensions(A, (0, 3, 2, 1))
#         A = K.softmax(A)
#         # 输出并mask
#         O_seq = K.batch_dot(A, V_seq, axes=[3, 2])
#         O_seq = K.permute_dimensions(O_seq, (0, 2, 1, 3))
#         O_seq = K.reshape(O_seq, (-1, K.shape(O_seq)[1], self.output_dim))
#         O_seq = self.Mask(O_seq, Q_len, 'mul')
#         return O_seq

#     def compute_output_shape(self, input_shape):
#         return (input_shape[0][0], input_shape[0][1], self.output_dim)
