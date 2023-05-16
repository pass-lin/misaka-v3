# -*- coding: utf-8 -*-
"""
Created on Wed May 25 13:47:50 2022

@author: Administrator
"""
#import os
#os.environ['TF_KERAS'] = '1'
from .my_bert4keras.models import *
class GatedAttentionUnit_cross(Layer):
    """门控注意力单元
    链接：https://arxiv.org/abs/2202.10447
    介绍：https://kexue.fm/archives/8934
    在苏神基础上支持cross-attention
    """
    def __init__(
        self,
        units,
        key_size,
        activation='swish',
        use_bias=True,
        normalization='squared_relu',
        attention_scale=True,
        attention_dropout=None,
        kernel_initializer='glorot_uniform',
        low_rank=False,
        **kwargs
    ):
        super(GatedAttentionUnit_cross, self).__init__(**kwargs)
        self.units = units
        self.key_size = key_size
        self.activation = activation
        self.use_bias = use_bias
        self.normalization = normalization
        self.attention_scale = attention_scale
        self.attention_dropout = attention_dropout
        self.kernel_initializer = initializers.get(kernel_initializer)
        self.low_rank=low_rank
    def initializer(self, shape, dtype=None, order=3, gain=1.0):
        if shape[0] > 10000 or shape[0] < 10:
            hidden_size = shape[1]
        else:
            hidden_size = shape[0]
        gain *= (self.num_hidden_layers*5/2)**(-1. / order)
        stddev = 1.13684723 / hidden_size**0.5 * gain
        return K.truncated_normal(shape, stddev=stddev)
    @integerize_shape
    def build(self, input_shape):
        super(GatedAttentionUnit_cross, self).build(input_shape)
        hidden_size = input_shape[-1]
        if isinstance(hidden_size, (list, tuple)):
            hidden_size = input_shape[0][-1]
        if self.low_rank:
            self.kv_dense = Dense(
                units=self.key_size + self.key_size,
                activation=self.activation,
                use_bias=self.use_bias,
                kernel_initializer=self.kernel_initializer
            )
            self.high_rank=Dense(
                units=self.units,
                activation=self.activation,
                use_bias=self.use_bias,
                kernel_initializer=self.kernel_initializer
            )
        else:
            self.kv_dense = Dense(
                units=self.units + self.key_size,
                activation=self.activation,
                use_bias=self.use_bias,
                kernel_initializer=self.kernel_initializer
            )
        self.uq_dense = Dense(
            units=self.units + self.key_size,
            activation=self.activation,
            use_bias=self.use_bias,
            kernel_initializer=self.kernel_initializer
        )
        self.o_dense = Dense(
            units=hidden_size,
            use_bias=self.use_bias,
            kernel_initializer=self.kernel_initializer
        )


    @recompute_grad
    def call(self, inputs, mask=None, a_bias=False, p_bias=None):
        if not isinstance(inputs, list):
            inputs, mask = [inputs], [mask]
        x,c= inputs[:2]
        n=2

        mask = None if mask is None else mask[1]
        if a_bias:
            a_bias = inputs[n]
            n += 1
        # 投影变换
        x = self.uq_dense(x)
        u,q = tf.split(x, [self.units, self.key_size], axis=-1)
        
        c=self.kv_dense(c)
        if self.low_rank:
            v,k = tf.split(c, [self.key_size, self.key_size], axis=-1)
        else:
            v,k = tf.split(c, [self.units, self.key_size], axis=-1)
        # 加入RoPE

        if p_bias == 'rotary':
            q, k = apply_rotary_position_embeddings(inputs[n], q, k)
        # Attention

        a = tf.matmul(q, tf.transpose(k,[0,2,1]))#tf.einsum('bmd,bnd->bmn', q, k)
       
        if self.attention_scale:
            a = a / self.key_size**0.5
        A = attention_normalize(a, mask, -1, self.normalization, a_bias)
        if self.attention_dropout:
            A = Dropout(self.attention_dropout)(A)
            

        # 计算输出
        A=tf.matmul(A, v)#tf.einsum('bmn,bnd->bmd', A, v)
        if self.low_rank:
            A = self.high_rank(A)
        o = self.o_dense(u * A)
        

        return o

    def compute_mask(self, inputs, mask=None):
        return mask
    def compute_output_shape(self, input_shape):
        if isinstance(input_shape[0], (list, tuple)):
            return input_shape[0]
        else:
            return input_shape

    def get_config(self):
        config = {
            'low_rank':self.low_rank,
            'units': self.units,
            'key_size': self.key_size,
            'activation': activations.serialize(self.activation),
            'use_bias': self.use_bias,
            'normalization': self.normalization,
            'attention_scale': self.attention_scale,
            'attention_dropout': self.attention_dropout,
            'kernel_initializer':
                initializers.serialize(self.kernel_initializer),
        }
        base_config = super(GatedAttentionUnit_cross, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))
class RWKV_Simply(Layer):
    """
    链接：https://arxiv.org/abs/2202.10447
    介绍：https://kexue.fm/archives/8934
    在苏神基础上支持cross-attention
    """
    def __init__(
        self,
        units,
        key_size,
        num_heads,
        gate_activation='sigmoid',
        use_bias=False,
        bidirectional=True,
        attention_scale=True,
        attention_dropout=None,
        kernel_initializer='glorot_uniform',
        **kwargs
    ):
        super(RWKV_Simply, self).__init__(**kwargs)
        self.units = units
        self.key_size = key_size
        self.num_heads = num_heads
        self.activation = gate_activation
        self.use_bias = use_bias
        self.bidirectional = bidirectional
        self.attention_scale = attention_scale
        self.attention_dropout = attention_dropout
        self.kernel_initializer = initializers.get(kernel_initializer)
    @integerize_shape
    def build(self, input_shape):
        super(RWKV_Simply, self).build(input_shape)
        hidden_size = input_shape[-1]
        if isinstance(hidden_size, (list, tuple)):
            hidden_size = input_shape[0][-1]
        self.v_dense = Dense(
            units=self.key_size,
            use_bias=self.use_bias,
            kernel_initializer=self.kernel_initializer
        )
        self.k_dense = Dense(
            units=self.key_size,
            use_bias=self.use_bias,
            kernel_initializer=self.kernel_initializer
        )
        self.r_dense = Dense(
            units=self.key_size,
            activation=self.activation,
            use_bias=self.use_bias,
            kernel_initializer=self.kernel_initializer
        )
        self.o_dense = Dense(
            units=self.units,
            use_bias=self.use_bias,
            kernel_initializer=self.kernel_initializer
        )
        self.time_k=self.add_weight(
                shape=(1,1,self.units), initializer="ones",name='self.time_k'
            )
        self.time_v=self.add_weight(
                shape=(1,1,self.units), initializer="ones",name='self.time_v'
            )
        self.time_r=self.add_weight(
                shape=(1,1,self.units), initializer="ones",name='self.time_r'
            )
        self.WKV=WKV(self.key_size,self.num_heads,self.bidirectional)
    def time_shift(self,x):
        return K.temporal_padding(x,[1,0])[:,:-1]
    @recompute_grad
    def call(self, inputs, mask=None, a_bias=False, p_bias=None):
        if not isinstance(inputs, list):
            inputs, mask = [inputs], [mask]
        x,r,w= inputs[:]              
        if mask!=None and mask[0]!=None:
            mask=K.cast(K.expand_dims(mask[0],-1),x.dtype)
            x*=mask    
        xx=self.time_shift(x)
        
        k=self.k_dense(x*self.time_k+xx*(1-self.time_k))
        v=self.v_dense(x*self.time_v+xx*(1-self.time_v))
        r=self.r_dense(r*self.time_r+(1-self.time_r)*self.time_shift(r))
        
        
        wkv=self.WKV([w,k,v])
        if wkv.dtype!=r.dtype:
            wkv=K.cast(wkv,r.dtype)
        if self.attention_dropout:
            wkv = Dropout(self.attention_dropout)(wkv)
        rwkv=r*wkv
        return self.o_dense(rwkv)
    def compute_mask(self, inputs, mask=None):
        return mask[0]
    def compute_output_shape(self, input_shape):
        return input_shape[0]
    def get_config(self):
        config = {
            'units': self.units,
            'key_size': self.key_size,
            'num_heads' : self.num_heads,
            'gate_activation': activations.serialize(self.activation),
            'use_bias': self.use_bias,
            'bidirectional': self.bidirectional,
            'attention_scale': self.attention_scale,
            'attention_dropout': self.attention_dropout,
            'kernel_initializer':
                initializers.serialize(self.kernel_initializer),
        }
        base_config = super(RWKV_Simply, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

class WKV(Layer):
    def __init__(
            self,
            hidden_size,
            num_heads,
            bidirectional=True,
            **kwargs
        ):
            super(WKV, self).__init__(**kwargs)
            self.bidirectional = bidirectional
            self.num_heads = num_heads
            self.hidden_size=hidden_size
    def get_config(self):
        config = {
            'bidirectional': self.bidirectional,
            'num_heads' : self.num_heads,
            'hidden_size':self.hidden_size
        }
        base_config = super(WKV, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))
    def compute_output_shape(self, input_shape):
        return input_shape[-1]
    def build(self, input_shape):
        super(WKV, self).build(input_shape)

        h = K.arange(self.num_heads,dtype=K.floatx())
        self.decay_speed = -5 + 8 * (h / (self.num_heads-1))**2
        self.decay_speed = tf.reshape(self.decay_speed,[-1,1,1])
        self.channal_rate=self.add_weight(
                shape=(self.num_heads,1,1), initializer="ones",name='time-deceay'
            )
        self.time_first=self.add_weight(
                shape=(1,1,self.hidden_size), initializer="zeros",name='time-first'
            )
    def clip(self,k):
        return K.clip(k,-10,6)
    def call(self, inputs, mask=None, a_bias=False, p_bias=None):
        w ,k ,v = inputs[:]
        k=self.clip(k)
        k=K.exp(k)
        B,L=  K.shape(k)[0],K.shape(k)[-2]
        w = K.exp(K.exp(K.minimum(self.decay_speed*self.channal_rate, 3))*w)
        u = K.exp(self.time_first)
        

        D = self.hidden_size
        
        kv = k*v
        
        kv_t = tf.transpose(K.reshape(kv,[B,L,self.num_heads,D//self.num_heads]),[0,2,1,3])
        k_t = tf.transpose(K.reshape(k,[B,L,self.num_heads,D//self.num_heads]),[0,2,1,3])
        if self.bidirectional:    
            kv_t=K.concatenate([kv_t[:,:,:,:D//self.num_heads//2],
                                K.reverse(kv_t[:,:,:,:D//self.num_heads//2], -1)],axis=-1)
            k_t=K.concatenate([k_t[:,:,:,:D//self.num_heads//2],
                                K.reverse(k_t[:,:,:,:D//self.num_heads//2], -1)],axis=-1)
        kv_t = w@kv_t
        k_t = w@k_t
        if self.bidirectional:    
            kv_t=K.concatenate([kv_t[:,:,:,:D//self.num_heads//2],
                                K.reverse(kv_t[:,:,:,:D//self.num_heads//2], -1)],axis=-1)
            k_t=K.concatenate([k_t[:,:,:,:D//self.num_heads//2],
                                K.reverse(k_t[:,:,:,:D//self.num_heads//2], -1)],axis=-1)
        wkv = K.reshape(tf.transpose(kv_t ,[0,2,1,3]),[B,L,D])+kv*u
        wk = K.reshape(tf.transpose(k_t ,[0,2,1,3]),[B,L,D])+k*u
        
       
        wkv=wkv/(wk+1e-5)
        return wkv
class Get_Weight(Layer):

    def call(self, inputs):
        seq_len = K.shape(inputs)[1]
        idxs = K.arange(1, seq_len+1,dtype=K.floatx())
        mask = idxs[None, :] < idxs[:, None]
        mask = K.cast(mask, K.floatx())
        idxs = K.expand_dims(idxs,0)-K.expand_dims(idxs,1)
        idxs = K.expand_dims(idxs*mask+(mask-1)*1e9,0)
        return idxs
    def compute_output_shape(self, input_shape):
        return [1,input_shape[1],input_shape[1]]
class ReZero(Layer):
    def build(self, input_shape):
        super(ReZero, self).build(input_shape)
        self.alpha=self.add_weight(
                shape=(1,1,1), initializer="zeros"
            )
    def call(self, inputs):
        xi,x=inputs[:]
        return xi+K.abs(self.alpha)*x
class Channal_mix(FeedForward):
    @integerize_shape
    def build(self, input_shape):
        super(Channal_mix, self).build(input_shape)
        self.r_gate= Dense(
            units=input_shape[-1],
            activation='sigmoid',
            name='channal_gate',
            use_bias=self.use_bias,
            kernel_initializer=self.kernel_initializer,
        )
        self.time_k=self.add_weight(
                shape=(1,1,input_shape[-1]), initializer="ones",name='self.time_k'
            )
        self.time_r=self.add_weight(
                shape=(1,1,input_shape[-1]), initializer="ones",name='self.time_r'
            )
    def time_shift(self,x):
        return K.temporal_padding(x,[1,0])[:,:-1]
    def call(self, inputs):
        xx = self.time_shift(inputs)
        xk = self.time_k * inputs + (1-self.time_k) * xx
        xr = self.time_r * inputs + (1-self.time_r) * xx
        r=self.r_gate(xr)
        y=super(Channal_mix, self).call(xk)
        return r*y

class LM_W(object):
    """定义下三角Attention Mask（语言模型用）"""

    def compute_attention_bias(self, inputs=None):
        """通过idxs序列的比较来得到对应的mask"""
        if self.attention_bias is None:

                

            self.attention_bias = self.apply(
                inputs=self.inputs[0],
                layer=Get_Weight,
                name="Get-W-mat",
            )

        return self.attention_bias
class BiRWKV_Base(LM_W,Transformer):
    """构建只有encoder的RWKV模型"""
    def __init__(self, with_mlm=False,num_heads=64, **kwargs):
        super(BiRWKV_Base, self).__init__(**kwargs)
        self.with_mlm = with_mlm
    def get_inputs(self):
        """BERT的输入是token_ids和segment_ids
        （但允许自行传入位置id，以实现一些特殊需求）
        """
        x_in = self.apply(
            layer=Input, shape=(self.sequence_length,), name="Input-Token"
        )
        return x_in

    def apply_embeddings(self, inputs):
        """BERT的embedding是token、position、segment三者embedding之和"""
        x = inputs

        x = self.apply(
            inputs=x,
            layer=Embedding,
            input_dim=self.vocab_size,
            output_dim=self.embedding_size,
            embeddings_initializer=self.initializer,
            mask_zero=True,
            name="Embedding-Token",
        )
        x = self.apply(
            inputs=x,
            layer=LayerNormalization,
            zero_mean=False,
            scale=True,
            offset=False,
            name="Embedding-Norm",
        )
        x = self.apply(
            inputs=x, layer=Dropout, rate=self.dropout_rate, name="Embedding-Dropout"
        )
        if self.embedding_size!=self.hidden_size:
            x = self.apply(
                inputs=x,
                layer=Dense,
                units=self.hidden_size,
                kernel_initializer=self.initializer,
                name="Embedding-Mapping",
            )

        return x
    def compute_attention_bias(self, inputs=None):
        """修改LM Mask的序列长度（从 self.inputs[0] 改为 self.inputs[1] ）
        """
        old_inputs = self.inputs[:]
        self.inputs = [old_inputs[0]]
        mask = super(BiRWKV_Base, self).compute_attention_bias(inputs)
        self.inputs = old_inputs
        return mask
    def apply_main_layers(self, inputs, index):
        '对齐bert'
        'RWKV_simple的采用postNorm RWKV->ADD->LN->FFN->ADD->LN'
        x = inputs

        attention_name = "Transformer-%d-RWKV" % index
        feed_forward_name = "Transformer-%d-FeedForward" % index
        w = self.compute_attention_bias(index)

        # RWKV
        xi = x
        
        x = [x, x, w]


        x = self.apply(
            inputs=x,
            layer=RWKV_Simply,
            units=self.hidden_size,
            num_heads = self.num_attention_heads,
            key_size=self.hidden_size,
            kernel_initializer=self.initializer,
            attention_dropout=self.attention_dropout_rate,
            name=attention_name,
        )
        x = self.apply(
            inputs=x,
            layer=Dropout,
            rate=self.dropout_rate,
            name="%s-Dropout" % attention_name,
        )
        x = self.apply(inputs=[xi, x], layer=Add, name="%s-Add" % attention_name)
        x = self.apply(
            inputs=x,
            layer=LayerNormalization,
            zero_mean=False,
            scale=True,
            offset=False,
            name="%s-RMSNorm" % attention_name,
        )

        # Feed Forward
        
        xi = x
        
        
        
        
        x = self.apply(
            inputs=x,
            layer=Channal_mix,
            units=self.intermediate_size,
            activation=self.hidden_act,
            use_bias=False,
            kernel_initializer=self.initializer,
            name=feed_forward_name,
        )
        x = self.apply(
            inputs=x,
            layer=Dropout,
            rate=self.dropout_rate,
            name="%s-Dropout" % feed_forward_name,
        )
        x = self.apply(inputs=[xi, x], layer=Add, name="%s-Add" % feed_forward_name)
        x = self.apply(
            inputs=x,
            layer=LayerNormalization,
            zero_mean=False,
            scale=True,
            offset=False,
            name="%s-RMSNorm" % feed_forward_name,
        )

        return x

    def apply_final_layers(self, inputs):
        """根据剩余参数决定输出"""
        x = inputs
        outputs = [x]


        if self.with_mlm:
            # Masked Language Model部分
            
            
            
            x = self.apply(
                inputs=x,
                layer=Dense,
                units=self.embedding_size,
                activation=self.hidden_act,
                kernel_initializer=self.initializer,
                name="MLM-Dense",
            )
            
            x =self.apply(
                inputs=x,
                layer=LayerNormalization,
                zero_mean=False,
                scale=True,
                offset=False,
                name='OUT-Norm',
            )
            
            x = self.apply(
                inputs=x,
                layer=Embedding,
                arguments={"mode": "dense"},
                name="Embedding-Token",
            )
            x = self.apply(inputs=x, layer=ScaleOffset, scale=False, name="MLM-Bias")
            mlm_activation = "softmax" if self.with_mlm is True else self.with_mlm
            x = self.apply(
                inputs=x,
                layer=Activation,
                activation=mlm_activation,
                name="MLM-Activation",
            )
            outputs.append(x)

        if len(outputs) == 1:
            outputs = outputs[0]
        elif len(outputs) == 2:
            outputs = outputs[1]
        else:
            outputs = outputs[1:]

        return outputs
class RWKV_encoder(BiRWKV_Base):
    def apply_final_layers(self, inputs):
        """根据剩余参数决定输出"""
        return inputs
class RWKV_decoder(BiRWKV_Base):
    def __init__(self, with_lm=True, **kwargs):
        super(RWKV_decoder, self).__init__(**kwargs)
        self.with_lm = with_lm
class Misaka_Base(RoFormerV2):
    def initializer(self, shape, dtype=None, order=3, gain=1.0):
        return super(Misaka_Base, self).initializer(shape, dtype, order, gain)
    def variable_mapping(self):
        pass
class Misaka_encoder(Misaka_Base):
    """基于GAU-α的encoder
    链接：https://kexue.fm/archives/9052
    """
    def get_inputs(self):
        """Misaka的Encoder的输入只有token_ids
        """
        x_in = self.apply(
            layer=Input,
            shape=(self.sequence_length,),
            name='Encoder-Input-Token'
        )
        return x_in
    def apply_embeddings(self, inputs):
        """
        Misaka embeding只有word embeding
        """
        x=inputs

        x = self.apply(
            inputs=x,
            layer=Embedding,
            input_dim=self.vocab_size,
            output_dim=self.embedding_size,
            embeddings_initializer=self.initializer,
            mask_zero=True,
            name='Embedding-Token'
        )
        x = self.apply(
            inputs=x,
            layer=Dropout,
            rate=self.dropout_rate,
            name='Embedding-Dropout'
        )
        if self.embedding_size != self.hidden_size:
            x = self.apply(
                inputs=x,
                layer=Dense,
                units=self.hidden_size,
                use_bias=False,
                kernel_initializer=self.initializer,
                name='Embedding-Mapping'
            )

        return x
    def apply_main_layers(self, inputs, index):
        """Misaka-encoder 的主体是基于Gated Attention Unit的模块
        顺序：GAU  --> Add --> LN
        """
        x = inputs

        attention_name = 'Misaka-Encoder-%d-GatedAttentionUnit' % index
        attention_mask = self.compute_attention_bias(index)
        position_bias = self.compute_position_bias(x)
        
        # Self Attention
        xi = x
        x = [x, position_bias]
        arguments = {'a_bias': None, 'p_bias': 'rotary'}
        if attention_mask is not None:
            arguments['a_bias'] = True
            x.insert(1, attention_mask)
        x = self.apply(
            inputs=x,
            layer=GatedAttentionUnit,
            arguments=arguments,
            units=self.intermediate_size,
            key_size=self.attention_key_size,
            activation=self.hidden_act,
            use_bias=False,
            normalization='softmax_plus',
            attention_dropout=self.attention_dropout_rate,
            kernel_initializer=self.initializer,
            name=attention_name
        )
        x = self.apply(
            inputs=x,
            layer=Dropout,
            rate=self.dropout_rate,
            name='%s-Dropout' % attention_name
        )
        x = self.apply(
            inputs=[xi, x], layer=Add, name='%s-Add' % attention_name
        )
        x = self.apply(
            inputs=x,
            layer=LayerNormalization,
            zero_mean=False,
            scale=False,
            offset=False,
            name='%s-Norm' % attention_name
        )

        return x
    def apply_final_layers(self, inputs):
        """剩余部分
        """
        x = inputs
        x = self.apply(
            inputs=x,
            layer=Dropout,
            rate=self.dropout_rate,
            name='Encoder-Output-Dropout'
        )
        return x
class Misaka_decoder(LM_Mask,Misaka_Base):
    """Misaka模型（Decoder）
    """
    def __init__(self, with_lm=True, **kwargs):
        super(Misaka_decoder, self).__init__(**kwargs)
        self.with_lm = with_lm
        self.num_hidden_layers=self.num_hidden_layers//2
    def initializer(self, shape, dtype=None, order=3, gain=1.0):
        if shape[0] > 10000 or shape[0] < 10:
            hidden_size = shape[1]
        else:
            hidden_size = shape[0]
        gain *= (self.num_hidden_layers*5)**(-1. / order)
        stddev = 1.13684723 / hidden_size**0.5 * gain
        return K.truncated_normal(shape, stddev=stddev)
    def apply_embeddings(self, inputs):
        c, x = inputs

        x = self.apply(
            inputs=x,
            layer=Embedding,
            input_dim=self.vocab_size,
            output_dim=self.embedding_size,
            embeddings_initializer=self.initializer,
            mask_zero=True,
            name='Embedding-Token'
        )
        x = self.apply(
            inputs=x,
            layer=Dropout,
            rate=self.dropout_rate,
            name='Decoder-Embedding-Dropout'
        )
        if self.embedding_size != self.hidden_size:
            x = self.apply(
                inputs=x,
                layer=Dense,
                units=self.hidden_size,
                kernel_initializer=self.initializer,
                name='Decoder-Embedding-Mapping'
            )

        return [c, x]
    def get_inputs(self):
        """Misaka的Decoder的输入为context序列和token_ids
        """
        c_in = self.apply(
            layer=Input,
            shape=(self.sequence_length, self.hidden_size),
            name='Input-Context'
        )
        x_in = self.apply(
            layer=Input,
            shape=(self.sequence_length,),
            name='Decoder-Input-Token'
        )
        return [c_in, x_in]
    def apply_main_layers(self, inputs, index):
        """Misaka-encoder 的主体是基于Gated Attention Unit的模块
        顺序：LN --> GAU1 --> Add --> LN --> cross-attention  --> Add -->  LN --> GAU  --> Add
        其中cross-attention我使用的是自己改的GAU
        """
        c, x  = inputs[:]
        
        self_attention_1_name='Misaka-Dncoder-%d-GatedAttentionUnit-1' % index
        cross_attention_name = 'Misaka-Dncoder-%d-GatedAttentionUnit-cross' % index
        self_attention_2_name='Misaka-Dncoder-%d-GatedAttentionUnit-2' % index
        attention_mask = self.compute_attention_bias(index)
        position_bias = self.compute_position_bias(x)

        # GAU-1
        xi = x
        x = [x, position_bias]
        arguments = {'a_bias': None, 'p_bias': 'rotary'}
        if attention_mask is not None:
            arguments['a_bias'] = True
            x.insert(1, attention_mask)
        
        x = self.apply(
            inputs=x,
            layer=GatedAttentionUnit,
            arguments=arguments,
            units=self.intermediate_size,
            key_size=self.attention_key_size,
            activation=self.hidden_act,
            use_bias=False,
            normalization='softmax_plus',
            attention_dropout=self.attention_dropout_rate,
            kernel_initializer=self.initializer,
            name=self_attention_1_name
        )
        x = self.apply(
            inputs=x,
            layer=Dropout,
            rate=self.dropout_rate,
            name='%s-Dropout' % self_attention_1_name
        )
        x = self.apply(
            inputs=[xi, x], layer=Add, name='%s-Add' % self_attention_1_name
        )
        x = self.apply(
            inputs=x,
            layer=LayerNormalization,
            zero_mean=False,
            scale=False,
            offset=False,
            name='%s-Norm' % self_attention_1_name
        )
        
        # Cross Attention
        xi=x
        argument = {'a_bias': None}
        x = self.apply(
            inputs=[x,c],
            layer=GatedAttentionUnit_cross,
            units=self.intermediate_size,
            key_size=self.attention_key_size,
            activation=self.hidden_act,
            use_bias=False,
            arguments=argument,
            normalization='softmax_plus',
            attention_dropout=self.attention_dropout_rate,
            kernel_initializer=self.initializer,
            name=cross_attention_name
        )
        x = self.apply(
            inputs=x,
            layer=Dropout,
            rate=self.dropout_rate,
            name='%s-Dropout' % cross_attention_name
        )
        x = self.apply(
            inputs=[xi, x], layer=Add, name='%s-Add' % cross_attention_name
        )
        x = self.apply(
            inputs=x,
            layer=LayerNormalization,
            zero_mean=False,
            scale=False,
            offset=False,
            name='%s-Norm' % cross_attention_name
        )
        
        # GAU-2
        xi = x
        x = [x, position_bias]
        arguments = {'a_bias': None, 'p_bias': 'rotary'}
        if attention_mask is not None:
            arguments['a_bias'] = True
            x.insert(1, attention_mask)
        x = self.apply(
            inputs=x,
            layer=GatedAttentionUnit,
            arguments=arguments,
            units=self.intermediate_size,
            key_size=self.attention_key_size,
            activation=self.hidden_act,
            use_bias=False,
            normalization='softmax_plus',
            attention_dropout=self.attention_dropout_rate,
            kernel_initializer=self.initializer,
            name=self_attention_2_name
        )
        x = self.apply(
            inputs=x,
            layer=Dropout,
            rate=self.dropout_rate,
            name='%s-Dropout' % self_attention_2_name
        )
        x = self.apply(
            inputs=[xi, x], layer=Add, name='%s-Add' % self_attention_2_name
        )
        x = self.apply(
            inputs=x,
            layer=LayerNormalization,
            zero_mean=False,
            scale=False,
            offset=False,
            name='%s-Norm' % self_attention_2_name
        )

        return [c, x]

    def apply_final_layers(self, inputs):
        """剩余部分
        """
        c,x = inputs

        if self.with_lm:
            # 预测token概率部分
            if self.embedding_size != self.hidden_size:
                x = self.apply(
                    inputs=x,
                    layer=Dense,
                    units=self.embedding_size,
                    use_bias=False,
                    kernel_initializer=self.initializer,
                    name='Output-Mapping'
                )
            x = self.apply(
                inputs=x,
                layer=Dropout,
                rate=self.dropout_rate,
                name='Output-Output-Dropout'
            )
            Output_activation = 'softmax' if self.with_lm is True else self.with_lm
            
            x = self.apply(
                    inputs=x,
                    layer=Dense,
                    units=self.vocab_size,
                    activation= Output_activation,
                    use_bias=False,
                    kernel_initializer=self.initializer,
                    name='Decoder-Output-LM'
                )
        return x
    def compute_attention_bias(self, inputs=None):
        """修改LM Mask的序列长度（从 self.inputs[0] 改为 self.inputs[1] ）
        """
        old_inputs = self.inputs[:]
        self.inputs = [old_inputs[1]]
        mask = super(Misaka_decoder, self).compute_attention_bias(inputs)
        self.inputs = old_inputs
        return mask
class Misaka(Misaka_Base):
    """Misaka模型（Encoder-Decoder）
    """
    def __init__(self, **kwargs):
        super(Misaka, self).__init__(**kwargs)
        kwargs['layers'] = self.layers
        e_name, d_name = 'Misaka_encoder', 'Misaka_decoder'
        if 'name' in kwargs:
            e_name = '%s_%s' % (kwargs['name'], e_name)
            d_name = '%s_%s' % (kwargs['name'], d_name)
            del kwargs['name']  # 防止重复传参
        self._encoder = Misaka_encoder(name=e_name, **kwargs)
        self._decoder = Misaka_decoder(name=d_name, **kwargs)
    
    def build(self, **kwargs):
        """同时构建Encoder和Decoder
        """
        self._encoder.build(**kwargs)
        self._decoder.build(**kwargs)
        self._decoder.position_bias = None  # 下面call时将重新初始化
        self.encoder = self._encoder.model
        self.decoder = self._decoder.model
        self.inputs = self.encoder.inputs + self.decoder.inputs[1:]
        self.outputs = self._decoder.call(
            self.encoder.outputs + self.decoder.inputs[1:]
        )
        self.model = Model(self.inputs, self.outputs)
class Misaka_decoder_V3(Misaka_decoder):
    def apply_main_layers(self, inputs, index):
        """Misaka-encoder 的主体是基于Gated Attention Unit的模块
        顺序：LN --> cross-attention  --> Add  -->LN --> GAU1 --> Add 
        其中cross-attention我使用的是自己改的GAU
        """
        c, x  = inputs[:]
        
        self_attention_1_name='Misaka-Dncoder-%d-GatedAttentionUnit-1' % index
        cross_attention_name = 'Misaka-Dncoder-%d-GatedAttentionUnit-cross' % index
        feed_forward_name = "Transformer-%d-FeedForward" % index
        attention_mask = self.compute_attention_bias(index)
        position_bias = self.compute_position_bias(x)


        # GAU-1
        
        xi = x
        x = [x, x,position_bias]
        arguments = {'a_bias': None, 'p_bias': 'rotary'}
        if attention_mask is not None:
            arguments['a_bias'] = True
            x.insert(2, attention_mask)
        
        x = self.apply(
            inputs=x,
            layer=GatedAttentionUnit_cross,
            units=self.intermediate_size,
            key_size=self.attention_key_size,
            activation=self.hidden_act,
            use_bias=False,
            normalization='softmax_plus',
            attention_dropout=self.attention_dropout_rate,
            kernel_initializer=self.initializer,
            arguments=arguments,
            low_rank=True,
            name=self_attention_1_name
        )
        
        x = self.apply(
            inputs=x,
            layer=Dropout,
            rate=self.dropout_rate,
            name='%s-Dropout' % self_attention_1_name
        )
        x = self.apply(
            inputs=[xi, x], layer=Add, name='%s-Add' % self_attention_1_name
        )
        
        x = self.apply(
            inputs=x,
            layer=LayerNormalization,
            zero_mean=False,
            scale=True,
            offset=False,
            name='%s-RMSNorm' % self_attention_1_name
        ) 
        
        # Cross Attention
        
        xi=x
        argument = {'a_bias': None}
        x = self.apply(
            inputs=[x,c],
            arguments=argument,
            layer=GatedAttentionUnit_cross,
            units=self.intermediate_size,
            key_size=self.attention_key_size,
            activation=self.hidden_act,
            use_bias=False,
            normalization='softmax_plus',
            attention_dropout=self.attention_dropout_rate,
            kernel_initializer=self.initializer,
            low_rank=True,
            name=cross_attention_name
        )
        x = self.apply(
            inputs=x,
            layer=Dropout,
            rate=self.dropout_rate,
            name='%s-Dropout' % cross_attention_name
        )
        x = self.apply(
            inputs=[xi, x], layer=Add, name='%s-Add' % cross_attention_name
        )
        x = self.apply(
            inputs=x,
            layer=LayerNormalization,
            zero_mean=False,
            scale=True,
            offset=False,
            name='%s-RMSNorm' % cross_attention_name
        )
        
        # FFN
        
        xi=x
        x = self.apply(
            inputs=x,
            layer=FeedForward,
            units=self.intermediate_size,
            activation=[self.hidden_act,'linear'],
            use_bias=False,
            kernel_initializer=self.initializer,
            name=feed_forward_name
        )
        x = self.apply(
            inputs=x,
            layer=Dropout,
            rate=self.dropout_rate,
            name='%s-Dropout' % feed_forward_name
        )
        x = self.apply(
            inputs=[xi, x], layer=Add, name='%s-Add' % feed_forward_name
        )
        x = self.apply(
            inputs=x,
            layer=LayerNormalization,
            zero_mean=False,
            scale=True,
            offset=False,
            name='%s-RMSNorm' % feed_forward_name
        )
        return [c, x]
class Misaka_V3(Misaka):
    """Misaka模型（Encoder-Decoder）
    """
    def __init__(self, **kwargs):
        super(Misaka_V3, self).__init__(**kwargs)
        kwargs['layers'] = self.layers
        e_name, d_name = 'Misaka_encoder', 'Misaka_decoder'
        if 'name' in kwargs:
            e_name = '%s_%s' % (kwargs['name'], e_name)
            d_name = '%s_%s' % (kwargs['name'], d_name)
            del kwargs['name']  # 防止重复传参
        self._encoder = Misaka_encoder_V3(name=e_name, **kwargs)
        self._decoder = Misaka_decoder_V3(name=d_name, **kwargs)
    
class Misaka_encoder_V3(Misaka_encoder):
    def apply_main_layers(self, inputs, index):

        x = inputs

        attention_name = "Misaka-%d-MultiHeadSelfAttention" % index
        attention_mask = self.compute_attention_bias(index)
        position_bias = self.compute_position_bias(x)

        # Self Attention
        xi = x
        
        x = [x, x, position_bias]
        arguments = {"a_bias": None, "p_bias": "rotary"}
        if attention_mask is not None:
            arguments["a_bias"] = True
            x.insert(3, attention_mask)
        x = self.apply(
            inputs=x,
            layer=GatedAttentionUnit_cross,
            units=self.intermediate_size,
            key_size=self.attention_key_size,
            activation=self.hidden_act,
            use_bias=False,
            normalization='softmax_plus',
            attention_dropout=self.attention_dropout_rate,
            kernel_initializer=self.initializer,
            arguments=arguments,
            name=attention_name
        )
        
        x = self.apply(
            inputs=x,
            layer=Dropout,
            rate=self.dropout_rate,
            name="%s-Dropout" % attention_name,
        )
        x = self.apply(inputs=[xi, x], layer=Add, name="%s-Add" % attention_name)
        x = self.apply(
            inputs=x,
            layer=LayerNormalization,
            zero_mean=False,
            scale=True,
            offset=False,
            name="%s-Norm" % attention_name,
        )
        return x 
    