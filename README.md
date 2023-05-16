# misaka-v3 模型改进
misaka-writer迎来了第三次升级，这次升级是在[V2 版本的基础](https://github.com/pass-lin/misaka-writer-V2/edit/main/README.md)上做出了部分的改进   
首先参数量进一步升级，从V2的200M升级到了400M  
然后对模型结果做了改进，这次依然是采用GAU，但GAU去掉了苏神所使用的scaleoffset，qk分别使用两个独立的矩阵非线性变换而来 
然后是对decoder的改进，在V3里主要是想尽可能对模型提速，因此为了模型的推理速度我做出下列改进   
1.对于GAU的单头注意力做出低秩分解，原本的GAU是U * qkv ，现在GAU变成了U * qkv W。在原本的GAU里qk是维度是一样的，而UV是一个很大的维度。分解后qkv都是一样的小维度，通过W转化成和U一样的维度。这么做是主要是为了减少attention二次复杂度的计算量和减少cache的通讯量。当然效果会有所下降，但和我们翻倍的参数量比起来这不算什么  
2.在V2中我们使用的GAU-CrossGAU-GAU，在V3里把最后一个GAU换成的FFN。这么做也是为了在提参数量的情况下提速，FFN显然比GAU快，也能节省一倍的cache通讯量。除此之外就是我觉得俩GAU夹三明治好丑啊，所以改成了对齐原版transformer的形式  
3.激励函数换成了relu，这主要是和苏神交流过，他的实验里GAU用了relu效果甚至略好于swish。因为我的结构主要参考自苏神，就继承了这一点。  
4.模型全部使用tf2实现，并且提供了一个cpu和gpu都可用的attention-cache加速。当然不是说我放弃了tf1哈，只是tf2写cache比较方便。本文模型是在nvidia-tf下训练的，如果你不需要cache本文模型可以兼容tf1.14-tf2.12    
综上所述不难看出，本次优化主要是提速提质。在参数量翻倍的情况下，不论有没有cache v3模型都比V2要快  

 # 环境配置  
 虽然说训练兼容tf1，但推理部分都是在tf2写的所以本次环境配置只提供tf2的参考  
 对于 30系以前的卡，推荐tensorflow 2.2.0：

```sh
conda create -n misaka-writer python=3.8
conda activate misaka-writer
conda install -c conda-forge pandas cudatoolkit=10.1 cudnn
pip install tensorflow-gpu==2.2.0 keras==2.3.1 sklearn==0.0 pandas 
conda install cudatoolkit=10.1 cudnn=7.6
```
  
 对于 30系以后的卡，推荐tensorflow 2.5.0：
```sh
conda create -n misaka-writer python=3.8
conda activate misaka-writer
conda install -c conda-forge pandas cudatoolkit=10.1 cudnn
pip install tensorflow-gpu==2.5.0  sklearn==0.0 pandas 
conda install cudatoolkit=11.2 cudnn=8.1
```

对于 A卡可以使用tensorflow-directml（只限 Windows 10/11 或 wsl，此版本不需要安装 CUDA）：

```sh
conda create -n misaka-writer python=3.7
conda activate misaka-writer
conda install -c conda-forge pandas
pip install tensorflow-directml==2.2 keras==2.3.1 sklearn sklearn==0.0 pandas 
```

## 使用方法

推理见 `generate_cache.py`，基本上用的是 V1 的优化器，此外对 cpu 写了简单的 cache 优化。

`model_path` 是模型的权重路径。

`num` 代表生成的下文的数量。 `text` 为输入，建议输入在 20 到 128 字之间。
第一次生成会比较慢，因为要编译，后面就快了  

微调见 `finetuning.py`，具体参数注释给出来了，注意的是读取的是一个文件夹下全部的csv进行训练，csv的参考格式在data文件夹里  
因为模型比较大所以默认os.environ['RECOMPUTE'] = '1'，也就是开启重计算。如果不差卡嫌慢可以直接注释掉的  
 
## 模型权重
  
因为模型比较大就不放qq了，这次都丢百度云供下载    
权重里带expand的是扩写模型，其他是续写模型  
通过百度网盘分享的文件：misaka-v…  
链接:https://pan.baidu.com/s/1vhgxnJ9-snIvXbxthpHZfg?pwd=h69i   
提取码:h69i  
复制这段内容打开「百度网盘APP 即可获取」  


## 社区

如有问题可加 Q 群905398734（本项目群），本人 qq 935499957

---

最后用ai生成的misaka镇楼  
![QQ图片20221109142639](https://user-images.githubusercontent.com/62837036/200754613-febeb470-7e27-4347-9b31-340e090b87ab.png)
