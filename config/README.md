# config 配置参数说明

## 参数说明:
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;一般的，每个任务需要配置一个json格式的config文件，该文件包含当前任务下的所有参数，主要包含四个类别的参数:
### 任务参数:
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;任务参数主要包含当前任务的基本信息，一般仅需修改版本参数。
如:
```
    "task_info":{
        "name":"ar_domain_intent", #任务名称
        "task":"classify",   #任务类别，支持classify, ner, joint_learning
        "version":"v4.0"    #版本信息
    }
```
### 数据集参数:
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;数据集参数主要是指定训练集文件的路径，一般无需改动，如:
```
    "corpus_info":{
        "train_data_file":"./data/smart_app/train_data.csv", 
        "dev_data_file":"./data/smart_app/dev_data.csv",
        "test_data_file":"./data/smart_app/test_data.csv"
    }
```
### 训练器参数:
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;训练器参数主要是模型训练时和模型结构无关的超参配置，一般无需改动，如:
```
    "trainer_info":{
        "data_processor":"NerDataPreprocess", #数据预处理类名，支持ClassifyDataPreprocess, NerDataPreprocess, JointLearningDataPreprocess
        "trainer":"NerTrainer",  #训练器类名, 支持ClassifyTrainer, Nertrainer, JointLearningTrainer
        "mode":"train",  #运行模式，支持train, test, save_model
        "epochs":5,
        "batch_size":64,
        "learning_rate":1e-4,
        "results_dir":"./results/",
        "checkpoint_path":"save_model_dir/cp-model.ckpt",
        "saved_model_path":"saved_model/model"
    }
```
### 模型参数:
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; 模型参数相对比较灵活，可能需要进行修改，这部分参数可以分成三部分，下面分别对这三类参数进行举例说明:
#### 模型全局配置参数:
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; 全局配置参数一般情况下是不需经常修改的
```
    encoder_info":{
         "encoder":"NerEncoder", #模型结构定义类名, 支持ClassifierEncoder, NerEncoder, JointLearningEncoder
         "max_seq_len":16,  #query的最大长度   
         "vocab_size":21128,        #词表大小
         "tag_class_num":4          #NER任务tag类别数
         "intent_class_num":9       #分类任务intent类别数
    } 
```
#### 模型Embedding配置参数:
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; 这里集成了多种Embedding表示，需要根据选择的Embedding类别添加所需的参数，如:
```
    ### WordEmbedding:
    encoder_info":{
        "embedding":"WordEmbedding",
        "embedding_dims":128
    }
	
    ### WinPoolEmbedding:
    encoder_info":{
        "embedding":"WinPoolEmbedding",
        "region_size":3,
        "embedding_dims":128
    }
	
    ### ScalarRegionEmbedding:
    encoder_info":{
        "embedding":"ScalarRegionEmbedding",
        "region_size":3,
        "embedding_dims":128
    }
	
    ### EnhancedWordEmbedding:
    encoder_info":{
        "embedding":"EnhancedWordEmbedding",
        "scale":0.5,
        "embedding_dims":128
    }
	
    ### ContextWordRegionEmbedding:
    encoder_info":{
        "embedding":"ContextWordRegionEmbedding",
        "region_size":3,
        "embedding_dims":128
    }
	
    ### WordContextRegionEmbedding:
    encoder_info":{
        "embedding":"WordContextRegionEmbedding",
        "region_size":3,
        "embedding_dims":128
    }
```
#### 模型表示层配置参数:
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; 表示层需要指定隐层的网络结构，如:
```
    encoder_info":{
         "layers":["inputs", "albert", "lstm", "dense"], #表示inputs+albert+lstm+dense的网络结构      
         "hidden_size":64,  #lstm 隐层大小，dense.units参数用的也是这个参数
         "bert":"albert_base" #bert类别配置，支持的类别见common.bert_factory类注释
    }
    #layers中包含的层，所需要的参数都要加到encoder_info里。
    #隐层网络的定义非常灵活，这里目前支持的类别和对应需要添加到encoder_info中的参数映射关系如下:
    #inputs: pass 
    #embedding: pass
    #dense: 加入一个hidden_size
    #lamda: pass
    #pool1d: pass
    #lstm: 加入一个hidden_size
    #cnn:
        encoder_info":{
            "layers":["inputs", "embedding", "cnn"],       
            "num_filters":128,
            "kernel_sizes":[3,4,5]
        }
    #idcnn:
        encoder_info":{
            "layers":["inputs", "embedding", "idcnn"],       
            "num_filters":128,
            "filter_width":3,
            "repeat_times":4,
            "num_layers":[1,1,2]
        }
    #transformer:
        encoder_info":{
            "layers":["inputs", "embedding", "transformer"],       
            "dim_model": 64,
            "num_heads":8
        }       
```
#### 模型参数完整配置示例:
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; 一般将上述三类参数放在encoder_info里，即可完成整体的参数配置。如:
```
    ### NER-task inputs+albert+lstm+dense网络结构的encoder_info完整表示
    encoder_info":{
         "encoder":"NerEncoder", #模型结构定义类名, 支持ClassifierEncoder, NerEncoder, JointLearningEncoder
         "max_seq_len":16,
         "layers":["inputs", "albert", "lstm", "dense"],
         "vocab_size":21128,        
         "hidden_size":64,
         "bert":"albert_base",  #用bert时，无embedding参数配置
         "tag_class_num":4
	}
```

## 数据格式:
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;数据为json格式(分类型任务只有query和intent两列)。如:
```
{"query": "帮我打开粤北网", "intent": "AppOpen", "tags": ["O", "O", "O", "O", "B_APP", "I_APP", "E_APP"]} #联合训练
{"query": "帮我打开粤北网", "intent": "AppOpen"}  #分类
{"query": "帮我打开粤北网", "tags": ["O", "O", "O", "O", "B_APP", "I_APP", "E_APP"]}  #NER
```

## 完整示例:
   * 分类任务: config.app_intent_test_config
   * NER任务: config.app_ner_test_config
   * 联合训练: config.app_test_config