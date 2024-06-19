# 离线模型训练框架

## 任务
本分支相较于主分支主要修改或新增了以下子任务：
1. 意图分类模型，对应配置文件为`./config/app_intent_config`。
2. 文本匹配模型，对应配置文件为`./config/app_text_match_config`。

## 环境
### 方法一
`conda create -n SINO python=3.8`  
`source activate SINO`  
`pip install -r requirements.txt`  
  
### 方法二
`conda create -n SINO python=3.8`  
`source activate SINO`  
`pip install tensorflow==2.4.0`  
`pip install params-flow==0.8.2`  
`pip install tensorflow-addons==0.12.0`  
`pip install pandas==1.2.3`  

## 运行
### 意图分类模型
#### 训练  
将`main.py`中的`config_file`设置为`./config/app_intent_config`，执行`python main.py`。  
#### 测试  
执行`python ./test/eval_classify.py`。

### 文本匹配模型
#### 训练  
将`main.py`中的`config_file`设置为`./config/app_text_match_config`，执行`python main.py`。  
#### 测试
执行`python ./test/eval_text_match.py`。
