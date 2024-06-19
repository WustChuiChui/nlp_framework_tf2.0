import sys
from config.config_parser import ConfigParser
from utils.preprocess import *
from trainer.trainer import *
from common.const import *

class App:
    def __init__(self):
        pass

    def __call__(self, config_file):
        config = ConfigParser(config_file=config_file)
        print("config params: ", config)

        processor = getattr(sys.modules[DATA_PREPROCESS_MODULE], config.trainer_info.data_processor)()

        train, dev, test, vocab_size = processor(config)
        print('x_train shape: {}, x_test shape: {}'.format(train[0].shape, test[0].shape))

        trainer = getattr(sys.modules[TRAINER_MODULE], config.trainer_info.trainer)(config)

        trainer.run(train, dev, test)

if __name__ == "__main__":
    # task config_file example
    # config_file = "./config/app_ner_config" #应用操作垂域NER task
    # config_file = "./config/app_intent_config" #应用操作垂域intent task
    # config_file = "./config/app_multi_task_config"  # 应用操作垂域multi-task
    # config_file = "./config/ar_domain_intent_config" #中控仲裁全垂域 task(车机&AR眼镜)
    # config_file = "./config/ar_domain_teacher_intent_config" #中控仲裁Teacher 分类模型
    # config_file = "./config/ar_domain_student_intent_config" #中控仲裁排序模型
    # config_file = "./config/arbitrator_intent_config" #孔明2.0中控仲裁(车机) task
    # config_file = "./config/classify_demo_test_config"


    # config_file = "./config/app_intent_config"  # 意图分类模型
    config_file = "./config/app_text_match_config"  # 文本匹配模型
    app = App()(config_file=config_file)
