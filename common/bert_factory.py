import tensorflow as tf
import bert

class BertFactory():
    def __init__(self, config):
        self.config = config
        """
        @support bert:
            {
                "albert_tiny":        "https://storage.googleapis.com/albert_zh/albert_tiny.zip",
                "albert_tiny_489k":   "https://storage.googleapis.com/albert_zh/albert_tiny_489k.zip",
                "albert_base":        "https://storage.googleapis.com/albert_zh/albert_base_zh.zip",
                "albert_base_36k":    "https://storage.googleapis.com/albert_zh/albert_base_zh_additional_36k_steps.zip",
                "albert_large":       "https://storage.googleapis.com/albert_zh/albert_large_zh.zip",
                "albert_xlarge":      "https://storage.googleapis.com/albert_zh/albert_xlarge_zh_177k.zip",
                "albert_xlarge_183k": "https://storage.googleapis.com/albert_zh/albert_xlarge_zh_183k.zip",
            }
        """
    def load_albert_model(self):
        albert_model_name = self.config.bert if hasattr(self.config, "bert") else "albert_tiny"
        albert_dir = bert.fetch_brightmart_albert_model(albert_model_name, "./bert/models")

        model_params = bert.params_from_pretrained_ckpt(albert_dir)
        model_params.vocab_size = model_params.vocab_size + 2
        model_params.adapter_size = 1

        al_bert = bert.BertModelLayer.from_params(model_params, name="albert")
        al_bert(tf.keras.layers.Input(shape=(self.config.max_seq_len,)))
        bert.load_albert_weights(al_bert, albert_dir)

        return al_bert