# coding=utf-8

from __future__ import division, absolute_import, print_function
import os
import re
import urllib
import params_flow as pf
import tensorflow as tf
from tensorflow import keras
from bert import BertModelLayer, loader

_verbose = os.environ.get('VERBOSE', 1)  # verbose print per default
trace = print if int(_verbose) else lambda *a, **k: None

albert_models_brightmart = {
    "albert_tiny":        "https://storage.googleapis.com/albert_zh/albert_tiny.zip",
    "albert_tiny_489k":   "https://storage.googleapis.com/albert_zh/albert_tiny_489k.zip",
    "albert_base":        "https://storage.googleapis.com/albert_zh/albert_base_zh.zip",
    "albert_base_36k":    "https://storage.googleapis.com/albert_zh/albert_base_zh_additional_36k_steps.zip",
    "albert_large":       "https://storage.googleapis.com/albert_zh/albert_large_zh.zip",
    "albert_xlarge":      "https://storage.googleapis.com/albert_zh/albert_xlarge_zh_177k.zip",
    "albert_xlarge_183k": "https://storage.googleapis.com/albert_zh/albert_xlarge_zh_183k.zip",
}

def fetch_brightmart_albert_model(model_name: str, fetch_dir: str):
    if model_name not in albert_models_brightmart:
        raise ValueError("ALBERT model with name:[{}] not found at brightmart/albert_zh, try one of:{}".format(
            model_name, albert_models_brightmart))
    else:
        fetch_url = albert_models_brightmart[model_name]

    fetched_file = pf.utils.fetch_url(fetch_url, fetch_dir=fetch_dir)
    fetched_dir = pf.utils.unpack_archive(fetched_file)
    return fetched_dir

def _is_tfhub_model(tfhub_model_path):
    try:
        assets_files     = tf.io.gfile.glob(os.path.join(tfhub_model_path, "assets/*"))
        variables_files  = tf.io.gfile.glob(os.path.join(tfhub_model_path, "variables/variables.*"))
        pb_files = tf.io.gfile.glob(os.path.join(tfhub_model_path, "*.pb"))
    except tf.errors.NotFoundError:
        assets_files, variables_files, pb_files = [], [], []

    return len(pb_files) >= 2 and len(assets_files) >= 1 and len(variables_files) >= 2

def load_albert_weights(bert: BertModelLayer, tfhub_model_path, tags=[]):
    if not _is_tfhub_model(tfhub_model_path): 
        trace("Loading brightmart/albert_zh weights...")
        map_to_stock_fn = loader.map_to_stock_variable_name
        return loader.load_stock_weights(bert, tfhub_model_path, map_to_stock_fn=map_to_stock_fn)

    assert isinstance(bert, BertModelLayer), "Expecting a BertModelLayer instance as first argument"
    prefix = loader.bert_prefix(bert)

    with tf.Graph().as_default():
        sm = tf.compat.v2.saved_model.load(tfhub_model_path, tags=tags)
        with tf.compat.v1.Session() as sess:
            sess.run(tf.compat.v1.global_variables_initializer())
            stock_values = {v.name.split(":")[0]: v.read_value() for v in sm.variables}
            stock_values = sess.run(stock_values)

    loaded_weights = set()
    skip_count = 0
    weight_value_tuples = []
    skipped_weight_value_tuples = []

    bert_params = bert.weights
    param_values = keras.backend.batch_get_value(bert.weights)
    for ndx, (param_value, param) in enumerate(zip(param_values, bert_params)):
        stock_name = map_to_tfhub_albert_variable_name(param.name, prefix)

        if stock_name in stock_values:
            ckpt_value = stock_values[stock_name]

            if param_value.shape != ckpt_value.shape:
                trace("loader: Skipping weight:[{}] as the weight shape:[{}] is not compatible "
                      "with the checkpoint:[{}] shape:{}".format(param.name, param.shape,
                                                                 stock_name, ckpt_value.shape))
                skipped_weight_value_tuples.append((param, ckpt_value))
                continue

            weight_value_tuples.append((param, ckpt_value))
            loaded_weights.add(stock_name)
        else:
            trace("loader: No value for:[{}], i.e.:[{}] in:[{}]".format(param.name, stock_name, tfhub_model_path))
            skip_count += 1
    keras.backend.batch_set_value(weight_value_tuples)

    trace("Done loading {} BERT weights from: {} into {} (prefix:{}). "
          "Count of weights not found in the checkpoint was: [{}]. "
          "Count of weights with mismatched shape: [{}]".format(
              len(weight_value_tuples), tfhub_model_path, bert, prefix, skip_count, len(skipped_weight_value_tuples)))
    trace("Unused weights from saved model:",
          "\n\t" + "\n\t".join(sorted(set(stock_values.keys()).difference(loaded_weights))))

    return skipped_weight_value_tuples  # (bert_weight, value_from_ckpt)
