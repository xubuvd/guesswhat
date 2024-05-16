from tqdm import tqdm
from numpy import float32
import numpy as np
import copy
import os
import itertools
from collections import OrderedDict
import tensorflow as tf
import logging

# TODO check if optimizers are always ops? Maybe there is a better check
def is_optimizer(x):
    return hasattr(x, 'op_def')

def is_summary(x):
    return isinstance(x, tf.Tensor) and x.dtype is tf.string

def is_float(x):
    return isinstance(x, tf.Tensor) and x.dtype is tf.float32

def is_scalar(x):
    return isinstance(x, tf.Tensor) and x.dtype is tf.float32 and len(x.shape) == 0

class Evaluator(object):
    def __init__(self, provided_sources, scope="", writer=None,
                 network=None, tokenizer=None): # debug purpose only, do not use in the code
        self.provided_sources = provided_sources
        #Sources: images, dialogues, answer_mask, dialog, num_qa_pairs, padding_mask, seq_length, cum_reward, state_c, state_h, is_training, greedy
        self.scope = scope
        self.writer = writer
        if len(scope) > 0 and not scope.endswith("/"):
            self.scope += "/"
        self.use_summary = False

        # Debug tools (should be removed on the long run)
        self.network=network
        self.tokenizer = tokenizer
        self.logger = logging.getLogger()

    def process(self, sess, iterator, outputs, listener=None):
        assert isinstance(outputs, list), "outputs must be a list"
        original_outputs = list(outputs)
        is_training = any([is_optimizer(x) for x in outputs])
        if listener is not None:
            outputs += [listener.require()]  # add require outputs
            # outputs = flatten(outputs) # flatten list (when multiple requirement)
            outputs = list(OrderedDict.fromkeys(outputs))  # remove duplicate while preserving ordering
            listener.before_epoch(is_training)
        n_iter = 1.
        aggregated_outputs = [0.0 for v in outputs if is_scalar(v) and v in original_outputs]
        #self.logger.info("aggregated_outputs:{}".format(aggregated_outputs))
        #aggregated_outputs:[0.0, 0.0]

        #self.logger.info("outputs:{}".format(outputs))
        #[<tf.Tensor 'qgen/ml_loss/truediv:0' shape=() dtype=float32>, <tf.Tensor 'qgen/ml_loss/truediv:0' shape=() dtype=float32>]

        for batch in tqdm(iterator):
            # Appending is_training flag to the feed_dict
            batch["is_training"] = is_training
            # evaluate the network on the batch
            results = self.execute(sess, outputs, batch)
            # process the results
            i = 0
            for var, result in zip(outputs, results):
                if is_scalar(var) and var in original_outputs:
                    # moving average
                    aggregated_outputs[i] = ((n_iter - 1.) / n_iter) * aggregated_outputs[i] + result / n_iter
                    i += 1
                elif is_summary(var):  # move into listener?
                    self.writer.add_summary(result)
                if listener is not None and listener.valid(var):
                    listener.after_batch(result, batch, is_training)
            n_iter += 1
        if listener is not None:
            listener.after_epoch(is_training)
        return aggregated_outputs

    def execute(self, sess, output, batch):
        #print("sess.run provided_sources:{}".format(self.provided_sources))
        feed_dict = {self.scope + key + ":0": value for key, value in batch.items() if key in self.provided_sources}
        
        #self.provided_sources:images, dialogues, answer_mask, dialog, num_qa_pairs, padding_mask, seq_length, cum_reward, state_c, state_h, is_training, greedy
        #ans = [np.shape(value) for key,value in batch.items() if key in self.provided_sources]
        #self.logger.info("sess.run feed_dict:{}".format(ans))
        #sess.run feed_dict:['qgen/dialogues:0', 'qgen/images:0', 'qgen/seq_length:0', 'qgen/padding_mask:0', 'qgen/answer_mask:0', 'qgen/is_training:0']
        #sess.run feed_dict:['qgen/dialogues:0', 'qgen/dialog:0', 'qgen/images:0', 'qgen/seq_length:0', 'qgen/padding_mask:0', 'qgen/answer_mask:0', 'qgen/num_qa_pairs:0', 'qgen/is_training:0'] 
        #sess.run feed_dict:[(32, 46),            (32, 5, 7),      (32, 1000),              (32,),         (32, 46),                (32, 46),                 (32, 5),            ()]
        return sess.run(output, feed_dict=feed_dict)

class MultiGPUEvaluator(object):
    """Wrapper for evaluating on multiple GPUOptions

    parameters
    ----------
        provided_sources: list of sources
            Each source has num_gpus placeholders with name:
            name_scope[gpu_index]/network_scope/source
        network_scope: str
            Variable scope of the model
        name_scopes: list of str
            List that defines name_scope for each GPU
    """

    def __init__(self, provided_sources, name_scopes, writer=None,
                 networks=None, tokenizer=None): #Debug purpose only, do not use here

        # Dispatch sources
        self.provided_sources = provided_sources
        self.name_scopes = name_scopes
        self.writer = writer

        self.multi_gpu_sources = []
        for source in self.provided_sources:
            for name_scope in name_scopes:
                self.multi_gpu_sources.append(os.path.join(name_scope, source))

        # Debug tools, do not use in the code!
        self.networks = networks
        self.tokenizer = tokenizer

    def process(self, sess, iterator, outputs, listener=None):

        assert listener is None, "Listener are not yet supported with multi-gpu evaluator"
        assert isinstance(outputs, list), "outputs must be a list"

        # check for optimizer to define training/eval mode
        is_training = any([is_optimizer(x) for x in outputs])

        print("process, outputs:{}".format(outputs))

        # Prepare epoch
        n_iter = 1.
        aggregated_outputs = [0.0 for v in outputs if is_scalar(v)]

        scope_to_do = list(self.name_scopes)
        multi_gpu_batch = dict()
        for batch in tqdm(iterator):

            assert len(scope_to_do) > 0

            # apply training mode
            batch['is_training'] = is_training

            # update multi-gpu batch
            name_scope = scope_to_do.pop()
            for source, v in batch.items():
                multi_gpu_batch[os.path.join(name_scope, source)] = v

            if not scope_to_do: # empty list -> multi_gpu_batch is ready!
                n_iter += 1
                # Execute the batch
                results = self.execute(sess, outputs, multi_gpu_batch)

                # reset mini-batch
                scope_to_do = list(self.name_scopes)
                multi_gpu_batch = dict()

                # process the results
                i = 0
                for var, result in zip(outputs, results):
                    if is_scalar(var) and var in outputs:
                        # moving average
                        aggregated_outputs[i] = ((n_iter - 1.) / n_iter) * aggregated_outputs[i] + result / n_iter
                        i += 1
                    elif is_summary(var):  # move into listener?
                        self.writer.add_summary(result)
                    # No listener as "results" may arrive in different orders... need to find a way to unshuffle them
        return aggregated_outputs

    def execute(self, sess, output, batch):
        ans = ""
        for key, value in batch.items():
            if key in self.multi_gpu_sources:
                if ans != "":ans += " "
                ans += "{}".format(key)
        print("execute,feed_dict:{}".format(ans))
        feed_dict = {key + ":0": value for key, value in batch.items() if key in self.multi_gpu_sources}
        return sess.run(output, feed_dict=feed_dict)

