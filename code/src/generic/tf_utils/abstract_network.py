import tensorflow as tf
import os
import json

class AbstractNetwork(object):
    def __init__(self, scope_name, device=''):
        self.scope_name = scope_name
        self.device = device

    def get_parameters(self, finetune=None):
        return [v for v in tf.trainable_variables() if self.scope_name in v.name]

    def get_sources(self, sess):
        return [os.path.basename(tensor.name) for tensor in self.get_inputs(sess) if self.scope_name in tensor.name]

    def get_inputs(self, sess):
        placeholders = [p for p in sess.graph.get_operations() if "holder" in p.type if self.scope_name in p.name]
        if self.device is not '':
            return [p for p in placeholders if p.device[-1] == str(self.device)]  # use for multi-gpu computation
        return placeholders

    def get_loss(self):
        pass

    def get_accuracy(self):
        pass

# Should be in a different package as it depends on CBN (let's keep things simple!)
class ResnetModel(AbstractNetwork):
    def __init__(self, scope_name, device=''):
        super(ResnetModel, self).__init__(scope_name, device)

    def get_parameters(self, finetune=list()):
        """
        :param finetune: enable to finetune some resnet parameters
        :return: network trainable parameters (+cbn) by excluding resnet trainable parameters
        """

        params = super(ResnetModel, self).get_parameters()
        params = [v for v in params if (not 'resnet' in v.name or 'cbn_input' in v.name)]

        if len(finetune) > 0:
            for e in finetune:
                fine_tuned_params = [v for v in tf.trainable_variables() if e in v.name and v not in params]
                params += fine_tuned_params

        return params

    def get_resnet_parameters(self):

        # There is no clean way to only returns the ResNet parameters by only using the ResNet variable names...
        return [v for v in tf.global_variables() if self.scope_name in v.name and
                ('resnet' in v.name and
                 'cbn_input' not in v.name and
                 'Adam' not in v.name and
                 "local_step" not in v.name and
                 "moving_mean/biased" not in v.name)]
