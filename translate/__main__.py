import os
import sys
import logging
import argparse
import subprocess
import tensorflow as tf
import yaml
import shutil
import tarfile

from pprint import pformat
from operator import itemgetter
from translate import utils
from translate.translation_model import TranslationModel
from translate.multitask_model import MultiTaskModel

parser = argparse.ArgumentParser()
parser.add_argument('config', help='load a configuration file in the YAML format')
parser.add_argument('-v', '--verbose', help='verbose mode', action='store_true')
# use 'store_const' instead of 'store_true' so that the default value is `None` instead of `False`
parser.add_argument('--reset', help="reset model (don't load any checkpoint)", action='store_const', const=True)
parser.add_argument('--reset-learning-rate', help='reset learning rate', action='store_const', const=True)
parser.add_argument('--learning-rate', type=float, help='custom learning rate (triggers `reset-learning-rate`)')
parser.add_argument('--purge', help='remove previous model files', action='store_true')

# Available actions (exclusive)
parser.add_argument('--decode', help='translate this corpus (one filename for each encoder)', nargs='*')
parser.add_argument('--align', help='translate and show alignments by the attention mechanism', nargs='+')
parser.add_argument('--eval', help='compute BLEU score on this corpus (source files and target file)', nargs='+')
parser.add_argument('--train', help='train an NMT model', action='store_true')

# TensorFlow configuration
parser.add_argument('--gpu-id', type=int, help='index of the GPU where to run the computation')
parser.add_argument('--no-gpu', action='store_true', help='run on CPU')

# Decoding options (to avoid having to edit the config file)
parser.add_argument('--beam-size', type=int)
parser.add_argument('--ensemble', action='store_const', const=True)
parser.add_argument('--lm-file')
parser.add_argument('--load', nargs='+')
parser.add_argument('--lm-weight', type=float)
parser.add_argument('--output')
parser.add_argument('--max-steps', type=int)
parser.add_argument('--remove-unk', action='store_const', const=True)
parser.add_argument('--raw-output', action='store_const', const=True)
parser.add_argument('--model-dir')
parser.add_argument('--batch-size', type=int)

parser.add_argument('--align-encoder-id', type=int, default=0)

def main(args=None):
    args = parser.parse_args(args)

    # read config file and default config
    with open('config/default.yaml') as f:
        default_config = utils.AttrDict(yaml.safe_load(f))

    with open(args.config) as f:
        config = utils.AttrDict(yaml.safe_load(f))
        
        if args.learning_rate is not None:
            args.reset_learning_rate = True
        
        # command-line parameters have higher precedence than config file
        for k, v in vars(args).items():
            if v is not None:
                config[k] = v

        # set default values for parameters that are not defined
        for k, v in default_config.items():
            config.setdefault(k, v)

    # enforce parameter constraints
    assert config.steps_per_eval % config.steps_per_checkpoint == 0, (
        'steps-per-eval should be a multiple of steps-per-checkpoint')
    assert args.decode is not None or args.eval or args.train or args.align, (
        'you need to specify at least one action (decode, eval, align, or train)')

    if args.purge:
        utils.log('deleting previous model')
        shutil.rmtree(config.model_dir, ignore_errors=True)

    os.makedirs(config.model_dir, exist_ok=True)

    # copy config file to model directory
    config_path = os.path.join(config.model_dir, 'config.yaml')
    if not os.path.exists(config_path):
        shutil.copy(args.config, config_path)

    # also copy default config
    config_path = os.path.join(config.model_dir, 'default.yaml')
    if not os.path.exists(config_path):
        shutil.copy('config/default.yaml', config_path)

    # copy source code to model directory
    tar_path =  os.path.join(config.model_dir, 'code.tar.gz')
    if not os.path.exists(tar_path):
        with tarfile.open(tar_path, "w:gz") as tar:
            for filename in os.listdir('translate'):
                if filename.endswith('.py'):
                    tar.add(os.path.join('translate', filename), arcname=filename)

    logging_level = logging.DEBUG if args.verbose else logging.INFO
    # always log to stdout in decoding and eval modes (to avoid overwriting precious train logs)
    log_path = os.path.join(config.model_dir, config.log_file)
    logger = utils.create_logger(log_path if args.train else None)
    logger.setLevel(logging_level)

    utils.log('label: {}'.format(config.label))
    utils.log('description:\n  {}'.format('\n  '.join(config.description.strip().split('\n'))))

    utils.log(' '.join(sys.argv))  # print command line
    try:  # print git hash
        commit_hash = subprocess.check_output(['git', 'rev-parse', 'HEAD']).decode().strip()
        utils.log('commit hash {}'.format(commit_hash))
    except:
        pass

    # log parameters
    utils.debug('program arguments')
    for k, v in sorted(config.items(), key=itemgetter(0)):
        utils.debug('  {:<20} {}'.format(k, pformat(v)))

    if isinstance(config.dev_prefix, str):
        config.dev_prefix = [config.dev_prefix]

    if config.tasks is not None:
        config.tasks = [utils.AttrDict(task) for task in config.tasks]
        tasks = config.tasks
    else:
        tasks = [config]

    for task in tasks:
        for parameter, value in config.items():
            task.setdefault(parameter, value)

        task.encoders = [utils.AttrDict(encoder) for encoder in task.encoders]
        task.decoders = [utils.AttrDict(decoder) for decoder in task.decoders]

        for encoder_or_decoder in task.encoders + task.decoders:
            for parameter, value in task.items():
                encoder_or_decoder.setdefault(parameter, value)

    device = None
    if config.no_gpu:
        device = '/cpu:0'
    elif config.gpu_id is not None:
        device = '/gpu:{}'.format(config.gpu_id)

    utils.log('creating model')
    utils.log('using device: {}'.format(device))

    with tf.device(device):
        config.checkpoint_dir = os.path.join(config.model_dir, 'checkpoints')

        if config.weight_scale:
            if config.initializer == 'uniform':
                initializer = tf.random_uniform_initializer(minval=-config.weight_scale, maxval=config.weight_scale)
            else:
                initializer = tf.random_normal_initializer(stddev=config.weight_scale)
        else:
            initializer = None


        tf.get_variable_scope().set_initializer(initializer)

        config.decode_only = args.decode is not None or args.eval or args.align  # exempt from creating gradient ops

        if config.tasks is not None:
            model = MultiTaskModel(**config)
        else:
            model = TranslationModel(**config)

    # count parameters
    utils.log('model parameters ({})'.format(len(tf.global_variables())))
    parameter_count = 0
    for var in tf.global_variables():
        utils.log('  {} {}'.format(var.name, var.get_shape()))

        if not var.name.startswith('gradients'):
            v = 1
            for d in var.get_shape():
                v *= d.value
            parameter_count += v
    utils.log('number of parameters: {:.2f}M'.format(parameter_count / 1e6))

    tf_config = tf.ConfigProto(log_device_placement=False, allow_soft_placement=True)
    tf_config.gpu_options.allow_growth = config.allow_growth
    tf_config.gpu_options.per_process_gpu_memory_fraction = config.mem_fraction

    with tf.Session(config=tf_config) as sess:
        best_checkpoint = os.path.join(config.checkpoint_dir, 'best')

        if config.ensemble and (args.eval or args.decode is not None):
            # create one session for each model in the ensemble
            sess = [tf.Session() for _ in config.load]
            for sess_, checkpoint in zip(sess, config.load):
                model.initialize(sess_, [checkpoint])
        elif (not config.load and (args.eval or args.decode is not None or args.align) and
             (os.path.isfile(best_checkpoint + '.index') or os.path.isfile(best_checkpoint + '.index'))):
            # in decoding and evaluation mode, unless specified otherwise (by `load`),
            # try to load the best checkpoint)
            model.initialize(sess, [best_checkpoint])
        else:
            # loads last checkpoint, unless `reset` is true
            model.initialize(sess, **config)

        # Inspect variables:
        # tf.get_variable_scope().reuse_variables()
        # import pdb; pdb.set_trace()
        if args.decode is not None:
            model.decode(sess, **config)
        elif args.eval:
            model.evaluate(sess, on_dev=False, **config)
        elif args.align:
            model.align(sess, **config)
        elif args.train:
            try:
                model.train(sess=sess, **config)
            except (KeyboardInterrupt, utils.FinishedTrainingException):
                utils.log('exiting...')
                model.save(sess)
                sys.exit()


if __name__ == '__main__':
    main()
