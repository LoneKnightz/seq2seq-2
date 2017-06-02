import tensorflow as tf
import os
import pickle
import re
import time
import sys
import math
import shutil
from translate import utils, evaluation
from translate.seq2seq_model import Seq2SeqModel


class TranslationModel:
    def __init__(self, encoders, decoder, checkpoint_dir, learning_rate, learning_rate_decay_factor,
                 batch_size, keep_best=1, max_input_len=None, max_output_len=None, dev_prefix=None,
                 score_function='corpus_scores', name=None, **kwargs):

        self.batch_size = batch_size
        self.src_ext = [encoder.get('ext') or encoder.name for encoder in encoders]
        self.trg_ext = decoder.get('ext') or decoder.name
        self.pred_edits = decoder.pred_edits
        self.dev_prefix = dev_prefix
        self.name = name

        self.extensions = self.src_ext + [self.trg_ext]

        self.max_input_len = max_input_len
        self.max_output_len = max_output_len

        encoders_and_decoder = encoders + [decoder]
        self.character_level = [encoder_or_decoder.character_level for encoder_or_decoder in encoders_and_decoder]
        self.pred_characters = self.character_level[-1]
        if self.pred_edits:
            self.character_level[-1] = False

        self.learning_rate = tf.Variable(learning_rate, trainable=False, name='learning_rate', dtype=tf.float32)
        self.learning_rate_decay_op = self.learning_rate.assign(self.learning_rate * learning_rate_decay_factor)

        with tf.device('/cpu:0'):
            self.global_step = tf.Variable(0, trainable=False, name='global_step')

        self.filenames = utils.get_filenames(extensions=self.extensions, dev_prefix=dev_prefix, name=name,
                                             **kwargs)
        # TODO: check that filenames exist
        utils.debug('reading vocabularies')
        self.read_vocab()

        for encoder_or_decoder, vocab in zip(encoders + [decoder], self.vocabs):
            encoder_or_decoder.vocab_size = len(vocab.reverse)

        utils.debug('creating model')
        self.seq2seq_model = Seq2SeqModel(encoders, decoder, self.learning_rate, self.global_step, name=name,
                                          max_input_len=max_input_len, max_output_len=max_output_len, **kwargs)

        self.batch_iterator = None
        self.dev_batches = None
        self.train_size = None
        self.saver = None
        self.keep_best = keep_best
        self.checkpoint_dir = checkpoint_dir

        self.training = utils.AttrDict()  # used to keep track of training

        try:
            self.reversed_scores = getattr(evaluation, score_function).reversed  # the lower the better
        except AttributeError:
            self.reversed_scores = False  # the higher the better

    def read_data(self, max_train_size, max_dev_size, read_ahead=10, batch_mode='standard', shuffle=True,
                  **kwargs):
        utils.debug('reading training data')
        max_len = [self.max_input_len for _ in self.src_ext] + [self.max_output_len]
        train_set = utils.read_dataset(self.filenames.train, self.extensions, self.vocabs,
                                       max_size=max_train_size, character_level=self.character_level,
                                       max_seq_len=max_len)
        self.train_size = len(train_set)
        self.batch_iterator = utils.read_ahead_batch_iterator(train_set, self.batch_size, read_ahead=read_ahead,
                                                              mode=batch_mode, shuffle=shuffle)

        utils.debug('reading development data')

        dev_sets = [
            utils.read_dataset(dev, self.extensions, self.vocabs, max_size=max_dev_size,
                               character_level=self.character_level)
            for dev in self.filenames.dev
        ]
        # subset of the dev set whose perplexity is periodically evaluated
        self.dev_batches = [utils.get_batches(dev_set, batch_size=self.batch_size) for dev_set in dev_sets]

    def read_vocab(self):
        # don't try reading vocabulary for encoders that take pre-computed features
        self.vocabs = [
            utils.initialize_vocabulary(vocab_path)
            for ext, vocab_path in zip(self.extensions, self.filenames.vocab)
        ]
        *self.src_vocab, self.trg_vocab = self.vocabs

    def eval_step(self, sess):
        # compute perplexity on dev set
        for prefix, dev_batches in zip(self.dev_prefix, self.dev_batches):
            eval_loss = sum(
                self.seq2seq_model.step(sess, batch, update_model=False).loss * len(batch)
                for batch in dev_batches
            )
            eval_loss /= sum(map(len, dev_batches))

            utils.log("  {} eval: loss {:.2f}".format(prefix, eval_loss))

    def decode_sentence(self, sess, sentence_tuple, beam_size=1, remove_unk=False, early_stopping=True):
        return next(self.decode_batch(sess, [sentence_tuple], beam_size, remove_unk, early_stopping))

    def decode_batch(self, sess, sentence_tuples, batch_size, beam_size=1, remove_unk=False, early_stopping=True):
        beam_search = beam_size > 1 or isinstance(sess, list)

        if beam_search:
            batch_size = 1

        if batch_size == 1:
            batches = ([sentence_tuple] for sentence_tuple in sentence_tuples)   # lazy
        else:
            batch_count = int(math.ceil(len(sentence_tuples) / batch_size))
            batches = [sentence_tuples[i * batch_size:(i + 1) * batch_size] for i in range(batch_count)]

        def map_to_ids(sentence_tuple):
            token_ids = [
                utils.sentence_to_token_ids(sentence, vocab.vocab, character_level=char_level)
                if vocab is not None else sentence  # when `sentence` is not a sentence but a vector...
                for vocab, sentence, char_level in zip(self.vocabs, sentence_tuple, self.character_level)
            ]
            return token_ids

        for batch in batches:
            token_ids = list(map(map_to_ids, batch))

            if beam_search:
                hypotheses, _ = self.seq2seq_model.beam_search_decoding(sess, token_ids[0], beam_size,
                                                                        early_stopping=early_stopping,
                                                                        vocabs=self.vocabs)
                batch_token_ids = [hypotheses[0]]  # first hypothesis is the highest scoring one
            else:
               batch_token_ids = self.seq2seq_model.greedy_decoding(sess, token_ids)


            for src_tokens, trg_token_ids in zip(batch, batch_token_ids):
                trg_token_ids = list(trg_token_ids)
                if utils.EOS_ID in trg_token_ids:
                    trg_token_ids = trg_token_ids[:trg_token_ids.index(utils.EOS_ID)]

                trg_tokens = [self.trg_vocab.reverse[i] if i < len(self.trg_vocab.reverse) else utils._UNK
                              for i in trg_token_ids]

                raw = ' '.join(trg_tokens)

                if remove_unk:
                    trg_tokens = [token for token in trg_tokens if token != utils._UNK]

                if self.pred_edits:
                    if self.pred_characters:
                        trg_tokens = utils.reverse_edits(list(src_tokens[0].strip()), trg_tokens)
                    else:
                        trg_tokens = utils.reverse_edits(src_tokens[0].split(), trg_tokens)

                if self.pred_characters:
                    yield ''.join(trg_tokens).replace('<SPACE>', ' '), raw
                else:
                    yield ' '.join(trg_tokens).replace('@@ ', ''), raw  # merge subword units

    def align(self, sess, output=None, align_encoder_id=0, **kwargs):
        if len(self.filenames.test) != len(self.extensions):
            raise Exception('wrong number of input files')

        for line_id, lines in enumerate(utils.read_lines(self.filenames.test, self.extensions)):
            token_ids = [
                utils.sentence_to_token_ids(sentence, vocab.vocab, character_level=char_level)
                if vocab is not None else sentence
                for vocab, sentence, char_level in zip(self.vocabs, lines, self.character_level)
            ]

            _, weights = self.seq2seq_model.step(sess, data=[token_ids], forward_only=True, align=True,
                                                 update_model=False)

            trg_tokens = [self.trg_vocab.reverse[i] if i < len(self.trg_vocab.reverse) else utils._UNK
                          for i in token_ids[-1]]

            # if self.pred_edits:
            #     src_tokens = lines[0].split()
            #     new_trg_tokens = []
            #     for trg_token in trg_tokens:
            #         if len(src_tokens) > 0 and trg_token == utils._KEEP or trg_token == utils._DEL:
            #             src_token = src_tokens.pop(0)
            #             trg_token = '{} {}'.format(src_token, trg_token)
            #         else:
            #             trg_token = '{} {}'.format(trg_token, utils._INS)
            #
            #         new_trg_tokens.append(trg_token)
            #     trg_tokens = new_trg_tokens

            weights = weights.squeeze()
            max_len = weights.shape[1]

            utils.debug(weights)

            trg_tokens.append(utils._EOS)
            src_tokens = lines[align_encoder_id].split()[:max_len - 1] + [utils._EOS]

            output_file = '{}.{}.svg'.format(output, line_id + 1) if output is not None else None

            utils.heatmap(src_tokens, trg_tokens, weights, output_file=output_file)

    def decode(self, sess, beam_size, output=None, remove_unk=False, early_stopping=True, raw_output=False, **kwargs):
        utils.log('starting decoding')

        # empty `test` means that we read from standard input, which is not possible with multiple encoders
        assert len(self.src_ext) == 1 or self.filenames.test
        # check that there is the right number of files for decoding
        assert not self.filenames.test or len(self.filenames.test) == len(self.src_ext)

        output_file = None
        try:
            output_file = sys.stdout if output is None else open(output, 'w')

            lines = utils.read_lines(self.filenames.test, self.src_ext)

            if self.filenames.test is None:   # interactive mode
                batch_size = 1
            else:
                batch_size = self.batch_size
                lines = list(lines)

            hypothesis_iter = self.decode_batch(sess, lines, batch_size, beam_size=beam_size,
                                                early_stopping=early_stopping, remove_unk=remove_unk)

            for hypothesis, raw in hypothesis_iter:
                if raw_output:
                    hypothesis = raw

                output_file.write(hypothesis + '\n')
                output_file.flush()
        finally:
            if output_file is not None:
                output_file.close()

    def evaluate(self, sess, beam_size, score_function, on_dev=True, output=None, remove_unk=False, max_dev_size=None,
                 script_dir='scripts', early_stopping=True, raw_output=False, **kwargs):
        """
        :param score_function: name of the scoring function used to score and rank models
          (typically 'bleu_score')
        :param on_dev: if True, evaluate the dev corpus, otherwise evaluate the test corpus
        :param output: save the hypotheses to this file
        :param remove_unk: remove the UNK symbols from the output
        :param max_dev_size: maximum number of lines to read from dev files
        :param script_dir: parameter of scoring functions
        :return: scores of each corpus to evaluate
        """
        utils.log('starting decoding')
        assert on_dev or len(self.filenames.test) == len(self.extensions)

        filenames = self.filenames.dev if on_dev else [self.filenames.test]

        # convert `output` into a list, for zip
        if isinstance(output, str):
            output = [output]
        elif output is None:
            output = [None] * len(filenames)

        scores = []

        for filenames_, output_, prefix in zip(filenames, output, self.dev_prefix):  # evaluation on multiple corpora
            lines = list(utils.read_lines(filenames_, self.extensions))
            if on_dev and max_dev_size:
                lines = lines[:max_dev_size]

            hypotheses = []
            references = []

            output_file = None

            try:
                if output_ is not None:
                    output_file = open(output_, 'w')

                *src_sentences, trg_sentences = zip(*lines)
                src_sentences = list(zip(*src_sentences))

                hypothesis_iter = self.decode_batch(sess, lines, self.batch_size, beam_size=beam_size,
                                                    early_stopping=early_stopping, remove_unk=remove_unk)
                for i, (sources, hypothesis, reference) in enumerate(zip(src_sentences, hypothesis_iter,
                                                                         trg_sentences)):
                    hypothesis, raw = hypothesis

                    if self.pred_edits:
                        if self.pred_characters:
                            reference = utils.reverse_edits(list(sources[0].strip()), reference.split())
                            reference = ''.join(reference).replace('<SPACE>', ' ')
                        else:
                            reference = utils.reverse_edits(sources[0].split(), reference.split())
                            reference = ' '.join(reference)

                    hypotheses.append(hypothesis)
                    references.append(reference.strip().replace('@@ ', ''))

                    if output_file is not None:
                        if raw_output:
                            hypothesis = raw

                        output_file.write(hypothesis + '\n')
                        output_file.flush()

            finally:
                if output_file is not None:
                    output_file.close()

            # default scoring function is utils.bleu_score
            score, score_summary = getattr(evaluation, score_function)(hypotheses, references, script_dir=script_dir)

            # print scoring information
            score_info = [prefix, 'score={:.2f}'.format(score)]

            if score_summary:
                score_info.append(score_summary)

            if self.name is not None:
                score_info.insert(0, self.name)

            utils.log(' '.join(map(str, score_info)))
            scores.append(score)

        return scores

    def train(self, sess, **kwargs):
        self.init_training(sess=sess, **kwargs)

        utils.log('starting training')
        while True:
            try:
                self.train_step(sess=sess, **kwargs)
            except utils.CheckpointException:
                self.save(sess)
                step, score = self.training.scores[-1]
                self.manage_best_checkpoints(step, score)

    def init_training(self, sess, sgd_after_n_epoch=None, **kwargs):
        self.read_data(**kwargs)

        global_step = self.global_step.eval(sess)
        epoch = self.batch_size * global_step // self.train_size
        if sgd_after_n_epoch is not None and epoch >= sgd_after_n_epoch:  # already switched to SGD
            self.training.use_sgd = True
        else:
            self.training.use_sgd = False

        if kwargs.get('batch_mode') != 'random' and not kwargs.get('shuffle'):
            # read all the data up to this step (only if the batch iteration method is deterministic)
            for _ in range(global_step):
                next(self.batch_iterator)

        # those parameters are used to track the progress of training
        self.training.time = 0
        self.training.steps = 0
        self.training.loss = 0
        self.training.losses = []
        self.training.last_decay = global_step
        self.training.scores = []

    def train_step(self, sess, beam_size, steps_per_checkpoint, model_dir, steps_per_eval=None, eval_output=None,
                   max_steps=0, max_epochs=0, eval_burn_in=0, decay_if_no_progress=None, decay_after_n_epoch=None,
                   decay_every_n_epoch=None, sgd_after_n_epoch=None, sgd_learning_rate=None, **kwargs):
        start_time = time.time()
        res = self.seq2seq_model.step(sess, next(self.batch_iterator), update_model=True,
                                      use_sgd=self.training.use_sgd)
        self.training.loss += res.loss

        self.training.time += time.time() - start_time
        self.training.steps += 1
        global_step = self.global_step.eval(sess)

        epoch = self.batch_size * global_step // self.train_size

        if decay_after_n_epoch is not None and epoch >= decay_after_n_epoch:
            if decay_every_n_epoch is not None and (self.batch_size * (global_step - self.training.last_decay)
                                                    >= decay_every_n_epoch * self.train_size):
                sess.run(self.learning_rate_decay_op)
                utils.debug('  decaying learning rate to: {:.4f}'.format(self.learning_rate.eval()))
                self.training.last_decay = global_step

        if sgd_after_n_epoch is not None and epoch >= sgd_after_n_epoch:
            if not self.training.use_sgd:
                utils.debug('epoch {}, starting to use SGD'.format(epoch + 1))
                self.training.use_sgd = True
                if sgd_learning_rate is not None:
                    sess.run(self.learning_rate.assign(sgd_learning_rate))
                self.training.last_decay = global_step  # reset learning rate decay

        if steps_per_checkpoint and global_step % steps_per_checkpoint == 0:
            loss = self.training.loss / self.training.steps
            step_time = self.training.time / self.training.steps

            summary = 'step {} epoch {} learning rate {:.4f} step-time {:.4f} loss {:.4f}'.format(
                global_step, epoch + 1, self.learning_rate.eval(), step_time, loss)
            if self.name is not None:
                summary = '{} {}'.format(self.name, summary)
            utils.log(summary)

            if decay_if_no_progress and len(self.training.losses) >= decay_if_no_progress:
                if loss >= max(self.training.losses[:decay_if_no_progress]):
                    sess.run(self.learning_rate_decay_op)

            self.training.losses.append(loss)
            self.training.loss, self.training.time, self.training.steps = 0, 0, 0
            self.eval_step(sess)

        if steps_per_eval and global_step % steps_per_eval == 0 and 0 <= eval_burn_in <= global_step:

            eval_dir = 'eval' if self.name is None else 'eval_{}'.format(self.name)
            eval_output = os.path.join(model_dir, eval_dir)

            os.makedirs(eval_output, exist_ok=True)

            # if there are several dev files, we define several output files
            output = [
                os.path.join(eval_output, '{}.{}.out'.format(prefix, global_step))
                for prefix in self.dev_prefix
            ]

            kwargs_ = dict(kwargs)
            kwargs_['output'] = output
            score, *_ = self.evaluate(sess, beam_size, on_dev=True, **kwargs_)
            self.training.scores.append((global_step, score))

        if 0 < max_steps <= global_step or 0 < max_epochs <= epoch:
            raise utils.FinishedTrainingException
        elif steps_per_checkpoint and global_step % steps_per_checkpoint == 0:
            raise utils.CheckpointException

    def manage_best_checkpoints(self, step, score):
        score_filename = os.path.join(self.checkpoint_dir, 'scores.txt')
        # try loading previous scores
        try:
            with open(score_filename) as f:
                # list of pairs (score, step)
                scores = [(float(line.split()[0]), int(line.split()[1])) for line in f]
        except IOError:
            scores = []

        if any(step_ >= step for _, step_ in scores):
            utils.warn('inconsistent scores.txt file')

        best_scores = sorted(scores, reverse=not self.reversed_scores)[:self.keep_best]

        def full_path(filename):
            return os.path.join(self.checkpoint_dir, filename)

        lower = (lambda x, y: y < x) if self.reversed_scores else (lambda x, y: x < y)

        if any(lower(score_, score) for score_, _ in best_scores) or not best_scores:
            # if this checkpoint is in the top, save it under a special name

            prefix = 'translate-{}.'.format(step)
            dest_prefix = 'best-{}.'.format(step)

            for filename in os.listdir(self.checkpoint_dir):
                if filename.startswith(prefix):
                    dest_filename = filename.replace(prefix, dest_prefix)
                    shutil.copy(full_path(filename), full_path(dest_filename))

                    # also copy to `best` if this checkpoint is the absolute best
                    if all(lower(score_, score) for score_, _ in best_scores):
                        dest_filename = filename.replace(prefix, 'best.')
                        shutil.copy(full_path(filename), full_path(dest_filename))

            best_scores = sorted(best_scores + [(score, step)], reverse=not self.reversed_scores)

            for _, step_ in best_scores[self.keep_best:]:
                # remove checkpoints that are not in the top anymore
                prefix = 'best-{}'.format(step_)
                for filename in os.listdir(self.checkpoint_dir):
                    if filename.startswith(prefix):
                        os.remove(full_path(filename))

        # save scores
        scores.append((score, step))

        with open(score_filename, 'w') as f:
            for score_, step_ in scores:
                f.write('{:.2f} {}\n'.format(score_, step_))

    def initialize(self, sess, load=None, reset=False, reset_learning_rate=False, max_to_keep=1,
                   keep_every_n_hours=0, **kwargs):
        if keep_every_n_hours <= 0 or keep_every_n_hours is None:
            keep_every_n_hours = float('inf')

        self.saver = tf.train.Saver(max_to_keep=max_to_keep, keep_checkpoint_every_n_hours=keep_every_n_hours,
                                    sharded=False)

        sess.run(tf.global_variables_initializer())
        blacklist = ['dropout_keep_prob']

        if reset_learning_rate or reset:
            blacklist.append('learning_rate')
        if reset:
            blacklist.append('global_step')

        if load:  # load partial checkpoints
            for checkpoint in load:  # checkpoint files to load
                load_checkpoint(sess, None, checkpoint, blacklist=blacklist)
        elif not reset:
            load_checkpoint(sess, self.checkpoint_dir, blacklist=blacklist)

    def save(self, sess):
        save_checkpoint(sess, self.saver, self.checkpoint_dir, self.global_step)


variable_mapping = {   # for backward compatibility with old models
    r'encoder_(.*?)/forward_1/initial_state': r'encoder_\1/initial_state_fw',
    r'encoder_(.*?)/backward_1/initial_state': r'encoder_\1/initial_state_bw',
    r'encoder_(.*?)/forward_1/basic_lstm_cell': r'encoder_\1/stack_bidirectional_rnn/cell_0/bidirectional_rnn/fw/layer_norm_basic_lstm_cell',
    r'encoder_(.*?)/backward_1/basic_lstm_cell': r'encoder_\1/stack_bidirectional_rnn/cell_0/bidirectional_rnn/bw/layer_norm_basic_lstm_cell',
    r'decoder_(.*?)/basic_lstm_cell': r'decoder_\1/layer_norm_basic_lstm_cell',
    r'Matrix': r'kernel',
    r'Bias': r'bias',
    r'map_attns/Matrix': r'map_attns/matrix'
}


def load_checkpoint(sess, checkpoint_dir, filename=None, blacklist=()):
    """
    if `filename` is None, we load last checkpoint, otherwise
      we ignore `checkpoint_dir` and load the given checkpoint file.
    """
    if filename is None:
        # load last checkpoint
        ckpt = tf.train.get_checkpoint_state(checkpoint_dir)
        if ckpt is not None:
            filename = ckpt.model_checkpoint_path
    else:
        checkpoint_dir = os.path.dirname(filename)

    var_file = os.path.join(checkpoint_dir, 'vars.pkl')

    def get_variable_by_name(name):
        for var in tf.global_variables():
            if var.name == name:
                return var
        return None

    if os.path.exists(var_file):
        with open(var_file, 'rb') as f:
            var_names = pickle.load(f)

        variables = {}

        if variable_mapping is not None:
            for var_name in var_names:
                new_var_name = var_name

                for key, value in variable_mapping.items():
                    new_var_name = re.sub(key, value, new_var_name)

                variables[var_name] = get_variable_by_name(new_var_name)
    else:
        variables = {var.name: var for var in tf.global_variables()}

    # remove variables from blacklist
    # variables = [var for var in variables if not any(prefix in var.name for prefix in blacklist)]
    variables = {name[:-2]: var for name, var in variables.items() if not any(prefix in name for prefix in blacklist)}

    if filename is not None:
        utils.log('reading model parameters from {}'.format(filename))
        tf.train.Saver(variables).restore(sess, filename)

        utils.debug('retrieved parameters ({})'.format(len(variables)))
        for name, var in variables.items():
            utils.debug('  {} {}'.format(name, var.get_shape()))


def save_checkpoint(sess, saver, checkpoint_dir, step=None, name=None):
    var_file = os.path.join(checkpoint_dir, 'vars.pkl')
    name = name or 'translate'
    os.makedirs(checkpoint_dir, exist_ok=True)

    with open(var_file, 'wb') as f:
        var_names = [var.name for var in tf.global_variables()]
        pickle.dump(var_names, f)

    utils.log('saving model to {}'.format(checkpoint_dir))
    checkpoint_path = os.path.join(checkpoint_dir, name)
    saver.save(sess, checkpoint_path, step, write_meta_graph=False)

    utils.log('finished saving model')
