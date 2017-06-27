import tensorflow as tf


def stack_bidirectional_dynamic_rnn(cells_fw, cells_bw, inputs, initial_states_fw=None, initial_states_bw=None,
                                    dtype=None, sequence_length=None, parallel_iterations=None, scope=None,
                                    time_pooling=None, pooling_avg=None):
    states_fw = []
    states_bw = []
    prev_layer = inputs

    with tf.variable_scope(scope or "stack_bidirectional_rnn"):
        for i, (cell_fw, cell_bw) in enumerate(zip(cells_fw, cells_bw)):
            initial_state_fw = None
            initial_state_bw = None
            if initial_states_fw:
                initial_state_fw = initial_states_fw[i]
            if initial_states_bw:
                initial_state_bw = initial_states_bw[i]

            with tf.variable_scope('cell_{}'.format(i)):
                outputs, (state_fw, state_bw) = tf.nn.bidirectional_dynamic_rnn(
                    cell_fw,
                    cell_bw,
                    prev_layer,
                    initial_state_fw=initial_state_fw,
                    initial_state_bw=initial_state_bw,
                    sequence_length=sequence_length,
                    parallel_iterations=parallel_iterations,
                    dtype=dtype)
                # Concat the outputs to create the new input.
                prev_layer = tf.concat(outputs, 2)

                if time_pooling and i < len(cells_fw) - 1:
                    prev_layer, sequence_length = apply_time_pooling(prev_layer, sequence_length, time_pooling[i],
                                                                     pooling_avg)

            states_fw.append(state_fw)
            states_bw.append(state_bw)

    return prev_layer, tuple(states_fw), tuple(states_bw)


def apply_time_pooling(inputs, sequence_length, stride, pooling_avg=False):
    shape = [tf.shape(inputs)[0], tf.shape(inputs)[1], inputs.get_shape()[2].value]

    if pooling_avg:
        inputs_ = [inputs[:, i::stride, :] for i in range(stride)]

        max_len = tf.shape(inputs_[0])[1]
        for k in range(1, stride):
            len_ = tf.shape(inputs_[k])[1]
            paddings = tf.stack([[0, 0], [0, max_len - len_], [0, 0]])
            inputs_[k] = tf.pad(inputs_[k], paddings=paddings)

        inputs = tf.reduce_sum(inputs_, 0) / len(inputs_)
    else:
        inputs = inputs[:, ::stride, :]

    inputs = tf.reshape(inputs, tf.stack([shape[0], tf.shape(inputs)[1], shape[2]]))
    sequence_length = (sequence_length + stride - 1) // stride  # rounding up

    return inputs, sequence_length
