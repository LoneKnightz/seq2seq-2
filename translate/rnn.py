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


# def multi_bidirectional_rnn(cells, inputs, sequence_length=None, dtype=None, parallel_iterations=None,
#                             swap_memory=False, time_major=False, time_pooling=None, pooling_avg=None,
#                             residual_connections=False, trainable_initial_state=True, **kwargs):
#     if not time_major:
#         time_dim = 1
#         batch_dim = 0
#     else:
#         time_dim = 0
#         batch_dim = 1
#
#     batch_size = tf.shape(inputs)[batch_dim]
#
#     output_states_fw = []
#     output_states_bw = []
#     for i, (cell_fw, cell_bw) in enumerate(cells):
#         # forward direction
#         with tf.variable_scope('forward_{}'.format(i + 1)) as fw_scope:
#             if trainable_initial_state:
#                 initial_state = get_variable_unsafe('initial_state', initializer=tf.zeros([cell_fw.state_size]),
#                                                     dtype=dtype)
#                 initial_state = tf.reshape(tf.tile(initial_state, [batch_size]),
#                                            shape=[batch_size, cell_fw.state_size])
#             else:
#                 initial_state = None
#
#             inputs_fw, output_state_fw = rnn.dynamic_rnn(
#                 cell=cell_fw, inputs=inputs, sequence_length=sequence_length, initial_state=initial_state,
#                 dtype=dtype, parallel_iterations=parallel_iterations, swap_memory=swap_memory,
#                 time_major=time_major, scope=fw_scope
#             )
#
#         # backward direction
#         inputs_reversed = tf.reverse_sequence(
#             input=inputs, seq_lengths=sequence_length, seq_dim=time_dim, batch_dim=batch_dim
#         )
#
#         with tf.variable_scope('backward_{}'.format(i + 1)) as bw_scope:
#             if trainable_initial_state:
#                 initial_state = get_variable_unsafe('initial_state', initializer=tf.zeros([cell_bw.state_size]),
#                                                     dtype=dtype)
#                 initial_state = tf.reshape(tf.tile(initial_state, [batch_size]),
#                                            shape=[batch_size, cell_bw.state_size])
#             else:
#                 initial_state = None
#
#             inputs_bw, output_state_bw = rnn.dynamic_rnn(
#                 cell=cell_bw, inputs=inputs_reversed, sequence_length=sequence_length, initial_state=initial_state,
#                 dtype=dtype, parallel_iterations=parallel_iterations, swap_memory=swap_memory, time_major=time_major,
#                 scope=bw_scope
#             )
#
#         inputs_bw_reversed = tf.reverse_sequence(
#             input=inputs_bw, seq_lengths=sequence_length,
#             seq_dim=time_dim, batch_dim=batch_dim
#         )
#         new_inputs = tf.concat([inputs_fw, inputs_bw_reversed], 2)
#
#         if residual_connections and i < len(cells) - 1:
#             # the output's dimension is twice that of the initial input (because of bidir)
#             if i == 0:
#                 inputs = tf.tile(inputs, (1, 1, 2))  # FIXME: temporary solution
#             inputs = new_inputs + inputs
#         else:
#             inputs = new_inputs
#
#         if time_pooling and i < len(cells) - 1:
#             inputs, sequence_length = apply_time_pooling(inputs, sequence_length, time_pooling[i], pooling_avg)
#
#         output_states_fw.append(output_state_fw)
#         output_states_bw.append(output_state_bw)
#
#     return inputs, tf.concat(output_states_fw, 1), tf.concat(output_states_bw, 1)
#


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
