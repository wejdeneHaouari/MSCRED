import tensorflow as tf
from tensorflow.keras.layers import ConvLSTM2D
import utils as util
import numpy as np
import os


def cnn_encoder_layer(data, filter_layer, strides):
    """
    :param data: the input data, when it is the first layer is 5 * 30 * 30 * 3, the second layer is 30 * 30 * 32,
                 the third layer is 15 * 15 * 64, the fourth layer is 8 * 8 * 128
    :param filter_layer:
    :param strides:
    :return: the result after conv, the first layer is 30 * 30 * 32, the second layer is 15 * 15 * 64, the third layer
             is 8 * 8 * 128, the final layer is 4 * 4 * 256
    """

    result = tf.nn.conv2d(
        input=data,
        filters=filter_layer,  # Change from 'filter' to 'filters'
        strides=strides,
        padding="SAME")
    return tf.nn.selu(result)


def tensor_variable(shape, name):
    """
    Tensor variable declaration initialization
    :param shape:
    :param name:
    :return:
    """
    initializer = tf.initializers.GlorotUniform()
    variable = tf.Variable(initializer(shape=shape), name=name)
    return variable


def cnn_encoder(data):
    """
    :param data: the input data size is 5 * 30 * 30 * 3
    :return: Outputs after each convolutional layer
    """

    # the first layer,the output size is 30 * 30 * 32
    cnn1_out = tf.keras.layers.Conv2D(32, kernel_size=(3, 3), strides=(1, 1), activation='selu', padding='same', name="filter1")(data)

    # the second layer, the output size is 15 * 15 * 64
    cnn2_out = tf.keras.layers.Conv2D(64, kernel_size=(3, 3), strides=(2, 2), activation='selu', padding='same', name="filter2")(cnn1_out)

    # the third layer, the output size is 8 * 8 * 128
    cnn3_out = tf.keras.layers.Conv2D(128, kernel_size=(2, 2), strides=(2, 2), activation='selu', padding='same', name="filter3")(cnn2_out)

    # the fourth layer, the output size is 4 * 4 * 256
    cnn4_out = tf.keras.layers.Conv2D(256, kernel_size=(2, 2), strides=(2, 2), activation='selu', padding='same', name="filter4")(cnn3_out)

    return cnn1_out, cnn2_out, cnn3_out, cnn4_out


def cnn_lstm_attention_layer(input_data, layer_number):
    """
    :param input_data:
    :param layer_number:
    :return:
    """

    convlstm_layer = ConvLSTM2D(
        filters=input_data.shape[-1],  # output channels
        kernel_size=(2, 2),
        padding="same",
        return_sequences=True,
        name="conv_lstm_cell" + str(layer_number)
    )

    outputs = convlstm_layer(input_data)

    # attention based on inner-product between feature representation of last step and other steps
    attention_w = []
    for k in range(util.step_max):
        attention_w.append(tf.reduce_sum(tf.multiply(outputs[0][k], outputs[0][-1])) / util.step_max)
    attention_w = tf.reshape(tf.nn.softmax(tf.stack(attention_w)), [1, util.step_max])

    outputs = tf.reshape(outputs[0], [util.step_max, -1])
    outputs = tf.matmul(attention_w, outputs)
    outputs = tf.reshape(outputs, [1, input_data.shape[2], input_data.shape[3], input_data.shape[4]])

    return outputs, attention_w


def cnn_decoder_layer(conv_lstm_out_c, filter_weights, output_shape, strides):
    """
    :param conv_lstm_out_c:
    :param filter_weights: A 4-D tensor; should be the same as the filter used in `tf.nn.conv2d_transpose`
    :param output_shape: 1-D tensor of type int32, of 4 elements. The shape of the output from this op.
    :param strides: A list of ints. The stride of the sliding window for each dimension of the input tensor.
    :return:
    """

    deconv = tf.nn.conv2d_transpose(
        input=conv_lstm_out_c,
        filters=filter_weights,
        output_shape=output_shape,
        strides=strides,
        padding="SAME")
    deconv = tf.nn.selu(deconv)
    return deconv

def tensor_variable(shape, name):
    """
    Tensor variable declaration initialization
    :param shape:
    :param name:
    :return:
    """
    initializer = tf.keras.initializers.GlorotUniform()
    variable = tf.Variable(initializer(shape=shape), name=name)
    return variable

def cnn_decoder(lstm1_out, lstm2_out, lstm3_out, lstm4_out):
    d_filter4 = tensor_variable([2, 2, 128, 256], "d_filter4")
    dec4 = cnn_decoder_layer(lstm4_out, d_filter4, [1, 8, 8, 128], (1, 2, 2, 1))
    dec4_concat = tf.concat(values=[dec4, lstm3_out], axis=3)

    d_filter3 = tensor_variable([2, 2, 64, 256], "d_filter3")
    dec3 = cnn_decoder_layer(dec4_concat, d_filter3, [1, 15, 15, 64], (1, 2, 2, 1))
    dec3_concat = tf.concat(values=[dec3, lstm2_out], axis=3)

    d_filter2 = tensor_variable([3, 3, 32, 128], "d_filter2")
    dec2 = cnn_decoder_layer(dec3_concat, d_filter2, [1, 30, 30, 32], (1, 2, 2, 1))
    dec2_concat = tf.concat(values=[dec2, lstm1_out], axis=3)

    d_filter1 = tensor_variable([3, 3, 3, 64], "d_filter1")
    dec1 = cnn_decoder_layer(dec2_concat, d_filter1, [1, 30, 30, 3], (1, 1, 1, 1))

    return dec1


def main():
    # Read dataset from file
    matrix_data_path = util.train_data_path + "train.npy"
    matrix_gt_1 = np.load(matrix_data_path)

    # Define the model
    model_input = tf.keras.Input(shape=(util.step_max, 30, 30, 3))
    conv1_out, conv2_out, conv3_out, conv4_out = cnn_encoder(model_input)

    # Reshape outputs
    conv1_out = tf.reshape(conv1_out, [-1, 5, 30, 30, 32])
    conv2_out = tf.reshape(conv2_out, [-1, 5, 15, 15, 64])
    conv3_out = tf.reshape(conv3_out, [-1, 5, 8, 8, 128])
    conv4_out = tf.reshape(conv4_out, [-1, 5, 4, 4, 256])

    # LSTM with attention layers
    conv1_lstm_attention_out, _ = cnn_lstm_attention_layer(conv1_out, 1)
    conv2_lstm_attention_out, _ = cnn_lstm_attention_layer(conv2_out, 2)
    conv3_lstm_attention_out, _ = cnn_lstm_attention_layer(conv3_out, 3)
    conv4_lstm_attention_out, _ = cnn_lstm_attention_layer(conv4_out, 4)

    # Decoder
    deconv_out = cnn_decoder(conv1_lstm_attention_out, conv2_lstm_attention_out, conv3_lstm_attention_out,
                             conv4_lstm_attention_out)

    model = tf.keras.Model(inputs=model_input, outputs=deconv_out)

    # Define loss and optimizer
    loss_fn = tf.keras.losses.MeanSquaredError()
    optimizer = tf.keras.optimizers.Adam(learning_rate=util.learning_rate)

    # Training
    for idx in range(util.train_start_id, util.train_end_id):
        matrix_gt = matrix_gt_1[idx - util.train_start_id]
        matrix_gt_tensor = tf.convert_to_tensor(matrix_gt, dtype=tf.float32)

        with tf.GradientTape() as tape:
            # expand the dimensions of matrix_gt to include a batch dimension
            matrix_gt_tensor = tf.expand_dims(matrix_gt, axis=0)

            reconstructed = model(matrix_gt_tensor)
            loss = loss_fn(matrix_gt_tensor[-1], reconstructed)

        grads = tape.gradient(loss, model.trainable_variables)
        optimizer.apply_gradients(zip(grads, model.trainable_variables))
        print("mse of last train data:", loss.numpy())

        # Testing
        matrix_data_path = util.test_data_path + "test.npy"
        matrix_gt_test = np.load(matrix_data_path)
        result_all = []  # Initialize result_all here

        for idx in range(util.test_start_id, util.test_end_id):
            matrix_gt = matrix_gt_test[idx - util.test_start_id]
            matrix_gt_tensor = tf.convert_to_tensor(matrix_gt, dtype=tf.float32)
            # expand the dimensions of matrix_gt to include a batch dimension
            matrix_gt_tensor = tf.expand_dims(matrix_gt, axis=0)
            reconstructed = model(matrix_gt_tensor)
            result_all.append(reconstructed.numpy().squeeze())
            loss = loss_fn(matrix_gt_tensor[-1], reconstructed)
            print("mse of last test data:", loss.numpy())

        # Save reconstructed test data
        reconstructed_path = util.reconstructed_data_path
        if not os.path.exists(reconstructed_path):
            os.makedirs(reconstructed_path)
        reconstructed_path = os.path.join(reconstructed_path, "test_reconstructed.npy")

        # Convert result_all list to numpy array and save to disk
        result_all = np.array(result_all).reshape((-1, 30, 30, 3))
        np.save(reconstructed_path, result_all)

        # Save the model and results
        model.save_weights(os.path.join(util.reconstructed_data_path, 'model_weights.h5'))


if __name__ == '__main__':
    main()
