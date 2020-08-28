from tensorflow.keras.backend import square, mean


warmup_steps = 50

def loss_mse_warmup(y_true, y_pred):
    """
    Calculate the Mean Squared Error between y_true and y_pred,
    but ignore the beginning "warmup" part of the sequences.

    y_true is the desired output.
    y_pred is the model's output.
    """

    # The shape of both input tensors are:
    # [batch_size, sequence_length, num_y_signals].

    # Ignore the "warmup" parts of the sequences
    # by taking slices of the tensors.
    #print(y_true.shape, "True")
    #print(y_pred.shape, "Pred")

    y_true_slice = y_true[:, warmup_steps:, :]
    y_pred_slice = y_pred[:, warmup_steps:, :]

    #print(y_true_slice.shape, "True slice")
    #print(y_pred_slice.shape, "Pred slice")
    #print(y_true_slice)
    # These sliced tensors both have this shape:
    # [batch_size, sequence_length - warmup_steps, num_y_signals]

    # Calculate the Mean Squared Error and use it as loss.
    mse = mean(square(y_true_slice - y_pred_slice))

    return mse


def loss_manhattan_warmup(y_true, y_pred):
    """
    Calculate the mean Manhattan error between y_true and y_pred,
    but ignore the beginning "warmup" part of the sequences.

    y_true is the desired output.
    y_pred is the model's output.
    """

    # The shape of both input tensors are:
    # [batch_size, sequence_length, num_y_signals].

    # Ignore the "warmup" parts of the sequences
    # by taking slices of the tensors.
    #print(y_true.shape, "True")
    #print(y_pred.shape, "Pred")

    y_true_slice = y_true[:, warmup_steps:, :]
    y_pred_slice = y_pred[:, warmup_steps:, :]

    #print(y_true_slice.shape, "True slice")
    #print(y_pred_slice.shape, "Pred slice")
    #print(y_true_slice)
    # These sliced tensors both have this shape:
    # [batch_size, sequence_length - warmup_steps, num_y_signals]

    # Calculate the Mean Manhattan and use it as loss.
    manhattan = mean(abs(y_true_slice - y_pred_slice))
    return manhattan

