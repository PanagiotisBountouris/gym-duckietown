


# function to form the name of the model
def form_model_name(batch_size, lr, optimizer, epochs):

    return "batch={},lr={},optimizer={},epochs={}".format(batch_size, lr, optimizer, epochs)


# auxiliary function to run training and validation
def auxiliary_fun(data_size, x_data, y_data):
    pred_loss = 0
    i = 0
    while i <= data_size - 1:

        # extract batch
        if i + batch_size <= data_size - 1:
            train_x = x_data[i: i + batch_size]
            train_y = y_data[i: i + batch_size]
        else:
            train_x = x_data[i:]
            train_y = y_data[i:]

        # train using the batch and calculate the loss
        _, c = sess.run([opt, loss], feed_dict={x: train_x, y_true: train_y})

        pred_loss += c
        i += batch_size

    # for each epoch calculate the average loss of the model
    avg_loss = pred_loss / data_size
    return avg_loss