import numpy as np

class Conv2D(object):

    def __init__(self, stride, padding, W, b):
        """
        :param stride: the number of cells to move the convolution window - integer
        :param padding: the amount of padding to apply to the input volume - integer
        :param W: Weights, numpy array of shape (f, f, n_C_prev, n_C)
        :param b:  Biases, numpy array of shape (1, 1, 1, n_C)
        """
        self.W = W
        self.b = b
        self.stride = int(stride)
        self.pad = int(padding)
        self.cache = None

    def forward(self, A_prev):
        """
        Implements the forward propagation for a convolution function

        Arguments:
        A_prev -- output activations of the previous layer, numpy array of shape (m, n_H_prev, n_W_prev, n_C_prev)

        Returns:
        Z -- conv output, numpy array of shape (m, n_H, n_W, n_C)

        Side effects:
        cache -- cache of values needed for the backward propagation is updated with A_prev
        """

        # Retrieve dimensions from A_prev's shape
        (m, n_H_prev, n_W_prev, n_C_prev) = A_prev.shape

        # Retrieve dimensions from W's shape
        (f, f, n_C_prev, n_C) = self.W.shape

        # Compute the dimensions of the output volume.
        n_H = int(((n_H_prev + 2 * self.pad - f) / self.stride)) + 1
        n_W = int(((n_W_prev + 2 * self.pad - f) / self.stride)) + 1

        # Initialize the output volume Z with zeros.
        Z = np.zeros([m, n_H, n_W, n_C])

        # Create A_prev_pad by padding A_prev
        A_prev_pad = self.zero_pad(A_prev)
        for i in range(m):  # loop over the batch of training examples
            a_prev_pad = A_prev_pad[i]  # Select ith training example's padded activation
            for h in range(n_H):  # loop over vertical axis of the output volume
                for w in range(n_W, ):  # loop over horizontal axis of the output volume
                    for c in range(n_C):  # loop over channels (= #filters) of the output volume

                        # Find the corners of the current "slice"
                        vert_start = h * self.stride
                        vert_end = vert_start + f
                        horiz_start = w * self.stride
                        horiz_end = horiz_start + f
                        # Use the corners to define the (3D) slice of a_prev_pad
                        a_slice_prev = a_prev_pad[vert_start:vert_end, horiz_start:horiz_end, :]

                        # Convolve the (3D) slice with the correct filter W and bias b, to get back one output neuron.
                        conv_filter = self.W[:, :, :, c]
                        bias = self.b[:, :, :, c]
                        Z[i, h, w, c] = self.__convolve(a_slice_prev, conv_filter, bias)

        # Make sure the output shape is correct
        assert (Z.shape == (m, n_H, n_W, n_C))

        # Save information in "cache" for the back-propagation
        self.cache = (A_prev)

        return Z

    def __convolve(self, A_slice_prev, conv_filter, bias):
        """
        Implements a single convolution.

        Apply one filter (conv_filter) defined by parameters W on a single slice (A_slice_prev) of
        the output activation of the previous layer.

        Arguments:
        A_slice_prev -- slice of input data (activation) - matrix of shape (f, f, n_C_prev)
        conv_filter  -- convolutional kernel (filter) to apply - matrix of shape (f, f, n_C_prev)
        bias -- Bias parameters contained in a window - matrix of shape (1, 1, 1)

        Returns:
        z -- the resulting feature map value (a scalar)
        """

        # Element-wise product between activation A_slice_prev and W. Do not add the bias yet.
        s = conv_filter * A_slice_prev
        # Sum over all entries of the volume s.
        z = np.sum(s)
        # Add bias b to z. Cast b to a float() so that z results in a scalar value.
        z = z + float(bias)

        return z

    def backward(self, dZ):
        """
        Implements the backward propagation for a convolution function

        Arguments:
        dZ -- gradient of the cost with respect to the output (Z), numpy array of shape (m, n_H, n_W, n_C)

        State:
        cache -- activation of the previous layer used for the forward propagation

        Returns:
        dA_prev -- gradient of the cost with respect to the input of the conv layer (A_prev),
                   numpy array of shape (m, n_H_prev, n_W_prev, n_C_prev)
        """

        # Retrieve information from "cache"
        (A_prev) = self.cache

        # Retrieve dimensions from A_prev's shape
        (m, n_H_prev, n_W_prev, n_C_prev) = A_prev.shape

        # Retrieve dimensions from W's shape
        (f, f, n_C_prev, n_C) = self.W.shape

        # Retrieve dimensions from dZ's shape
        (m, n_H, n_W, n_C) = dZ.shape

        # dA_prev -- gradient of the cost with respect to the input of the conv layer (A_prev),
        dA_prev = np.zeros([m, n_H_prev, n_W_prev, n_C_prev])

        # dW -- gradient of the cost with respect to the weights of the conv layer (W)
        dW = np.zeros([f, f, n_C_prev, n_C])

        # db -- gradient of the cost with respect to the biases of the conv layer (b)
        db = np.zeros([1, 1, 1, n_C])

        # Pad A_prev and dA_prev
        A_prev_pad = self.zero_pad(A_prev)
        dA_prev_pad = self.zero_pad(dA_prev)

        for i in range(m):  # loop over the training examples

            # select ith training example from A_prev_pad and dA_prev_pad
            a_prev_pad = A_prev_pad[i]
            da_prev_pad = dA_prev_pad[i]

            for h in range(n_H):  # loop over vertical axis of the output volume
                for w in range(n_W):  # loop over horizontal axis of the output volume
                    for c in range(n_C):  # loop over the channels of the output volume

                        # Find the corners of the current "slice"
                        vert_start = h * self.stride
                        vert_end = vert_start + f
                        horiz_start = w * self.stride
                        horiz_end = horiz_start + f

                        # Use the corners to define the slice from a_prev_pad
                        a_slice = a_prev_pad[vert_start:vert_end, horiz_start:horiz_end, :]

                        # Update gradients for the window and the filter's parameters
                        da_prev_pad[vert_start:vert_end, horiz_start:horiz_end, :] += self.W[:, :, :, c] * dZ[i, h, w, c]
                        dW[:, :, :, c] += a_slice * dZ[i, h, w, c]
                        db[:, :, :, c] += dZ[i, h, w, c]

            # Set the ith training example's dA_prev to the unpadded da_prev_pad
            dA_prev[i, :, :, :] = da_prev_pad[self.pad:-self.pad, self.pad:-self.pad, :]

        # Make sure the output shape is correct
        assert (dA_prev.shape == (m, n_H_prev, n_W_prev, n_C_prev))

        return dA_prev


    def zero_pad(self, X):
        """
        Pad with zeros all images of the dataset X. The padding is applied to the height and width of an image,
        as illustrated in Figure 1.

        Argument:
        X -- python numpy array of shape (m, n_H, n_W, n_C) representing a batch of m images
        pad -- integer, amount of padding around each image on vertical and horizontal dimensions

        Returns:
        X_pad -- padded image of shape (m, n_H + 2*pad, n_W + 2*pad, n_C)
        """

        X_pad = np.pad(X, ((0, 0), (self.pad, self.pad), (self.pad, self.pad), (0, 0)), 'constant', constant_values=(0, 0))

        return X_pad