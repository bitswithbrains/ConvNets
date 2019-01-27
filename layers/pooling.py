import numpy as np

class Pool(object):

    def __init__(self, f, stride, mode="max"):
        """
        :param f: filter size
        :param stride: stride for the sliding window
        :param mode: pooling method to use, "max" for maximum pooling, "avg" for average pooling
        """
        self.f = f
        self.stride = stride
        self.mode = mode
        self.cache = None

    def forward(self, A_prev):
        """
        Implements the forward pass of the pooling layer

        Arguments:
        A_prev -- Input data, numpy array of shape (m, n_H_prev, n_W_prev, n_C_prev)

        Returns:
        A -- output of the pool layer, a numpy array of shape (m, n_H, n_W, n_C)
        cache -- cache used in the backward pass of the pooling layer, contains the input A_prev
        """

        # Retrieve dimensions from the input shape
        (m, n_H_prev, n_W_prev, n_C_prev) = A_prev.shape

        # Define the dimensions of the output
        n_H = int(1 + (n_H_prev - self.f) / self.stride)
        n_W = int(1 + (n_W_prev - self.f) / self.stride)
        n_C = n_C_prev

        # Initialize output matrix A
        A = np.zeros((m, n_H, n_W, n_C))

        for i in range(m):  # loop over the training examples
            for h in range(n_H):  # loop on the vertical axis of the input volume
                for w in range(n_W):  # loop on the horizontal axis of the input volume
                    for c in range(n_C):  # loop over the channels of the input volume

                        # Find the corners of the current "slice"
                        vert_start = h * self.stride
                        vert_end = vert_start + self.f
                        horiz_start = w * self.stride
                        horiz_end = horiz_start + self.f

                        # Use the corners to define the current slice on the ith training example of A_prev, channel c.
                        a_prev_slice = A_prev[i, vert_start:vert_end, horiz_start:horiz_end, c]

                        # Compute the pooling operation on the slice.
                        if self.mode == "max":
                            A[i, h, w, c] = np.max(a_prev_slice)
                        elif self.mode == "avg":
                            A[i, h, w, c] = np.average(a_prev_slice)

        # Store the input used in cache for backward pass
        self.cache = A_prev

        # Making sure the output shape is correct
        assert (A.shape == (m, n_H, n_W, n_C))

        return A

    def backward(self, dA):
        """
        Implements the backward pass of the pooling layer

        Arguments:
        dA -- gradient of cost with respect to the output of the pooling layer, same shape as A in forward pass

        Returns:
        dA_prev -- gradient of cost with respect to the input of the pooling layer, same shape as A_prev in forward pass
        """

        # Retrieve the activation used in the forward pass from the cache
        A_prev = self.cache

        # Retrieve dimensions from A_prev's shape and dA's shape
        m, n_H_prev, n_W_prev, n_C_prev = A_prev.shape
        m, n_H, n_W, n_C = dA.shape

        # Gradient matrix dA_prev has same shape as A_prev
        dA_prev = np.zeros([m, n_H_prev, n_W_prev, n_C_prev])

        for i in range(m):  # loop over the training examples

            # select training example from A_prev
            a_prev = A_prev[i]

            for h in range(n_H):  # loop on the vertical axis
                for w in range(n_W):  # loop on the horizontal axis
                    for c in range(n_C):  # loop over the channels (depth)

                        # Find the corners of the current "slice"
                        vert_start = h * self.stride
                        vert_end = vert_start + self.f
                        horiz_start = w * self.stride
                        horiz_end = horiz_start + self.f

                        # Compute the backward propagation gradient update depending on modes.
                        if self.mode == "max":

                            # Use the corners and "c" to define the current slice from a_prev
                            a_prev_slice = a_prev[vert_start:vert_end, horiz_start:horiz_end, c]

                            # Create the mask from a_prev_slice which holds a 1 for only the maximum
                            # value. Only this value has contributed to the error
                            mask = self.__create_mask_from_window(a_prev_slice)

                            # Error (dA) is applied fully to the max value
                            grad_update = dA[i, h, w, c] * mask

                        elif self.mode == "avg":

                            # Get the value a from dA
                            da = dA[i, h, w, c]

                            # Error (dA) is contributed equally to every position in the filter
                            grad_update = self.__distribute_value(da, [self.f, self.f])

                        # Apply gradient update for current training sample to input slice
                        dA_prev[i, vert_start:vert_end, horiz_start:horiz_end, c] += grad_update

        # Make sure the output shape is correct
        assert (dA_prev.shape == A_prev.shape)

        return dA_prev

    def __create_mask_from_window(self, x):
        """
        Creates a mask from an input matrix x, to identify the max entry of x.

        Arguments:
        x -- Array of shape (f, f)

        Returns:
        mask -- Array of the same shape as window, contains a True at the position corresponding to the max entry of x.
        """

        mask = (x == np.max(x))

        return mask

    def __distribute_value(self, val, shape):
        """
        Distributes the input value in the matrix of dimension shape

        Arguments:
        val -- value to distribute, scalar
        shape -- the shape (n_H, n_W) of the output matrix for which we want to distribute the value

        Returns:
        a -- Array of size (n_H, n_W) containing average value for each cell
        """

        # Retrieve dimensions from shape
        (n_H, n_W) = shape

        # Compute the value to distribute on the matrix
        average = val / (n_H * n_W)

        # Create a matrix where every entry is the "average" value (≈1 line)
        a = np.full(shape,average)

        return a
