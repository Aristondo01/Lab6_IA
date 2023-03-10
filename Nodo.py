class Nodo(object):
    def __init__(self, feature=None, threshold=None, data_left=None, data_right=None, gain=None, value=None, depth = None, aux_value = None):
        self.feature = feature
        self.threshold = threshold
        self.data_left = data_left
        self.data_right = data_right
        self.gain = gain
        self.value = value
        self.aux_value = aux_value
        self.depth = depth