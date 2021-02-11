class ParameterSuffix(object):
    def __init__(self, sep_char="^"):
        self.sep_char = sep_char

    def __call__(self, trigger_idx):
        return "{}({})".format(self.sep_char, trigger_idx + 1)
