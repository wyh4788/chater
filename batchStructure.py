class Batch:
    """
    This class provides the most basic data structure for training:

    """

    def __init__(self):
        self.msg = 'here'
        self.inputMsgIDs = []
        self.inputMsgLength = []
        self.outputResponseIDs = []
        self.outputResponseLength = []