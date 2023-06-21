from options.test_options import TestOptions

class PredictOptions(TestOptions):
    def initialize(self):
        TestOptions.initialize(self)
        self.parser.add_argument('--confidence',type=float,default="0.5",help="confidence level")
        self.is_train = False