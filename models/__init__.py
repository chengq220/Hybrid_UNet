def create_model(opt):
    from .classifier import ClassifierModel
    model = ClassifierModel(opt)
    return model