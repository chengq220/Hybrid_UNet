from options.test_options import TestOptions
from data import DataLoader
from models import create_model
from utils.writer import Writer


def run_test(epoch=-1):
    opt = TestOptions().parse()
    opt.serial_batches = True  # no shuffle
    dataset = DataLoader(opt)
    model = create_model(opt)
    accuracy = 0
    length = len(dataset)
    for i, data in enumerate(dataset):
        model.set_input(data)
        accuracy += model.test()
    return (accuracy/length)


if __name__ == '__main__':
    run_test()