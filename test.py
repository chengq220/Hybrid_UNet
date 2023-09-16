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
    acc_hist = []
    for i, data in enumerate(dataset):
        model.set_input(data)
        curr_acc = model.test()
        acc_hist.append(curr_acc)
        accuracy += curr_acc

    return (accuracy/length), acc_hist


if __name__ == '__main__':
    run_test()