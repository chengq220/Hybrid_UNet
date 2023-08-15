import time
from options.train_options import TrainOptions
from data.base_dataset import BaseDataset
from data import DataLoader
from models import create_model
from utils.writer import Writer
from test import run_test
from predict import predict
import wandb


if __name__ == '__main__':
    opt = TrainOptions().parse()
    model = create_model(opt)
    dataset = DataLoader(opt)
    dataset_size = len(dataset)
    writer = Writer(opt)
    total_steps = 0
    best_loss = 1 

    # wandb.init(project="Experiment")
    # wandb.watch(model.net, log='all')

    for epoch in range(opt.epoch_count, opt.niter + opt.niter_decay + 1):
        total_steps += 1
        train_loss = 0
        for i, data in enumerate(dataset):
            model.set_input(data)
            model.optimize_parameters()
            train_loss += model.loss
            exit()
        exit()
        model.save_network('latest')
        train_loss /= dataset_size   
        if(train_loss < best_loss):
            model.save_network('best')
        best_loss = train_loss
        test_acc = run_test(epoch)
        val_acc = predict(total_steps)

        # model.update_learning_rate()
        wandb.log({"One Test Point Accuracy":val_acc, "Dice Loss": train_loss, "Testing Accuracy": test_acc})