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

    wandb.init(project="debug")
    wandb.watch(model.net, log='all')

    data = next(iter(dataset))
    for epoch in range(opt.epoch_count, opt.niter + opt.niter_decay + 1):
        total_steps += 1
        loss = 0
        for i in range(10):
            model.set_input(data)
            model.optimize_parameters()
            model.save_network('latest')
            loss += model.loss
        loss /= 10   
        # print("epoch {}'s loss: {}".format(epoch,loss))
        if(loss < best_loss):
            model.save_network('best')
        best_loss = loss
        acc = predict(total_steps)
        model.update_learning_rate()
        wandb.log({"accuracy":acc, "loss": loss})