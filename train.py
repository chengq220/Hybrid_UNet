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
    dataset = DataLoader(opt)
    dataset_size = len(dataset)
    model = create_model(opt)
    writer = Writer(opt)
    total_steps = 0
    best_loss = 1 

    # run = wandb.init(
    #     project="project"
    # )

    for epoch in range(opt.epoch_count, opt.niter + opt.niter_decay + 1):
        epoch_start_time = time.time()
        iter_data_time = time.time()
        epoch_iter = 0

        for i, data in enumerate(dataset):
            iter_start_time = time.time()
            if total_steps % opt.print_freq == 0:
                t_data = iter_start_time - iter_data_time
            total_steps += opt.batch_size
            epoch_iter += opt.batch_size
            model.set_input(data)
            model.optimize_parameters()

            if total_steps % opt.print_freq == 0:
                loss = model.loss
                t = (time.time() - iter_start_time) / opt.batch_size
                writer.print_current_losses(epoch, epoch_iter, loss, t, t_data)
                writer.plot_loss(loss, epoch, epoch_iter, dataset_size)
            # if(loss < best_loss):
            #         print('saving the latest model as best')
            #         model.save_network('best')

            if i % opt.save_latest_freq == 0:
                print('saving the latest model (epoch %d, total_steps %d)' %
                      (epoch, total_steps))
                model.save_network('latest')
            
            if total_steps % 2000 == 0:
                predict(total_steps)

            iter_data_time = time.time()

        print('End of epoch %d / %d \t Time Taken: %d sec' %
              (epoch, opt.niter + opt.niter_decay, time.time() - epoch_start_time))
        model.update_learning_rate()

        if epoch % opt.run_test_freq == 0:
            acc = run_test(epoch)
            writer.plot_acc(acc, epoch)
        # wandb.log({"accuracy": acc ,"loss": loss})

    writer.close()