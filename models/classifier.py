import torch
import torch.nn as nn
from . import networks
from os.path import join
from torchmetrics import Dice
import matplotlib.pyplot as plt

class ClassifierModel:
    """ Class for training Model weights
    """
    def __init__(self, opt):
        self.opt = opt
        self.gpu_ids = opt.gpu_ids
        self.is_train = opt.is_train
        self.device = torch.device('cuda:{}'.format(self.gpu_ids[0])) if self.gpu_ids else torch.device('cpu')
        self.save_dir = join(opt.checkpoints_dir, opt.name)
        self.train_acc = []
        self.optimizer = None
        self.features = None
        self.labels = None
        self.loss = None

        # load/define networks
        self.net = networks.define_classifier(opt.ncf, opt.classes, opt, self.gpu_ids, opt.arch)
        
        self.net.train(self.is_train)
        self.criterion = networks.define_loss(opt).to(self.device)

        if self.is_train:
            self.optimizer = torch.optim.Adam(self.net.parameters(), lr=opt.lr)
            self.scheduler = networks.get_scheduler(self.optimizer, opt)

        if not self.is_train or opt.continue_train:
            self.load_network(opt.which_epoch)

    def set_input(self, data):
        features = torch.from_numpy(data['features']).float()
        labels = torch.from_numpy(data['labels']).long()

        # set inputs
        self.features = features.to(self.device).requires_grad_(self.is_train)
        self.labels = labels.to(self.device)
        
    def forward(self):
        out = self.net(self.features)
        return out

    def backward(self, out):
        self.loss = self.criterion(out, self.labels.float())
        self.train_acc.append(1-self.loss) #1 ==> 1 - diceloss = dice
        self.loss.backward()

    def optimize_parameters(self):
        self.optimizer.zero_grad()
        out = self.forward()
        self.backward(out)
        self.optimizer.step()


    def load_network(self, which_epoch):
        """load model from disk"""
        save_filename = '%s_net.pth' % which_epoch
        load_path = join(self.save_dir, save_filename)
        net = self.net
        if isinstance(net, torch.nn.DataParallel):
            net = net.module
        print('loading the model from %s' % load_path)

        state_dict = torch.load(load_path, map_location=str(self.device))
        if hasattr(state_dict, '_metadata'):
            del state_dict._metadata
        net.load_state_dict(state_dict)


    def save_network(self, which_epoch):
        """save model to disk"""
        save_filename = '%s_net.pth' % (which_epoch)
        save_path = join(self.save_dir, save_filename)
        if len(self.gpu_ids) > 0 and torch.cuda.is_available():
            torch.save(self.net.module.cpu().state_dict(), save_path)
            self.net.cuda(self.gpu_ids[0])
        else:
            torch.save(self.net.cpu().state_dict(), save_path)

    def update_learning_rate(self):
        """update learning rate (called once every epoch)"""
        self.scheduler.step()
        lr = self.optimizer.param_groups[0]['lr']
        print('learning rate = %.7f' % lr)

    def clear_history(self):
        self.train_acc = []
        self.test_acc = []
    
    def export_box(self,test_acc):
        fig, ax = plt.subplots()
        data = torch.stack(self.train_acc).squeeze().cpu().detach()
        data1 = torch.stack(test_acc).squeeze().cpu().detach()
        box1 = ax.boxplot(data, positions=[1], labels=['Train'])
        box2 = ax.boxplot(data1, positions=[2], labels=['Test'])
        plt.ylabel("Dice Score")

        # Save the boxplot as a JPEG image
        plt.savefig(self.opt.export_folder + '/boxplot.jpg', format='jpeg')


    ################################################################################
    # Helper Functions
    ################################################################################
    def test(self):
        """tests model
        returns: dice loss 
        """
        with torch.no_grad():
            out = self.forward()
            dice = Dice(average='micro').to(self.device)
            dice_val = dice(out, self.labels)
        return dice_val

    def predict(self,pred):
        """
        returns the predictions of a specific image
        """
        features = pred.to(self.device)
        with torch.no_grad():
            out = self.net(features)
        return out
    
    ################################################################################
  