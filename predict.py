import torch
import numpy as np
import albumentations as A
import matplotlib.pyplot as plt
from PIL import Image, ImageOps
from models import create_model
from options.predict_options import PredictOptions
from utils.util import dice_coefficient
import os 

def predict(steps=1):
    print("Running Prediction")
    opt = PredictOptions().parse()
    resize = opt.resize
    img_transform = A.Compose([
        A.CenterCrop(resize[0],resize[1]),
        A.Normalize(
            mean=(0.485, 0.456, 0.406),
            std=(0.229, 0.224, 0.225),
        )
    ])

    mask_transform = A.Compose([
        A.CenterCrop(resize[0],resize[1])
    ])

    model = create_model(opt) 
    # turn off gradient tracking
    with torch.no_grad():
        # load the image from disk and peform center cropping
        image = Image.open(opt.pred)
        image = np.array(image)
        image = img_transform(image=image)["image"]

        #add batch size to the numpy array
        image = np.expand_dims(image, 0)
        #change the channels
        image = np.transpose(image,(0,3,1,2))

        #convert it to tensor
        image = torch.from_numpy(image)

        #remove batch
        predMask = (model.predict(image)).squeeze()
        #np.savetxt("output.txt",predMask.cpu().detach().numpy(),fmt="%s")

        #Read the ground truth/mask and remove the channel and change it to numpy int array
        mask = Image.open(opt.label).convert('L')
        mask = np.asarray(mask)/255
        mask = mask.astype(np.uint8)

        #remove batch
        mask = mask_transform(image=mask)["image"].squeeze()
        
        dice = dice_coefficient(predMask,torch.from_numpy(mask))
        print("Dice Score: {:.2f}".format(dice.item()))

        predMask = torch.sigmoid(predMask)
        predMask = predMask.cpu().numpy()

        #Any thing above confidence of 0.5 will count as 1 and everything else is 0
        predMask = (predMask >= opt.confidence) * 255
        predMask = predMask.astype(np.uint8)


        #plot the prediction
        f = plt.figure()
        ax1 = f.add_subplot(1,2,1)
        plt.imshow(predMask)
        ax1.set_xlabel('Prediction')
        ax2 = f.add_subplot(1,2, 2)
        plt.imshow(mask)
        ax2.set_xlabel('Ground Truth')

        directory = opt.export_folder
        save_loc = str(steps) + "_result.png"
        os.makedirs(directory,exist_ok=True)
        f.savefig(directory + "/" + save_loc)
        plt.close()

if __name__ == '__main__':
    predict()