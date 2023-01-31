""" Full assembly of the parts to form the complete network """
import importlib
import torch
import pytorch_lightning as pl
import torch.optim as optim

from datetime import datetime
from torch.utils.data import DataLoader
from torchgeo.datasets import BoundingBox
from torchgeo.samplers import RandomGeoSampler, GridGeoSampler, Units
from .unet_parts import *
from torchmetrics import ConfusionMatrix
from torchmetrics.functional import accuracy


rionegrodata = importlib.import_module("rionegrodata")
data_analysis = importlib.import_module("utils.stat_helper")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


_CLIP = 250 
_MEAN = [2.37, 3.25, 1.30, 26.04, 17.50, 42.33, 0.75, 218.71, 74.00, -1.11, 0.14, 1.0]
_STD = [0.98, 0.44, 0.46, 8.72, 5.56, 34.48, 1.28, 108.29, 13.37, 2.83, 2.79, 365.0]
_TRANSFORM = [True, True, True, False, False, False, True, False, False, False, False, False]
_PREPROCESS = [True, True, True, True, True, True, True, True, True, True, True, False]
_BINS = [0, 10, 30, 75] 
weights = torch.tensor([0.0,1/63.574,1/23.185,1/8.351,1/4.885]).to(device)

class UNet(pl.LightningModule):
    def __init__(self,
                 root: str,
                 reservoir: str,
                 window_size: int = 5,
                 prediction_horizon: int = 1,
                 n_bands: int = 17,
                 n_classes: int = 5,
                 learning_rate: float = 1e-5,
                 weight_decay: float = 1e-8, 
                 train_samples: int = 100,
                 batch_size: int = 6,
                 num_workers: int = 8,
                 bilinear: bool = False
                 ):

        # Since it inherits from lightning module, we call the super constructor
        super(UNet, self).__init__()

        # Load dataset
        self.dataset = rionegrodata.RioNegroData(
            root=root,
            reservoir=reservoir,
            window_size=window_size,
            prediction_horizon=prediction_horizon,
            input_size=224,
        )
        # get stats
        self.analysis = data_analysis.DataAnalysis()
                
        # Save parameters in class.
        self.save_hyperparameters()
        self.n_channels = (window_size *n_bands) + 2 #2 for spatial info added later
        self.n_classes = n_classes
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.train_samples = train_samples
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.bilinear = bilinear

        # U-Net Architecture.
        self.inc = DoubleConv(self.n_channels, 64)
        self.down1 = Down(64, 128)
        self.down2 = Down(128, 256)
        self.down3 = Down(256, 512)
        factor = 2 if bilinear else 1
        self.down4 = Down(512, 1024 // factor)
        self.up1 = Up(1024, 512 // factor, bilinear)
        self.up2 = Up(512, 256 // factor, bilinear)
        self.up3 = Up(256, 128 // factor, bilinear)
        self.up4 = Up(128, 64, bilinear)
        self.outc = OutConv(64, self.n_classes)

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        logits = self.outc(x)
        return logits

    def _process_batch(self, batch):
        """
        helper for processing the batch, which comes in as  a tuple of (images, masks, targets)
        that as explained in rionegrodata.py. 

        """
        # take the input images and ground truth to compare
        input_images, _, labeled_images,spatial_info = batch
        input_images=input_images.float()

        # mask for missing values.
        mask = torch.isnan(labeled_images)


        # TRAIN PREPROCESS

        # Clip data to avoid extreme values.
        input_images = torch.clamp(input_images, -1e6, _CLIP)

        # Apply Yeo-Johnson transform to certain channels.
        for i in range(input_images.shape[2]):
            if _PREPROCESS[i]:
                # fill nan with mean to not affect normalization later
                input_images[:,:,i,:,:]=torch.nan_to_num(input_images[:,:,i,:,:],nan=_MEAN[i])
            if _TRANSFORM[i]:
                input_images[:, :, i, :, :] = torch.log1p(input_images[:, :, i, :, :])

        # # Convert NaN to 0.
        input_images[torch.isnan(input_images)] = 0

        # # Normalize data.
        mean = torch.tensor(_MEAN)[None, None, :, None, None].to(device)
        std = torch.tensor(_STD)[None, None, :, None, None].to(device)
        assert torch.isnan(mean).any()==False
        assert torch.isnan(std).any()==False
        input_images = (input_images - mean) / std

        # # Collapse temporal and band dimensions.
        input_images = input_images.view(input_images.shape[0], -1, input_images.shape[3], input_images.shape[4])


        spatial_info=torch.squeeze(spatial_info, dim=1)
        
        #spatial_info is a 4D Tensor (batch_size,2,height,width)
        #add spatial info to collapsed dimension

        input_images = torch.cat((input_images,spatial_info), 1)

        # # LABEL PREPROCESS

        #Fill NaN values
        labeled_images = torch.nan_to_num(labeled_images, nan=-1.0)

        # Clip, apply log transform and normalize data.
        labeled_images = torch.clamp(labeled_images, -0.0001, _CLIP)

        if self.n_classes > 1:
            # Bin creation on chosen values (high refers to 75+).
            boundaries = torch.tensor(_BINS)
            boundaries = boundaries.to(device)
            labeled_images = torch.bucketize(labeled_images, boundaries)
            
        return input_images, labeled_images, mask

    def _shared_eval_step(self, batch):
        """
        After processing the batch is done, the evaluation step calculates the loss and accuracy. Used by both training and validation.
        """
        input_images, labeled_images, mask = self._process_batch(batch)
        output_images = self(input_images)
        assert torch.isnan(input_images).any()==False
        assert torch.isnan(output_images).any()==False

        # for multiple classes (classification issue)
        if self.n_classes > 1:
            # compute loss and accuracy.

            criterion = nn.CrossEntropyLoss(ignore_index=0,weight=weights) #index[0] of x_image is just batch-ignore
            loss = criterion(input=output_images, target=labeled_images) # calculate loss according to criterion
            acc = accuracy(preds=output_images, target=labeled_images, ignore_index=0) # calculate accuracy
            # label 0 is background class
            return loss, acc, output_images,labeled_images
        else:
            # take last (and only) element (only 1 class to put in)
            output_images = output_images[:, -1, :, :]

            # accuracy cannot be calculated since there's no true or wrong classification 
            labeled_images=torch.squeeze(labeled_images,1)
            out = (output_images[~mask] - labeled_images[~mask]) ** 2
            loss = out.mean()
            labeled_images=torch.unsqueeze(labeled_images,1)
            return loss



    def training_step(self, batch, batch_idx):
        """
        Start the training step in accordance with Lightning Module.
        """
        self.train()
        if self.n_classes > 1:
            loss, acc,_,_ = self._shared_eval_step(batch)
            self.log('train_acc', acc, on_epoch=True)
        else:
            # if it's one class we only care about loss
            loss,_,_,_ = self._shared_eval_step(batch)

        #self.log('train_loss', loss,on_epoch=True,on_step=True)
        self.log('train_loss', loss)
        return {'loss': loss}

    def validation_step(self, batch, batch_idx):
        """
        Start the validation step in accordance with Lightning Module
        """
        self.eval()

        if self.n_classes > 1:
            loss, acc,predicted,ground_truth = self._shared_eval_step(batch)
            self.log('val_acc', acc,on_epoch=True)

        else:
            # if it's one class we only care about loss            
            loss, _,_,_ = self._shared_eval_step(batch)
        
        self.log('val_loss', loss)

    def predict_step(self, batch, batch_idx):
        """
        Start the prediction step in accordance with Lightning Module
        """
        self.eval()

        input_image, labeled_image, mask = self._process_batch(batch)
        output_image = self(input_image)
        if self.n_classes > 1:
            # turn the calculated numbers in the output image to probabilities through softmax
            #used for predicting and getting the predictions back. Otherwise same functionality as validation
            predicted_classes = torch.argmax(output_image,dim=1).flatten(start_dim=1,end_dim=2).squeeze().cpu()
            ground_truth = labeled_image.flatten(start_dim=1,end_dim=2).squeeze().cpu()
            assert predicted_classes.shape==ground_truth.shape
            assert torch.isnan(predicted_classes).any()==False

            cm = ConfusionMatrix(task='multiclass',normalize='true',num_classes=self.n_classes,ignore_index=0) # 0 is ignored
            printedcm = cm(preds=predicted_classes,target=ground_truth)
            self.analysis.conf_append(printedcm)
            print(self.analysis.conf_calculate())
            return predicted_classes, ground_truth, mask.squeeze().cpu().numpy()
        else:
            # scale output of sigmoid function by the maximum value of chlorophyll-a concentration.
            output = torch.sigmoid(output_image)[0]
            output = torch.mul(output, 5968.32).squeeze()
            return output, labeled_image.squeeze(), mask.squeeze().cpu().numpy()

    def configure_optimizers(self):
        """
        Set the optimizer with learning rate and weight decay as input parameters.
        """
        optimizer = optim.Adam(self.parameters(), lr= self.learning_rate, weight_decay=self.weight_decay)
        return optimizer

    def train_dataloader(self):
        """
        Prepares the train split by randomly sampling bounding boxes from the dataset within the region of interest
        """
        train_roi = BoundingBox(
            minx=self.dataset.roi.minx,
            maxx=self.dataset.roi.maxx,
            miny=self.dataset.roi.miny,
            maxy=self.dataset.roi.maxy,
            mint=self.dataset.roi.mint,
            maxt=datetime(2021, 12, 31).timestamp(),
        )
        # randomly samples geographic data
        train_sampler = RandomGeoSampler(
            self.dataset.data_bio_unprocessed,
            size=float(self.dataset.input_size),
            length=self.train_samples,  # Number of iterations in one epoch.
            roi=train_roi,
        )

        #loads the sample
        train_loader = DataLoader(
            self.dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            sampler=train_sampler
        )

        return train_loader

    def _evaluation_dataloader(self):
        """
        Prepares the test split by randomly sampling bounding boxes from the dataset within the region of interest
        """
        # Test split. This will the original images without cropping from the dataset.
        test_roi = BoundingBox(
            minx=self.dataset.roi.minx,
            maxx=self.dataset.roi.maxx,
            miny=self.dataset.roi.miny,
            maxy=self.dataset.roi.maxy,
            mint=datetime(2021, 12, 31).timestamp(),
            maxt=self.dataset.roi.maxt,
        )

        test_sampler = GridGeoSampler(
            self.dataset.data_bio_unprocessed,
            size=(self.dataset.roi.maxy - self.dataset.roi.miny, self.dataset.roi.maxx - self.dataset.roi.minx),
            stride=1,
            roi=test_roi,
            units=Units.CRS,
        )

        test_loader = DataLoader(
            self.dataset,
            batch_size=1,
            num_workers=self.num_workers,
            sampler=test_sampler
        )

        return test_loader

    def val_dataloader(self):
        return self._evaluation_dataloader()

    def predict_dataloader(self):
        return self._evaluation_dataloader()
