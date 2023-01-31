import os
import time
from datetime import datetime, timedelta

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import DataLoader
from torchgeo.datasets import BoundingBox, RasterDataset  # type: ignore[attr-defined]
from torchgeo.samplers import GridGeoSampler, RandomGeoSampler  # type: ignore[attr-defined]
from torchgeo.samplers.constants import Units  # type: ignore[attr-defined]

plt.rcParams.update({"font.size": 8})

#_COORD = (-57.47600850853523, -56.9037816725511, -33.229580674865225, -32.90439054201396, 1484089200.0, 1640991599.999999)

class RioNegroBiological(RasterDataset):
    filename_glob = "*_biological_100m.tif"
    filename_regex = r".*_(?P<date>\d{8})"
    date_format = "%Y%m%d"
    is_image = True
    separate_files = False
    all_bands = [
        "chlorophyll-a-100m",
        "turbidity-100m",
        "cdom-100m",
        "chlorophyll-a-100m-mask",
        "turbidity-100m-mask",
        "cdom-100m-mask",
    ]

    def plot(self, sample):
        # Reorder and rescale the image
        image = sample["image"].permute(1, 2, 0)

        # Plot the image
        plt.imshow(image)


class RioNegroMeteorological(RasterDataset):
    filename_glob = "*_meteorological.tif"
    filename_regex = r"(?P<date>\d{8})"
    date_format = "%Y%m%d"
    is_image = True
    separate_files = False
    all_bands = [
        "air_temperature_min",
        "air_temperature_max",
        "air_temperature_mean",
        "cloud_coverage_mean",
        "precipitation_sum",
        "radiation_min",
        "radiation_max",
        "radiation_mean",
        "relative_humidity_min",
        "relative_humidity_max",
        "relative_humidity_mean",
        "u_wind_mean",
        "v_wind_mean",
    ]

    def plot(self, sample):
        # Reorder and rescale the image
        image = sample["image"].permute(1, 2, 0)
        image = image[:, :, 0]

        # Plot the image
        plt.imshow(image)


class RioNegroWaterTemperature(RasterDataset):
    filename_glob = "*_water_temperature.tif"
    filename_regex = r"(?P<date>\d{8})"
    date_format = "%Y%m%d"
    is_image = True
    separate_files = False
    all_bands = [
        "water_temperature", 
        "water_temperature_mask"
    ]

    def plot(self, sample):
        # Reorder and rescale the image
        image = sample["image"].permute(1, 2, 0)
        image = image[:, :, 0]

        # Plot the image
        plt.imshow(image)


class RioNegroData(torch.utils.data.Dataset):  # type: ignore[attr-defined]
    """
    Rio Negra dataset definition for PyTorch.

    The original biological dataset is used as the ground truth. It is used to retrieve
    the corresponding samples from the other datasets.

    The input data contains the following features, in this order:
    - Chlorophyll-a (processed)
    - Turbidity (processed)
    - CDOM (processed)
    - Water temperature (processed)
    - Air temperature min
    - Air temperature max
    - Air temperature mean
    - Cloud coverage mean
    - Precipitation sum
    - Radiation min
    - Radiation max
    - Radiation mean
    - Relative humidity min
    - Relative humidity max
    - Relative humidity mean
    - U wind mean
    - V wind mean

    Masks are provided to indicate which pixels are unmodified in the processed datasets
    for the following features: Chlorophyll-a, Turbidity, CDOM, Water temperature.

    Args:
        root (str): The root directory of the dataset.
        reservoir (str): The reservoir to use (e.g. 'palmar').
        window_size (int): The number of samples to use as input.
        prediction_horizon (int): The number of days to predict ahead.
        input_size (int): The size of the image.

    Returns:
        A tuple of (images, masks, targets) where: image is a batch of samples of shape
        (batch_size, window_size, num_bands, input_size, input_size), mask is a batch of
        shape (batch_size, window_size, 4, input_size, input_size) and target is a batch
        of samples of shape (batch_size, num_bands, input_size, input_size).

        The input samples are the <window_size> number of sequential samples
        <prediction horizon> timesteps before the target sample. E.g. for a prediction
        horizon of 4 days and a window size of 3, the input samples are samples (t-6),
        (t-5) and (t-4), where the target sample is (t).
    """

    def __init__(
        self,
        root: str,
        reservoir: str,
        window_size: int,
        prediction_horizon: int,
        input_size: int = 224,
    ):
        self.window_size = window_size
        self.prediction_horizon = prediction_horizon
        self.input_size = float(input_size)

        # Load separate datasets
        print("Loading datasets...")

        s = time.time()
        data_path = os.path.join(root, "biological", reservoir)
        self.data_bio_unprocessed = RioNegroBiological(root=data_path)
        e = time.time() - s
        print("Loaded unprocessed biological dataset in {:.2f} seconds".format(e))

        s = time.time()
        data_path = os.path.join(root, "biological_processed", reservoir)
        data_bio = RioNegroBiological(root=data_path)
        e = time.time() - s
        print("Loaded processed biological dataset in {:.2f} seconds".format(e))

        s = time.time()
        data_path = os.path.join(
            root, "physical_processed/water_temperature", reservoir
        )
        data_water_temp = RioNegroWaterTemperature(root=data_path)
        e = time.time() - s
        print("Loaded processed water temperature dataset in {:.2f} seconds".format(e))

        s = time.time()
        data_path = os.path.join(root, "meteorological_processed")
        data_meteo = RioNegroMeteorological(root=data_path)
        e = time.time() - s
        print("Loaded processed meteorological dataset in {:.2f} seconds".format(e))

        # Make intersection of datasets. TorchGeo will automatically convert resolutions
        # to be compatible.
        dataset_biophys = data_bio & data_water_temp
        self.dataset = dataset_biophys & data_meteo

        # Descriptions of bands.
        # 0,1,2,water_temp,meteo2,meteo3,meteo4,meteo7,meteo10,meteo11,meteo12
        self.all_bands = (
            data_bio.all_bands[:3]
            + [data_water_temp.all_bands[0]]
            + data_meteo.all_bands[2:5]
            + [data_meteo.all_bands[7]]
            + data_meteo.all_bands[10:]
        )

        # dataset_biophys = data_bio & data_water_temp
        # # self.dataset = dataset_biophys & data_meteo
        # self.dataset = dataset_biophys
        # # Descriptions of bands.
        # self.all_bands = (
        #         data_bio.all_bands[:3]
        #         + [data_water_temp.all_bands[0]]
        #         # + data_meteo.all_bands
        # )


        # We sample the ground-truth from the unprocessed bio data which has unmodified
        # values. We need to make sure that we sample within the range of all datasets.
        tdelta = timedelta(days=window_size + prediction_horizon)
        tmin = [
            self.data_bio_unprocessed.bounds.mint,
            (datetime.fromtimestamp(data_bio.bounds.mint) + tdelta).timestamp(),
            (datetime.fromtimestamp(data_meteo.bounds.mint) + tdelta).timestamp(),
            (datetime.fromtimestamp(data_water_temp.bounds.mint) + tdelta).timestamp(),
        ]
        tmax = [
            self.data_bio_unprocessed.bounds.maxt,
            data_bio.bounds.maxt,
            data_meteo.bounds.maxt,
            data_water_temp.bounds.maxt,
        ]

        self.roi = BoundingBox(
            self.data_bio_unprocessed.bounds.minx,
            self.data_bio_unprocessed.bounds.maxx,
            self.data_bio_unprocessed.bounds.miny,
            self.data_bio_unprocessed.bounds.maxy,
            np.max(np.array(tmin)),
            np.min(np.array(tmax)),
        )

    def __getitem__(self, bbox):
        # Get bio sample as ground truth.
        accessed_item=self.data_bio_unprocessed.__getitem__(bbox)
        gt = accessed_item["image"][0, :, :]

        # We then retrieve the corresponding samples from the processed datasets
        # according to the bounding box of the bio sample and the defined window size
        # and prediction horizon.

        # Convert unix timestamp float to datetime object.
        mint = datetime.fromtimestamp(bbox.mint)
        maxt = datetime.fromtimestamp(bbox.maxt)

        # Set time to 00:00:00 and 23:59:59.
        mint = mint.replace(hour=0, minute=0, second=0, microsecond=0)
        maxt = maxt.replace(hour=23, minute=59, second=59, microsecond=999999)

        sample_list = []
        # Iterate from current - (window_size + prediction_horizon) to current - prediction_horizon.
        for i in reversed(
            range(self.prediction_horizon, self.prediction_horizon + self.window_size)
        ):
            # Create bbox for the current time step.
            bbox = BoundingBox(
                mint=(mint + timedelta(days=-i)).timestamp(),
                maxt=(maxt + timedelta(days=-i)).timestamp(),
                minx=bbox.minx,
                maxx=bbox.maxx,
                miny=bbox.miny,
                maxy=bbox.maxy,
            )
            cur_mint=datetime.fromtimestamp(bbox.mint).timetuple().tm_yday
            
            #create a tuple of width,height with the respective timestamp
            temp_var=self.dataset.__getitem__(bbox)["image"]
            temporal_info=torch.full((1,int(temp_var.size(1)),int(temp_var.size(2))),cur_mint)
            time_as_band=torch.cat((temp_var,temporal_info),0)

            # Get samples.
            sample_list.append(time_as_band)


        bbox_y=torch.linspace(bbox.miny,bbox.maxy,int(accessed_item["image"].size(1)))     
        bbox_x=torch.linspace(bbox.minx,bbox.maxx,int(accessed_item["image"].size(2))) 
        grid_y, grid_x=torch.meshgrid(bbox_y,bbox_x,indexing="ij") # this will generate a tuple with
        spatial_info=torch.stack((grid_y,grid_x)) #2,height, width

        miny,maxy,minx,maxx,_,_= self.roi #BoundingBox

        # print("--------------------")
        # print("spatial info before norm")
        # print(spatial_info.shape)
        spatial_min = torch.tensor((minx,miny))[None, :, None, None]
        spatial_dims = torch.tensor((maxx-minx,maxy-miny))[None, :, None, None]
        assert torch.isnan(spatial_dims).any()==False
        spatial_info = (spatial_info-spatial_min) / spatial_dims # projects onto 0-1

        spatial_info = torch.mul((spatial_info-0.5),2)# projects onto -1 and 1
        
        # torch.where(spatial_info<-1 )
        # print("spatial info after norm")
        # print(spatial_info.shape)
        # print("--------------------")
        
        assert torch.all(torch.logical_and(spatial_info >= -1.001, spatial_info <= 1.001))
    
        
        x = torch.stack(sample_list)

        # Split images and masks into separate tensors.
        # Using bands [bio0,bio1,bio2,water_temp,meteo2,meteo3,meteo4,meteo7,meteo10,meteo11,meteo12].
        
        x_image = torch.cat(
            [
                x[:, :3, :, :],
                x[:, 6, :, :].unsqueeze(1),
                x[:, 10:13, :, :],
                x[:, 15, :, :].unsqueeze(1),
                x[:, 18:, :, :],
                
                #[x[:, :3, :, :], x[:, 6, :, :].unsqueeze(1), x[:, 8:, :, :]], dim=1
            ],
            dim=1,
        )
        #expecting size 12
        # print(x_image.shape)
        # print(x_image.size(1))
        x_mask = torch.cat([x[:, 3:6, :, :], x[:, 7, :, :].unsqueeze(1)], dim=1).bool()
        
        
        # # Split images and masks into separate tensors.
        # x_image = torch.cat(
        #     [x[:, :3, :, :], x[:, 6, :, :].unsqueeze(1), x[:, 8:, :, :]], dim=1
        # )
        # x_mask = torch.cat([x[:, 3:6, :, :], x[:, 7, :, :].unsqueeze(1)], dim=1).bool()


        # Assert x_mask only contains 0 and 1.
        assert torch.all(torch.logical_or(x_mask == 0, x_mask == 1))

        return x_image, x_mask, gt, spatial_info

    def __len__(self):
        return len(self.dataset)

    def __add__(self, _):
        raise NotImplementedError

    def plot(self, batch):
        sample = batch[0][0, 0, :, :, :].numpy()
        titles = self.all_bands

        plt.figure(figsize=(20, 5))
        for i in range(sample.shape[0]):
            plt.subplot(2, 9, i + 1)
            plt.title(titles[i])
            plt.imshow(sample[i, :, :], cmap="viridis")
            plt.axis("off")
        plt.show()


# for debugging purposes.
if __name__ == "__main__":

    # Load dataset
    dataset = RioNegroData(
        root="/.../", # to be filled in
        reservoir="palmar",
        window_size=4,
        prediction_horizon=5,
        input_size=224,
    )

    # Train split. This will sample random bounding boxes from the dataset of size input_size.
    train_roi = BoundingBox(
        minx=dataset.roi.minx,
        maxx=dataset.roi.maxx,
        miny=dataset.roi.miny,
        maxy=dataset.roi.maxy,
        mint=dataset.roi.mint,
        maxt=datetime(2021, 12, 31).timestamp(),
    )
    train_sampler = RandomGeoSampler(
        dataset.data_bio_unprocessed,
        size=float(dataset.input_size),
        length=100,  # Number of iterations in one epoch.
        roi=train_roi,
    )
    train_loader = DataLoader(
        dataset, batch_size=6, num_workers=0, sampler=train_sampler
    )

    # Iterate through training set. Will stop after one epoch.
    for i, (x_image, x_mask, y, _) in enumerate(train_loader):

        ####
        # Your training code goes here.
        ####

        print(i + 1, len(train_loader), x_image.shape, x_mask.shape, y.shape)

    # Test split. This will the original images without cropping from the dataset.
    test_roi = BoundingBox(
        minx=dataset.roi.minx,
        maxx=dataset.roi.maxx,
        miny=dataset.roi.miny,
        maxy=dataset.roi.maxy,
        mint=datetime(2021, 12, 31).timestamp(),
        maxt=dataset.roi.maxt,
    )
    test_sampler = GridGeoSampler(
        dataset.data_bio_unprocessed,
        size=(dataset.roi.maxy - dataset.roi.miny, dataset.roi.maxx - dataset.roi.minx),
        stride=1,
        roi=test_roi,
        units=Units.CRS,
    )

    test_loader = DataLoader(dataset, batch_size=1, num_workers=4, sampler=test_sampler)

    # Iterate through training set. Will stop after one epoch.
    for i, (x_image, x_mask, y) in enumerate(test_loader):

        ####
        # Your training code goes here.
        ####

        print(i + 1, len(test_loader), x_image.shape, x_mask.shape, y.shape)
