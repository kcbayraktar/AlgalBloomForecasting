import matplotlib.pyplot as plt
import os
import torch
import numpy as np


"""
Taken from Rodrigo
"""            
class Plots:
    def __init__(self, predictions, no_predictions):
        self.predictions = predictions
        self.no_predictions = no_predictions
        

    def plot_predictions(self):
        # create directories if they do not exist already.
        if not os.path.exists("results-palmar"):
                os.mkdir("results-palmar")
        for idx in range(0, self.no_predictions):

            parent_path = os.path.join("results-palmar", f"figure_{idx}")
            if not os.path.exists(parent_path):
                os.mkdir(parent_path)

            # make plots.
            predictions, labeled_image, mask = self.predictions[idx]
            self._plot_classification_prediction(predictions, labeled_image, mask, parent_path)
            
    #@staticmethod
    def _plot_classification_prediction(self, predictions, labeled_image, mask, parent_path):
        # shared settings.
        #vmin, vmax = 0, 4
        # Bin numbers taken from unet_model.py
        labels = ['0 mg/mL', '0-20 mg/mL', '20-50 mg/mL', '50-80 mg/mL', '>80 mg/mL']
        n = len(labels)
        ##r = vmax - vmin

        # plot predicted image.
        #batch size, num_bands
        plt.figure(figsize=(6, 4))
        plt.title('Predicted chlorophyll-a concentration')
        predicted_image = predictions.argmax(dim=0)
        predicted_image = np.expand_dims(predicted_image[~mask],axis=0)

        plt.imshow(predicted_image, cmap='hot', interpolation='nearest')
        ##ax = sns.heatmap(predicted_image, cmap=cmap, vmin=vmin, vmax=vmax, mask=mask, yticklabels=False, xticklabels=False)
        plt.savefig(os.path.join(parent_path, "predicted_image.pdf"))

        # plot actual image.
        plt.figure(figsize=(6, 4))
        plt.title('Actual chlorophyll-a concentration')

        plt.imshow(np.expand_dims(labeled_image[~mask],axis=0), cmap='hot', interpolation='nearest')
        plt.savefig(os.path.join(parent_path, "labeled_image.pdf"))

        # plot probabilities of the most probable class.
        plt.figure(figsize=(6, 4))
        plt.title('Confidence of the most probable class')
        probs_most_probable = torch.max(predictions, 0).values * 100
        plt.imshow(np.expand_dims(probs_most_probable[~mask],axis=0), cmap='hot', interpolation='nearest')
        plt.savefig(os.path.join(parent_path, "confidence_most_probable.pdf"))