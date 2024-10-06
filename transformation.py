# -*- coding: utf-8 -*-
"""

Author:   Metin Bicer
email:    m.bicer19@imperial.ac.uk

"""
import torch


class labelScale(object):
    '''
    scale labels (oldMin and oldMax as the actual min and max values) to a
    range [NewMin, newMax]
    labels.shape = #examples, #labels
    '''
    def __init__(self, newMin=0, newMax=1, oldMin=False, oldMax=False):
        # whether to find mins and max from the data itself (or already provided)
        self.findMinMax = False
        if type(oldMin) == bool:
          if oldMin == False:
            self.findMinMax = True

        # type conversion if required
        newMin = torch.as_tensor(newMin)
        newMax = torch.as_tensor(newMax)
        oldMin = torch.as_tensor(oldMin)
        oldMax = torch.as_tensor(oldMax)

        # dimensionality checks
        if newMin.ndim == 0:
            if oldMin.ndim >= 1:
              newMin = newMin * torch.ones(oldMin.shape)
              newMax = newMax * torch.ones(oldMin.shape)

        elif oldMin.ndim == 0:
            if newMin.ndim >= 1:
              oldMin = oldMin * torch.ones(newMin.shape)
              oldMax = oldMax * torch.ones(newMin.shape)

        # save as attrs
        self.newMin = newMin
        self.newMax = newMax
        self.oldMin = oldMin
        self.oldMax = oldMax

    def __call__(self, data):
        """
        Args:
            data (torch.Tensor): tensor to be scaled to a range.
        Returns:
            Tensor: scaled tensor.
        """
        # data.shape = #examples, #labels
        if self.findMinMax:
            oldMinData = torch.min(data, dim=0).values
            oldMaxData = torch.max(data, dim=0).values
            s = labelScale(self.newMin, self.newMax, oldMinData, oldMaxData)
            data = s(data)
        else:
            data = self.newMin + (self.newMax - self.newMin) * (data - self.oldMin) / (self.oldMax - self.oldMin)
            # due to 0 division, there might be nans (replace them by 0)
            data[data != data] = 0
        return data

    def __repr__(self):
        return self.__class__.__name__ + '()'


class HeightChannelScale(object):
    """
    scale all columns of every channel of time series data to a specified range
    """
    def __init__(self, newMin=0, newMax=1, oldMin=False, oldMax=False):
        # whether to find mins and max from the data itself (or already provided)
        self.findMinMax = False
        if type(oldMin) == bool:
          if oldMin == False:
            self.findMinMax = True

        # type conversion if required
        newMin = torch.as_tensor(newMin)
        newMax = torch.as_tensor(newMax)
        oldMin = torch.as_tensor(oldMin)
        oldMax = torch.as_tensor(oldMax)

        # dimensionality checks
        if newMin.ndim <= 1:
            if oldMin.ndim > 1:
              oldMin = oldMin.view(oldMin.shape[0], 1, oldMin.shape[-1])
              oldMax = oldMax.view(oldMax.shape[0], 1, oldMax.shape[-1])
              newMin = newMin * torch.ones(oldMin.shape)
              newMax = newMax * torch.ones(oldMin.shape)

        elif oldMin.ndim <= 1:
            if newMin.ndim > 1:
              newMin = newMin.view(newMin.shape[0], 1, newMin.shape[-1])
              newMax = newMax.view(newMax.shape[0], 1, newMax.shape[-1])
              oldMin = oldMin * torch.ones(newMin.shape)
              oldMax = oldMax * torch.ones(newMin.shape)

        # save as attrs
        self.newMin = newMin
        self.newMax = newMax
        self.oldMin = oldMin
        self.oldMax = oldMax

    def __call__(self, data):
        """
        Args:
            data (torch.Tensor): tensor to be scaled.
        Returns:
            Tensor: scaled tensor.
        """
        # data.shape = CxHxW
        inShape = data.shape

        # if min and max are to be found
        if self.findMinMax:
            if len(inShape) == 3:
                data = data.reshape(1, inShape[0], inShape[1], inShape[2])
            try:
                oldMinData = torch.amin(torch.amin(data, dim=0), dim=1)
                oldMaxData = torch.amax(torch.amax(data, dim=0), dim=1)
            except:
                oldMinData = torch.min(torch.min(data, dim=0).values, dim=1).values
                oldMaxData = torch.max(torch.max(data, dim=0).values, dim=1).values

            # create an object of this class with the found min and max vals
            s = HeightChannelScale(self.newMin, self.newMax, oldMinData, oldMaxData)
            data = s(data)
        else:
            # dimensionality checks
            if self.oldMin.shape != self.newMin.shape:
                self.oldMin = self.oldMin.view(self.newMin.shape[0], self.newMin.shape[1], self.newMin.shape[2])
                self.oldMax = self.oldMax.view(self.newMin.shape[0], self.newMin.shape[1], self.newMin.shape[2])
            self.oldMin = self.checkLen(self.oldMin, 2)
            self.oldMax = self.checkLen(self.oldMax, 2)
            self.newMin = self.checkLen(self.newMin, 2)
            self.newMax = self.checkLen(self.newMax, 2)

            # scale the data
            data = self.newMin + (self.newMax - self.newMin) * (data - self.oldMin) / (self.oldMax - self.oldMin)
            # due to 0 division, there might be nans (replace them by 0)
            data[data != data] = 0

        # output dimensionality checks
        if len(inShape) == 3:
            data = data.reshape(inShape[0], inShape[1], inShape[2])
        return data

    def checkLen(self, var, l):
        if len(var.shape) == l:
            return var.reshape(var.shape[0], 1, var.shape[1])
        else:
            return var

    def __repr__(self):
        return self.__class__.__name__ + '()'
