from torch import nn
#from torchvision.transforms.functional import center_crop
from torchvision import transforms

class CroppedLoss:

    def __init__(self, assigned_loss=nn.L1Loss()):
        self.assigned_loss = assigned_loss

    def __call__(self, x1, x2):
        """
        :param x1, x2: two tensors of shape (..., H, W)


        :return: assigned loss of both tensors, bigger H&W tensor is center cropped to size of smaller tensor
        """
        #print(x1.shape, x2.shape)
        min_shape = min(x1.shape[-1], x2.shape[-1])
        if x1.shape[-1]>x2.shape[-1]:
            x_bigger = x1
            x_smaller = x2
        else:
            x_bigger = x2
            x_smaller = x1
        size_difference_z = x_bigger.shape[2] - min_shape
        size_difference_y = x_bigger.shape[3] - min_shape
        size_difference_x = x_bigger.shape[4] - min_shape
        x_bigger = x_bigger[:, :,
                 size_difference_z // 2:x_bigger.shape[2] - size_difference_z // 2,
                 size_difference_y // 2:x_bigger.shape[3] - size_difference_y // 2,
                 size_difference_x // 2:x_bigger.shape[4] - size_difference_x // 2]
        return self.assigned_loss(x_bigger, x_smaller)

    def __repr__(self):
        return self.__class__.__name__ + '()'