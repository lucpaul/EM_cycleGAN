"""
This package includes all the modules related to data loading and preprocessing.

To add a custom dataset class called 'dummy', you need to add a file called 'dummy_dataset.py' and define a subclass 'DummyDataset' inherited from BaseDataset.
You need to implement four functions:
    -- <__init__>: initialize the class, first call BaseDataset.__init__(self, opt).
    -- <__len__>: return the size of dataset.
    -- <__getitem__>: get a data point from data loader.
    -- <modify_commandline_options>: (optionally) add dataset-specific options and set default options.

Now you can use the dataset class by specifying flag '--dataset_mode dummy'.
See our template dataset class 'template_dataset.py' for more details.
"""

import importlib
import torch.utils.data
from .base_dataset_2d import BaseDataset2D
from .base_dataset_3d import BaseDataset3D


def find_dataset_using_name(dataset_name):
    """
    Import the module "data/[dataset_name]_dataset.py" and return the dataset class.

    In the file, the class called DatasetNameDataset() will be instantiated. It has to be a subclass of BaseDataset2D or BaseDataset3D, and it is case-insensitive.

    Args:
        dataset_name (str): Name of the dataset.

    Returns:
        type: Dataset class.

    Raises:
        NotImplementedError: If the dataset class is not found.
    """
    dataset_filename = "data." + dataset_name + "_dataset"
    datasetlib = importlib.import_module(dataset_filename)

    dataset = None
    target_dataset_name = dataset_name.replace("_", "") + "dataset"
    for name, cls in datasetlib.__dict__.items():
        if name.lower() == target_dataset_name.lower() and (
            issubclass(cls, BaseDataset2D) or issubclass(cls, BaseDataset3D)
        ):
            dataset = cls

    if dataset is None:
        raise NotImplementedError(
            "In %s.py, there should be a subclass of BaseDataset with class name that matches %s in lowercase."
            % (dataset_filename, target_dataset_name)
        )

    return dataset


def get_option_setter(dataset_name):
    """
    Return the static method <modify_commandline_options> of the dataset class.

    Args:
        dataset_name (str): Name of the dataset.

    Returns:
        function: The modify_commandline_options static method of the dataset class.
    """
    dataset_class = find_dataset_using_name(dataset_name)
    return dataset_class.modify_commandline_options


def create_dataset(opt):
    """
    Create a dataset given the option.

    This function wraps the class CustomDatasetDataLoader. This is the main interface between this package and 'train.py'/'test.py'.

    Args:
        opt: Option class with dataset configuration.

    Returns:
        CustomDatasetDataLoader: The dataset loader object.
    """
    data_loader = CustomDatasetDataLoader(opt)
    dataset = data_loader.load_data()
    return dataset


class CustomDatasetDataLoader:
    """
    Wrapper class of Dataset class that performs multi-threaded data loading.
    """

    def __init__(self, opt):
        """
        Initialize this class.

        Step 1: Create a dataset instance given the name [dataset_mode].
        Step 2: Create a multi-threaded data loader.

        Args:
            opt: Option class with dataset configuration.
        """
        self.opt = opt
        # if not opt.use_zarr:
        dataset_class = find_dataset_using_name(opt.dataset_mode)
        self.dataset = dataset_class(opt)
        print(type(self.dataset))
        print("dataset [%s] was created" % type(self.dataset).__name__)

        if opt.phase == "train":
            self.dataloader = torch.utils.data.DataLoader(
                self.dataset,
                batch_size=opt.batch_size,
                shuffle=not opt.serial_batches,
                num_workers=int(opt.num_threads),
                pin_memory=True,
            )
        else:
            self.dataloader = torch.utils.data.DataLoader(
                self.dataset,
                batch_size=1,  # opt.batch_size,  # We do not use batch size from DataLoader class here, but build our own batch size
                shuffle=False,
                num_workers=int(opt.num_threads),
                pin_memory=True,
            )

    def load_data(self):
        """
        Return the data loader itself.

        Returns:
            CustomDatasetDataLoader: The data loader instance.
        """
        return self

    def __len__(self):
        """
        Return the number of data in the dataset.

        Returns:
            int: Number of batches in the dataset.
        """
        return min(len(self.dataset), self.opt.max_dataset_size) // self.opt.batch_size

    def __iter__(self):
        """
        Return a batch of data.

        Yields:
            dict: A batch of data from the dataset.
        """
        for i, data in enumerate(self.dataloader):
            if i * self.opt.batch_size >= self.opt.max_dataset_size:
                break
            yield data
