def data_from_data_info(data_info):
    if data_info["name"] == 'Geometry':
        return GeometricShapesLoader(data_info["nr_points"],
                                     data_info["batch_size"]).get_data()
    elif data_info["name"] == 'ModelNet40':
        return ModelNet40Loader(data_info["nr_points"],
                                     data_info["batch_size"]).get_data()
    elif data_info["name"] == 'ModelNet10':
        return ModelNet10Loader(data_info["nr_points"],
                                     data_info["batch_size"]).get_data()
    else:
        raise NotImplementedError

class BaseDataLoader():
    def __init__(self):
        self.name = 'base'

    def get_data(self):
        raise NotImplementedError

    def get_info(self):
        raise NotImplementedError

class GeometricShapesLoader(BaseDataLoader):
    def __init__(self, nr_points, batch_size):
        self.name = 'Geometry'
        self.nr_points = nr_points
        self.batch_size = batch_size
        self.nr_classes = -1

    def get_data(self):
        from torch_geometric.datasets.geometry import GeometricShapes
        from torch_geometric.transforms import SamplePoints, Compose,  NormalizeScale, RandomRotate, RandomTranslate
        from torch_geometric.data import DataLoader


        trans = Compose((
                SamplePoints(self.nr_points),
                NormalizeScale(),
                RandomTranslate(0.01),
                RandomRotate(180)))

        dataset = GeometricShapes('data/geometric', train=True, transform=trans)
        nr_classes = len(dataset)
        self.nr_classes = nr_classes

        dataset = dataset.shuffle()

        val_loader   = DataLoader(dataset, batch_size=self.batch_size, drop_last=True)
        train_loader = DataLoader(dataset, batch_size=self.batch_size, drop_last=True)


        return train_loader, val_loader

    def get_info(self):
        assert not self.nr_classes == -1, "Need to Load Data before getting info"
        data_info = {
            "name": self.name,
            "nr_points": self.nr_points,
            "nr_classes": self.nr_classes,
            "batch_size": self.batch_size
        }

        return data_info

class ModelNet40Loader(BaseDataLoader):
    def __init__(self, nr_points, batch_size):
        self.name = 'ModelNet40'
        self.nr_points = nr_points
        self.batch_size = batch_size
        self.nr_classes = -1

    def get_data(self):
        from torch_geometric.datasets.modelnet import ModelNet
        from torch_geometric.transforms import SamplePoints, Compose,  NormalizeScale, RandomRotate, RandomTranslate
        from torch_geometric.data import DataLoader


        trans = Compose((
                SamplePoints(self.nr_points),
                NormalizeScale(),
                RandomTranslate(0.01),
                RandomRotate(180)))

        #dataset = ModelNet('/media/j-pc-ub/ExtraLinux', name='40', train=True, transform=trans)
        dataset = ModelNet('data/mn40', name='40', train=True, transform=trans)
        nr_classes = len(dataset)
        self.nr_classes = nr_classes

        dataset = dataset.shuffle()
        train_loader = DataLoader(dataset, batch_size=self.batch_size, drop_last=True)

        dataset_val = ModelNet('data/mn40', name='40', train=False, transform=trans)
        val_loader = DataLoader(dataset_val, batch_size=self.batch_size, drop_last=True)



        return train_loader, val_loader

    def get_info(self):
        assert not self.nr_classes == -1, "Need to Load Data before getting info"
        data_info = {
            "name": self.name,
            "nr_points": self.nr_points,
            "nr_classes": self.nr_classes,
            "batch_size": self.batch_size
        }

        return data_info


class ModelNet10Loader(BaseDataLoader):
    def __init__(self, nr_points, batch_size):
        self.name = 'ModelNet10'
        self.nr_points = nr_points
        self.batch_size = batch_size
        self.nr_classes = -1

    def get_data(self):
        from torch_geometric.datasets.modelnet import ModelNet
        from torch_geometric.transforms import SamplePoints, Compose,  NormalizeScale, RandomRotate, RandomTranslate
        from torch_geometric.data import DataLoader

        trans = Compose((
                SamplePoints(self.nr_points),
                NormalizeScale(),
                RandomTranslate(0.01),
                RandomRotate(180)))

        dataset = ModelNet('data/mn10', name='10', train=True, transform=trans)
        nr_classes = len(dataset)
        self.nr_classes = nr_classes

        train_loader = DataLoader(dataset, batch_size=self.batch_size, drop_last=True)

        dataset_val = ModelNet('data/mn10', name='10', train=False, transform=trans)

        val_loader = DataLoader(dataset_val, batch_size=self.batch_size, drop_last=True)

        return train_loader, val_loader

    def get_info(self):
        assert not self.nr_classes == -1, "Need to Load Data before getting info"
        data_info = {
            "name": self.name,
            "nr_points": self.nr_points,
            "nr_classes": self.nr_classes,
            "batch_size": self.batch_size
        }

        return data_info