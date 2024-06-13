import torch
from torch.utils.data import ConcatDataset, Dataset
import h5py

class ConcatData(ConcatDataset):
    def __init__(self, datasets):
        super().__init__(datasets)

    def minmax_normalize(self):
        data_mins = []
        data_maxs = []
        for d in self.datasets:
            min, max = d._min_max()
            data_mins.append(min)
            data_maxs.append(max)
        num_vars = len(data_mins[0])
        abs_min = torch.zeros_like(data_mins[0])
        abs_max = torch.zeros_like(data_maxs[0]) 
        for var in range(num_vars):
            var_mins = torch.tensor([x[var].item() for x in data_mins])
            var_maxs = torch.tensor([x[var].item() for x in data_maxs])
            abs_min[var] = torch.min(var_mins)
            abs_max[var] = torch.max(var_maxs)
        for d in self.datasets:
            abs_min, abs_max = d.minmax_normalize(abs_min, abs_max)

        return abs_min, abs_max

class BubbleML(Dataset):
    def __init__(self, filename, norm='minmax'):
        super().__init__()
        self.filename = filename
        self.norm = norm
        self.data = {}
        with h5py.File(filename, 'r') as f:
            if 'Twall-' in filename:
                wall_temp = int(filename.split('/')[-1].split('.')[0].split('-')[-1])
                if 'saturated' in filename.lower():
                    min_temp = 58
                if 'subcooled' in filename.lower():
                    min_temp = 50
            else:
                wall_temp = 1
                min_temp = 0
            self.data['dfun'] = torch.from_numpy(f['dfun'][...])
            self.data['temperature'] = torch.from_numpy(f['temperature'][...]) * (wall_temp - min_temp) + min_temp
            self.data['velx'] = torch.from_numpy(f['velx'][...])
            self.data['vely'] = torch.from_numpy(f['vely'][...])
        
        self.in_channels = 4
        self.out_channels = 4
        self.num_variables = 4

    def _min_max(self):
        mins = torch.tensor([torch.min(self.data['dfun']), torch.min(self.data['temperature']), torch.min(self.data['velx']), torch.min(self.data['vely'])])
        maxs = torch.tensor([torch.max(self.data['dfun']), torch.max(self.data['temperature']), torch.max(self.data['velx']), torch.max(self.data['vely'])])

        return mins, maxs

    def __len__(self):
        total_size = self.data['temperature'].shape[0]
        return total_size - 1
    
    def minmax_normalize(self, data_min=None, data_max=None):
        if data_min is not None and data_max is not None:
            self.data_min, self.data_max = data_min, data_max
        else:
            self.data_min, self.data_max = self._min_max()
        return self.data_min, self.data_max

    def _get_data_minmax(self, timestep):
        dfun = self.data['dfun'][timestep]
        dfun = (dfun - self.data_min[0])/(self.data_max[0] - self.data_min[0])
        temp = self.data['temperature'][timestep]
        temp = (temp - self.data_min[1])/(self.data_max[1] - self.data_min[1])
        velx = self.data['velx'][timestep]
        velx = (velx - self.data_min[2])/(self.data_max[2] - self.data_min[3])
        vely = self.data['vely'][timestep]
        vely = (vely - self.data_min[3])/(self.data_max[3] - self.data_min[3])
        data = torch.stack([dfun, temp, velx, vely], dim=0)
        
        return data

    def __getitem__(self, timestep):
        data_presents = []
        data_futures = []
        future_step = 1 
        
        if self.norm == 'minmax':
            data_presents.append(self._get_data_minmax(timestep))
            data_futures.append(self._get_data_minmax(timestep+future_step))
        else:
            raise NotImplementedError("Use 'minmax' normalisation with BubbleML")

        inputs = torch.cat(data_presents, dim=0)
        outputs = torch.cat(data_futures, dim=0)
        return inputs, outputs

if __name__=='__main__':

    data = BubbleML('/share/crsp/lab/ai4ts/share/BubbleML_f32/PoolBoiling-Saturated-FC72-2D-0.1/Twall-100.hdf5',
                     norm='minmax')
    data_min, data_max = data.minmax_normalize()
    print(data.data_min, data.data_max) 
    print(len(data), data.in_channels, data.out_channels)
    
    for i in range(5):
        input, output = data[i]
        print(input.shape, output.shape)
        print(input.dtype, output.dtype)