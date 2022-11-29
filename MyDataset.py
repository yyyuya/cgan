import torch

class MyDataset(torch.utils.data.Dataset):
    def __init__(self, df, tag, labels):
        self.tags = df[tag].values
        # self.labels = df[labels].values
        self.transform = 

        PIL.Image.init()
        self._image_fnames = sorted(fname for fname in self._all_fnames if self._file_ext(fname) in PIL.Image.EXTENSION)
        if len(self._image_fnames) == 0:
            raise IOError('No image files found in the specified path')
        self._raw_shape = list(raw_shape)
        self._raw_labels = np.zeros([self._raw_shape[0], 0], dtype=np.float32)
    
    def __len__(self):
        return len(self.tags)
    
    def __getitem__(self, idx):
        tags_x = self.transform(torch.FloatTensor(self.tags[idx]))
        # labels = torch.LongTensor(self.labels[idx])
        return tags_x, labels
    
    def _get_raw_labels(self):
        if self._raw_labels is None:
            self._raw_labels = self._load_raw_labels() if self._use_labels else None
            if self._raw_labels is None:
                self._raw_labels = np.zeros([self._raw_shape[0], 0], dtype=np.float32)
            assert isinstance(self._raw_labels, np.ndarray)
            assert self._raw_labels.shape[0] == self._raw_shape[0]
            assert self._raw_labels.dtype in [np.float32, np.int64]
            if self._raw_labels.dtype == np.int64:
                assert self._raw_labels.ndim == 1
                assert np.all(self._raw_labels >= 0)
        return self._raw_labels

    def _load_raw_labels(self):
        fname = 'dataset.json'
        if fname not in self._all_fnames:
            return None
        with self._open_file(fname) as f:
            labels = json.load(f)['labels']
        if labels is None:
            return None
        labels = dict(labels)
        labels = [labels[fname.replace('\\', '/')] for fname in self._image_fnames]
        labels = np.array(labels)
        labels = labels.astype({1: np.int64, 2: np.float32}[labels.ndim])
        return labels


tag = 
dataset = MyDataset()
