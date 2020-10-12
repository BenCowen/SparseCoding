from itertools import product
from collections import namedtuple, MutableMapping
from random import randint
import os
import time
import torch
import copy
import sys
try:
    from urllib.request import urlopen # For Python 3.0 and later
except ImportError:
    from urllib2 import urlopen # Fall back to Python 2's urllib


class ddict(object):
    '''
    dd = ddict(lr=[0.1, 0.2], n_hiddens=[100, 500, 1000], n_layers=2)
    # Save to shelf:
    dd._save('my_file', date=False)
    # Load ddict from file:
    new_dd = ddict()._load('my_file')
    '''
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)

    def __add__(self, other):
        if isinstance(other, type(self)):
            sum_dct = copy.copy(self.__dict__)
            for k,v in other.__dict__.items():
                if k not in sum_dct:
                    sum_dct[k] = v
                else:
                    if type(v) is list and type(sum_dct[k]) is list:
                        sum_dct[k] = sum_dct[k] + v
                    elif type(v) is not list and type(sum_dct[k]) is list:
                        sum_dct[k] = sum_dct[k] + [v]
                    elif type(v) is list and type(sum_dct[k]) is not list:
                        sum_dct[k] = [sum_dct[k]] + v
                    else:
                        sum_dct[k] = [sum_dct[k]] + [v]
            return ddict(**sum_dct)

        elif isinstance(other, dict):
            return self.__add__(ddict(**other))
        else:
            raise ValueError("ddict or dict is required")

    def __radd__(self, other):
        return self.__add__(other)

    def __repr__(self):
        items = ("{}={!r}".format(k, self.__dict__[k]) for k in self._keys())
        return "{}({})".format(type(self).__name__, ", ".join(items))

    def __eq__(self, other):
        return self.__dict__ == other.__dict__

    def __iter__(self):
        return self.__dict__.__iter__()

    def __len__(self):
        return len(self.__dict__)

    def __setitem__(self, key, value):
        self.__dict__[key] = value

    def __getitem__(self, key):
        return self.__dict__[key]

    def _keys(self):
        return tuple(sorted([k for k in self.__dict__ if not k.startswith('_')]))

    def _values(self):
        return  tuple([self.__dict__[k] for k in self._keys()])

    def _items(self):
        return  tuple(zip(self._keys(), self._values()))

    def _save(self, filename=None, date=False):
        if filename is None:
            if not hasattr(self, '_filename'): # First save
                raise ValueError("filename must be provided the first time you call _save()")
            else: # Already saved
                torch.save(self, self._filename + '.pt')
        else: # New filename
            if date:
                filename += '_'+time.strftime("%Y%m%d-%H:%M:%S")
            # Check if filename does not already exist. If it does, change name.
            while os.path.exists(filename + '.pt') and len(filename) < 100:
                filename += str(randint(0,9))
            self._filename = filename
            torch.save(self, self._filename + '.pt')
        return self

    def _load(self, filename):
        try:
            self = torch.load(filename)
        except FileNotFoundError:
            self = torch.load(filename + '.pt')
        return self

    def _to_dict(self):
        "Returns a dict (it's recursive)"
        return_dict = {}
        for k,v in self.__dict__.items():
            if isinstance(v, type(self)):
                return_dict[k] = v._to_dict()
            else:
                return_dict[k] = v
        return return_dict

    @staticmethod
    def _flatten_dict(d, parent_key='', sep='_'):
        "Recursively flattens nested dicts"
        items = []
        for k, v in d.items():
            new_key = parent_key + sep + k if parent_key else k
            if isinstance(v, MutableMapping):
                items.extend(ddict._flatten_dict(v, new_key, sep=sep).items())
            else:
                items.append((new_key, v))
        return dict(items)

    def _flatten(self, parent_key='', sep='_'):
        "Recursively flattens nested ddicts"
        d = self._to_dict()
        return ddict._flatten_dict(d)


class Gridder(object):
    '''
    g = Gridder(**{
        'lr': 0.1,
        'n_hiddens': [100, 500, 1000],
        'repetitions': list(range(5))})
    h = Gridder(lr=[0.1, 0.2], n_hiddens=[100, 500, 1000], n_layers=2)
    '''
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)
        self.__grid__ = None
        self.__tuple__ = None
        self.__normalize__()

    def __normalize__(self):
        "Checks that all values are lists (and not for instance individual instances)"
        for k in self.__keys__():
            if type(self.__dict__[k]) is not list:
                self.__dict__[k] = [self.__dict__[k]]

    def __add__(self, other):
        if isinstance(other, type(self)):
            uniq_keys = set(self.__keys__() + other.__keys__())
            sum_dct = {}
            for k in uniq_keys:
                vals = []
                try:
                    vals += self.__dict__[k]
                except KeyError:
                    pass
                try:
                    vals += other.__dict__[k]
                except KeyError:
                    pass
                sum_dct[k] = list(set(vals))
            return Gridder(**sum_dct)

        elif isinstance(other, dict):
            return self.__add__(Gridder(**other))
        else:
            raise ValueError("dict or Gridder is required")

    def __radd__(self, other):
        return self.__add__(other)

    def __keys__(self):
        keys = sorted([k for k in self.__dict__
                       if not k.startswith('__') and not k.endswith('__')])
        return keys

    def __values__(self):
        return [self.__dict__[k] for k in self.__keys__()]

    def __repr__(self):
        keys = self.__keys__()
        items = ("{}={!r}".format(k, self.__dict__[k]) for k in keys)
        return "{}({})".format(type(self).__name__, ", ".join(items))

    def __eq__(self, other):
        return self.__dict__ == other.__dict__

    def __next__(self):
        if self.__grid__ is None:
            self.__grid__ = product(*self.__values__())
            self.__tuple__ = namedtuple('params', self.__keys__())
        return self.__tuple__(*next(self.__grid__))

    def __iter__(self):
        return self

    def __len__(self):
        return len(self.__keys__())


def download_progress(url, dest_folder):

    print('Downloading ' + url)

    with open(os.path.join(dest_folder, os.path.basename(url)), "wb") as local_file:

        def chunk_report(bytes_so_far, chunk_size, total_size):
            percent = float(bytes_so_far) / total_size
            percent = round(percent*100, 2)
            sys.stdout.write("  Downloaded %d of %d kB (%0.2f%%)\r" %
                (bytes_so_far/1024, total_size/1024, percent))
            if bytes_so_far >= total_size:
                sys.stdout.write('\n')

        def chunk_read(response, chunk_size=102400, report_hook=None):
            total_size = response.info().get('Content-Length').strip()
            total_size = int(total_size)
            bytes_so_far = 0
            while 1:
                chunk = response.read(chunk_size)
                local_file.write(chunk)

                bytes_so_far += len(chunk)
                if not chunk:
                    break
                if report_hook:
                    report_hook(bytes_so_far, chunk_size, total_size)
            return bytes_so_far

        response = urlopen(url)
        chunk_read(response, report_hook=chunk_report)
