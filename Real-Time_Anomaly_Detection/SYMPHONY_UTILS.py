
import math
import pickle

import numpy as np 
import pandas as pd 

import torch
import matplotlib.pyplot as plt
from torch.utils.data import Dataset
from torch.utils.data.sampler import Sampler
from torch.optim.lr_scheduler import _LRScheduler
from sklearn.metrics import classification_report


trash_col_final = ['node5_DISKBUSY_sda',
 'node7_CPU001_Steal%',
 'node5_CPU011_Steal%',
 'node7_JFSFILE_/mnt/disk2',
 'node7_VM_nr_slab',
 'node6_DISKBUSY_sdc',
 'node8_DISKWRITE_sdc',
 'node5_VM_pswpout',
 'node7_DISKBUSY_sdi1',
 'node7_DISKBUSY_sdi2',
 'node6_VM_pgscan_kswapd_high',
 'node6_VM_pgrotated',
 'node8_CPU009_Steal%',
 'node8_VM_pgmajfault',
 'node8_VM_pgscan_direct_normal',
 'node5_CPU013_Steal%',
 'node7_NET_em2-write-KB/s',
 'node8_CPU018_Steal%',
 'node8_NET_idrac-read-KB/s',
 'node5_MEM_hightotal',
 'node5_CPU016_Steal%',
 'node6_CPU016_Steal%',
 'node8_VM_pgscan_direct_dma',
 'node5_MEM_memshared',
 'node5_CPU_ALL_CPUs',
 'node5_DISKREAD_sdi2',
 'node7_CPU012_Steal%',
 'node7_VM_pgmajfault',
 'node8_NETPACKET_em4-write/s',
 'node7_DISKBUSY_sdi3',
 'node7_DISKBSIZE_sdd',
 'node5_MEM_swapcached',
 'node5_CPU002_Steal%',
 'node8_CPU006_Steal%',
 'node5_CPU024_Steal%',
 'node7_DISKREAD_sdd',
 'node7_DISKWRITE_sdi1',
 'node5_CPU009_Steal%',
 'node6_PROC_Blocked',
 'node7_MEM_lowtotal',
 'node7_CPU019_Steal%',
 'node7_VM_pswpin',
 'node7_PROC_sem',
 'node8_NET_em4-read-KB/s',
 'node5_JFSFILE_/mnt/disk4',
 'node7_CPU010_Steal%',
 'node5_VM_pgscan_direct_dma',
 'node8_CPU001_Steal%',
 'node6_NETPACKET_em4-read/s',
 'node7_VM_pgrefill_high',
 'node8_MEM_buffers',
 'node8_PROC_sem',
 'node7_DISKXFER_sdi1',
 'node7_DISKWRITE_sdc',
 'node6_NET_idrac-write-KB/s',
 'node6_DISKBUSY_sdi2',
 'node5_DISKBSIZE_sdb',
 'node8_JFSFILE_/mnt/disk3',
 'node5_NETPACKET_em3-read/s',
 'node7_VM_pgscan_direct_high',
 'node6_MEM_swaptotal',
 'node7_DISKXFER_sdb',
 'node8_CPU015_Steal%',
 'node7_CPU017_Steal%',
 'node7_JFSFILE_/mnt/disk1',
 'node8_DISKREAD_sdd',
 'node8_NETPACKET_em4-read/s',
 'node5_CPU003_Steal%',
 'node7_CPU025_Steal%',
 'node8_NET_em3-read-KB/s',
 'node6_NET_em4-read-KB/s',
 'node6_MEM_memshared',
 'node5_NETPACKET_em2-read/s',
 'node8_CPU005_Steal%',
 'node8_CPU010_Steal%',
 'node6_JFSFILE_/boot',
 'node7_DISKWRITE_sdi2',
 'node6_PROC_msg',
 'node5_DISKBUSY_sdi3',
 'node5_PROC_Blocked',
 'node8_PROC_write',
 'node6_VM_pgdeactivate',
 'node8_NETPACKET_em3-write/s',
 'node5_VM_pgrefill_high',
 'node5_VM_kswapd_steal',
 'node7_DISKREAD_sdi1',
 'node7_CPU011_Steal%',
 'node7_CPU029_Steal%',
 'node5_NETPACKET_em4-write/s',
 'node7_MEM_highfree',
 'node6_JFSFILE_/run',
 'node7_CPU016_Steal%',
 'node5_JFSFILE_/',
 'node6_CPU021_Steal%',
 'node5_VM_pswpin',
 'node8_CPU021_Steal%',
 'node7_DISKWRITE_sda',
 'node8_NETPACKET_em2-write/s',
 'node6_NETPACKET_idrac-read/s',
 'node8_NET_em2-read-KB/s',
 'node6_DISKWRITE_sdc',
 'node5_CPU027_Steal%',
 'node6_MEM_bigfree',
 'node6_VM_kswapd_steal',
 'node8_DISKBUSY_sdi2',
 'node7_NETPACKET_em4-read/s',
 'node6_CPU002_Steal%',
 'node8_CPU012_Steal%',
 'node8_CPU032_Steal%',
 'node5_DISKWRITE_sdi1',
 'node8_PROC_read',
 'node6_CPU018_Steal%',
 'node6_DISKBUSY_sdd',
 'node6_VM_allocstall',
 'node8_JFSFILE_/.1',
 'node8_NETPACKET_idrac-write/s',
 'node7_MEM_buffers',
 'node6_CPU031_Steal%',
 'node8_DISKREAD_sdi',
 'node7_NETPACKET_em3-write/s',
 'node7_NET_em2-read-KB/s',
 'node8_NET_em4-write-KB/s',
 'node8_VM_pgdeactivate',
 'node8_CPU022_Steal%',
 'node5_DISKXFER_sda',
 'node6_DISKREAD_sdc',
 'node6_DISKREAD_dm-0',
 'node6_JFSFILE_/mnt/disk3',
 'node7_PROC_msg',
 'node8_VM_pgrefill_high',
 'node5_CPU030_Steal%',
 'node5_MEM_buffers',
 'node7_VM_pswpout',
 'node6_DISKBSIZE_sdb',
 'node8_DISKREAD_sdi2',
 'node8_DISKWRITE_sdi1',
 'node7_JFSFILE_/',
 'node6_JFSFILE_/mnt/disk1',
 'node5_CPU025_Steal%',
 'node6_NETPACKET_em2-write/s',
 'node5_PROC_msg',
 'node7_CPU006_Steal%',
 'node8_CPU030_Steal%',
 'node7_NETPACKET_idrac-write/s',
 'node5_VM_pgdeactivate',
 'node5_VM_pgscan_direct_normal',
 'node6_PROC_write',
 'node5_CPU019_Steal%',
 'node5_DISKXFER_sdi1',
 'node7_CPU028_Steal%',
 'node5_NET_em4-read-KB/s',
 'node7_PROC_syscall',
 'node8_MEM_swapcached',
 'node7_CPU008_Steal%',
 'node5_CPU001_Steal%',
 'node5_DISKWRITE_sdi2',
 'node6_VM_pginodesteal',
 'node7_MEM_memshared',
 'node6_PROC_read',
 'node5_DISKBUSY_sdc',
 'node7_NETPACKET_em3-read/s',
 'node6_CPU032_Steal%',
 'node7_CPU020_Steal%',
 'node7_DISKBUSY_sdi',
 'node6_MEM_lowtotal',
 'node8_DISKBUSY_sdb',
 'node6_NET_em2-write-KB/s',
 'node8_VM_nr_slab',
 'node5_NETPACKET_idrac-read/s',
 'node7_DISKBUSY_sdb',
 'node5_DISKBSIZE_sdc',
 'node5_VM_pgscan_direct_high',
 'node7_VM_pgsteal_normal',
 'node8_CPU029_Steal%',
 'node8_NET_em2-write-KB/s',
 'node5_DISKREAD_sdc',
 'node8_MEM_swaptotal',
 'node8_DISKBUSY_sda',
 'node7_CPU026_Steal%',
 'node5_DISKREAD_dm-0',
 'node5_DISKBSIZE_sdi1',
 'node6_VM_pgrefill_high',
 'node5_DISKBUSY_sdi2',
 'node8_CPU031_Steal%',
 'node7_CPU014_Steal%',
 'node5_DISKWRITE_sdb',
 'node6_DISKBSIZE_sdi2',
 'node5_PROC_sem',
 'node8_DISKREAD_sdi1',
 'node7_DISKBUSY_sdc',
 'node7_CPU023_Steal%',
 'node6_CPU029_Steal%',
 'node5_DISKBUSY_sdb',
 'node5_CPU006_Steal%',
 'node8_JFSFILE_/run',
 'node5_CPU007_Steal%',
 'node5_VM_pgalloc_high',
 'node7_JFSFILE_/dev',
 'node8_NETPACKET_idrac-read/s',
 'node5_DISKBSIZE_sda',
 'node6_NET_em3-write-KB/s',
 'node8_VM_pgsteal_high',
 'node5_CPU014_Steal%',
 'node8_VM_allocstall',
 'node7_DISKBSIZE_sdc',
 'node8_VM_pginodesteal',
 'node6_DISKXFER_sdi2',
 'node8_VM_pgsteal_normal',
 'node6_DISKREAD_sda',
 'node6_DISKREAD_sdb',
 'node5_DISKREAD_sdi',
 'node8_CPU016_Steal%',
 'node7_CPU003_Steal%',
 'node6_MEM_highfree',
 'node5_DISKREAD_sdi3',
 'node8_VM_kswapd_steal',
 'node6_DISKBSIZE_sdc',
 'node5_NET_em4-write-KB/s',
 'node6_VM_pswpin',
 'node5_DISKBUSY_sdi',
 'node6_DISKBSIZE_sdd',
 'node7_DISKBSIZE_sda',
 'node8_VM_pgalloc_dma',
 'node8_VM_pgalloc_high',
 'node6_NETPACKET_idrac-write/s',
 'node6_DISKBSIZE_sdi1',
 'node8_DISKREAD_dm-0',
 'node6_DISKREAD_sdi2',
 'node6_CPU_ALL_Steal%',
 'node7_CPU022_Steal%',
 'node7_DISKREAD_sdb',
 'node5_MEM_swaptotal',
 'node6_VM_pgsteal_high',
 'node7_CPU_ALL_Steal%',
 'node7_VM_pginodesteal',
 'node6_CPU_ALL_Busy',
 'node5_VM_pgscan_kswapd_high',
 'node7_DISKBUSY_sdd',
 'node8_MEM_highfree',
 'node7_CPU027_Steal%',
 'node6_VM_pgalloc_high',
 'node7_CPU013_Steal%',
 'node8_JFSFILE_/',
 'node7_VM_kswapd_steal',
 'node7_DISKXFER_sda',
 'node8_PROC_msg',
 'node8_DISKREAD_sdb',
 'node8_MEM_memshared',
 'node8_DISKBUSY_dm-0',
 'node8_VM_pgscan_direct_high',
 'node5_DISKBUSY_dm-0',
 'node5_CPU026_Steal%',
 'node8_PROC_exec',
 'node7_CPU024_Steal%',
 'node6_PROC_syscall',
 'node5_DISKXFER_sdb',
 'node5_VM_pgsteal_high',
 'node6_CPU017_Steal%',
 'node8_DISKXFER_sdd',
 'node7_CPU009_Steal%',
 'node7_DISKBSIZE_sdi1',
 'node6_CPU010_Steal%',
 'node6_DISKWRITE_sdi2',
 'node8_CPU027_Steal%',
 'node5_DISKWRITE_sdd',
 'node6_NET_em3-read-KB/s',
 'node7_DISKBSIZE_sdb',
 'node7_DISKREAD_sdc',
 'node5_CPU004_Steal%',
 'node5_CPU031_Steal%',
 'node6_DISKWRITE_sdb',
 'node5_VM_pgalloc_dma',
 'node8_VM_pgsteal_dma',
 'node6_DISKBUSY_sdi',
 'node6_CPU003_Steal%',
 'node8_VM_pgrefill_normal',
 'node5_VM_pginodesteal',
 'node8_NET_idrac-write-KB/s',
 'node6_CPU009_Steal%',
 'node5_PROC_read',
 'node8_CPU002_Steal%',
 'node6_MEM_swapfree',
 'node8_CPU011_Steal%',
 'node6_MEM_lowfree',
 'node8_CPU019_Steal%',
 'node7_DISKBUSY_dm-0',
 'node7_MEM_bigfree',
 'node5_DISKWRITE_sda',
 'node8_DISKBSIZE_sda',
 'node7_NETPACKET_idrac-read/s',
 'node5_NETPACKET_em3-write/s',
 'node7_CPU032_Steal%',
 'node8_VM_pgscan_kswapd_high',
 'node5_DISKBSIZE_sdi2',
 'node8_PROC_syscall',
 'node6_DISKBUSY_dm-0',
 'node6_NETPACKET_em4-write/s',
 'node8_DISKBUSY_sdi3',
 'node6_NET_em4-write-KB/s',
 'node8_DISKXFER_sda',
 'node5_CPU_ALL_Steal%',
 'node8_JFSFILE_/dev',
 'node7_VM_pgscan_kswapd_dma',
 'node8_NET_em3-write-KB/s',
 'node8_CPU023_Steal%',
 'node5_VM_pgsteal_normal',
 'node6_CPU030_Steal%',
 'node6_CPU027_Steal%',
 'node5_CPU018_Steal%',
 'node5_VM_pgrefill_dma',
 'node6_NET_idrac-read-KB/s',
 'node7_MEM_swaptotal',
 'node5_DISKREAD_sdb',
 'node7_CPU021_Steal%',
 'node7_NETPACKET_em2-read/s',
 'node8_CPU017_Steal%',
 'node8_DISKREAD_sdc',
 'node6_DISKBUSY_sda',
 'node8_DISKWRITE_sdb',
 'node7_VM_pgalloc_dma',
 'node6_VM_pgmajfault',
 'node8_JFSFILE_/boot',
 'node8_MEM_bigfree',
 'node6_VM_pgscan_direct_normal',
 'node8_DISKWRITE_sdd',
 'node8_VM_pgrotated',
 'node8_DISKXFER_sdi1',
 'node5_DISKBSIZE_sdd',
 'node8_DISKBUSY_sdi',
 'node6_VM_nr_slab',
 'node5_NET_em2-write-KB/s',
 'node6_CPU014_Steal%',
 'node6_VM_pgscan_direct_dma',
 'node8_DISKWRITE_sda',
 'node6_CPU011_Steal%',
 'node6_CPU012_Steal%',
 'node6_PROC_sem',
 'node7_CPU030_Steal%',
 'node6_DISKBUSY_sdb',
 'node6_NETPACKET_em2-read/s',
 'node7_CPU005_Steal%',
 'node6_MEM_buffers',
 'node8_DISKBSIZE_sdi2',
 'node5_CPU020_Steal%',
 'node7_NET_idrac-write-KB/s',
 'node5_VM_allocstall',
 'node6_VM_pgrefill_normal',
 'node7_DISKREAD_sdi2',
 'node5_PROC_write',
 'node6_CPU022_Steal%',
 'node8_CPU008_Steal%',
 'node7_CPU_ALL_Busy',
 'node8_DISKWRITE_sdi2',
 'node7_VM_pgsteal_high',
 'node8_CPU020_Steal%',
 'node5_JFSFILE_/boot',
 'node6_CPU023_Steal%',
 'node6_CPU015_Steal%',
 'node6_DISKWRITE_sdi1',
 'node8_MEM_hightotal',
 'node5_CPU010_Steal%',
 'node6_DISKXFER_sdi1',
 'node6_DISKBUSY_sdi1',
 'node6_CPU007_Steal%',
 'node7_CPU015_Steal%',
 'node8_DISKBUSY_sdd',
 'node5_CPU032_Steal%',
 'node6_CPU028_Steal%',
 'node6_DISKWRITE_sdd',
 'node8_DISKREAD_sda',
 'node5_VM_nr_slab',
 'node8_JFSFILE_/mnt/disk4',
 'node5_CPU023_Steal%',
 'node8_CPU007_Steal%',
 'node6_CPU025_Steal%',
 'node7_NETPACKET_em4-write/s',
 'node6_CPU001_Steal%',
 'node5_CPU012_Steal%',
 'node7_JFSFILE_/mnt/disk3',
 'node6_CPU013_Steal%',
 'node6_DISKWRITE_sda',
 'node7_DISKWRITE_sdd',
 'node7_VM_allocstall',
 'node5_CPU022_Steal%',
 'node8_CPU026_Steal%',
 'node8_MEM_swapfree',
 'node6_PROC_exec',
 'node8_CPU014_Steal%',
 'node8_DISKXFER_sdi2',
 'node5_CPU021_Steal%',
 'node7_DISKXFER_sdd',
 'node6_MEM_swapcached',
 'node7_VM_pgrefill_dma',
 'node5_JFSFILE_/run',
 'node5_PROC_exec',
 'node8_CPU024_Steal%',
 'node7_CPU018_Steal%',
 'node8_CPU_ALL_Steal%',
 'node6_CPU004_Steal%',
 'node7_VM_pgrefill_normal',
 'node5_VM_pgrefill_normal',
 'node5_CPU008_Steal%',
 'node7_PROC_read',
 'node6_DISKXFER_sda',
 'node7_VM_pgscan_kswapd_high',
 'node6_VM_pgsteal_normal',
 'node6_CPU_ALL_CPUs',
 'node7_DISKWRITE_sdb',
 'node5_DISKBUSY_sdi1',
 'node8_DISKBSIZE_sdc',
 'node5_JFSFILE_/dev',
 'node5_PROC_syscall',
 'node8_CPU003_Steal%',
 'node6_VM_pgrefill_dma',
 'node5_CPU_ALL_Busy',
 'node5_JFSFILE_/.1',
 'node7_CPU004_Steal%',
 'node5_MEM_highfree',
 'node6_MEM_hightotal',
 'node7_CPU007_Steal%',
 'node6_JFSFILE_/dev',
 'node5_MEM_bigfree',
 'node8_PROC_Blocked',
 'node7_DISKXFER_sdi2',
 'node6_CPU020_Steal%',
 'node7_MEM_lowfree',
 'node6_DISKBSIZE_sda',
 'node5_DISKXFER_sdi2',
 'node5_NET_em2-read-KB/s',
 'node7_CPU_ALL_CPUs',
 'node5_DISKREAD_sdi1',
 'node6_JFSFILE_/mnt/disk4',
 'node7_VM_pgsteal_dma',
 'node5_CPU028_Steal%',
 'node7_NET_idrac-read-KB/s',
 'node8_DISKXFER_sdb',
 'node6_CPU008_Steal%',
 'node7_DISKBSIZE_sdi2',
 'node5_JFSFILE_/mnt/disk2',
 'node6_VM_pageoutrun',
 'node6_CPU026_Steal%',
 'node6_NETPACKET_em3-write/s',
 'node6_VM_pgscan_direct_high',
 'node6_DISKXFER_sdc',
 'node6_VM_pgsteal_dma',
 'node7_VM_pgdeactivate',
 'node5_NETPACKET_em4-read/s',
 'node5_VM_pgrotated',
 'node5_CPU005_Steal%',
 'node5_MEM_swapfree',
 'node8_VM_pgscan_kswapd_dma',
 'node8_CPU013_Steal%',
 'node8_CPU004_Steal%',
 'node5_DISKWRITE_sdc',
 'node7_NET_em4-read-KB/s',
 'node8_VM_pswpin',
 'node7_NET_em3-read-KB/s',
 'node5_NETPACKET_idrac-write/s',
 'node6_DISKXFER_sdd',
 'node7_NET_em3-write-KB/s',
 'node6_CPU006_Steal%',
 'node6_DISKREAD_sdi3',
 'node8_VM_pgrefill_dma',
 'node7_DISKBUSY_sda',
 'node8_CPU025_Steal%',
 'node6_VM_pgscan_kswapd_dma',
 'node5_NETPACKET_em2-write/s',
 'node5_MEM_lowtotal',
 'node5_NET_em3-read-KB/s',
 'node6_CPU019_Steal%',
 'node8_DISKBSIZE_sdb',
 'node5_NET_idrac-write-KB/s',
 'node8_JFSFILE_/mnt/disk2',
 'node8_DISKBUSY_sdi1',
 'node7_DISKXFER_sdc',
 'node7_NETPACKET_em2-write/s',
 'node7_JFSFILE_/.1',
 'node8_VM_pswpout',
 'node7_VM_pgscan_direct_dma',
 'node8_CPU_ALL_Busy',
 'node6_CPU005_Steal%',
 'node5_NET_em3-write-KB/s',
 'node7_PROC_write',
 'node5_DISKXFER_sdd',
 'node6_VM_pgalloc_dma',
 'node6_JFSFILE_/mnt/disk2',
 'node7_JFSFILE_/boot',
 'node5_CPU017_Steal%',
 'node7_DISKREAD_sda',
 'node8_MEM_lowtotal',
 'node5_MEM_lowfree',
 'node8_CPU_ALL_CPUs',
 'node7_VM_pgscan_direct_normal',
 'node8_NETPACKET_em3-read/s',
 'node7_MEM_hightotal',
 'node6_DISKREAD_sdi',
 'node8_DISKBSIZE_sdd',
 'node7_MEM_swapfree',
 'node5_VM_pgscan_kswapd_dma',
 'node5_VM_pgsteal_dma',
 'node5_DISKBUSY_sdd',
 'node7_CPU031_Steal%',
 'node8_NETPACKET_em2-read/s',
 'node8_DISKBUSY_sdc',
 'node6_VM_kswapd_inodesteal',
 'node8_DISKREAD_sdi3',
 'node8_MEM_lowfree',
 'node6_CPU024_Steal%',
 'node6_NETPACKET_em3-read/s',
 'node8_JFSFILE_/mnt/disk1',
 'node5_CPU015_Steal%',
 'node7_CPU002_Steal%',
 'node6_JFSFILE_/.1',
 'node6_DISKREAD_sdi1',
 'node6_VM_pswpout',
 'node7_JFSFILE_/mnt/disk4',
 'node7_JFSFILE_/run',
 'node8_DISKXFER_sdc',
 'node7_PROC_exec',
 'node5_DISKREAD_sda',
 'node7_VM_pgrotated',
 'node6_DISKBUSY_sdi3',
 'node5_DISKXFER_sdc',
 'node8_DISKBSIZE_sdi1',
 'node6_JFSFILE_/',
 'node7_VM_pgalloc_high',
 'node6_DISKXFER_sdb',
 'node5_JFSFILE_/mnt/disk3',
 'node5_NET_idrac-read-KB/s',
 'node5_DISKREAD_sdd',
 'node7_MEM_swapcached',
 'node5_CPU029_Steal%',
 'node6_NET_em2-read-KB/s',
 'node5_JFSFILE_/mnt/disk1',
 'node7_NET_em4-write-KB/s',
 'node7_PROC_Blocked',
 'node5_VM_pgmajfault',
 'node6_DISKREAD_sdd',
 'node8_CPU028_Steal%']

derived_col_list = ['1_executor_runTime_count',
 '2_executor_runTime_count',
 'node5_NET_ib0-write-KB/s',
 'node5_PROC_pswitch',
 'node6_NET_ib0-read-KB/s',
 'node6_NET_ib0-write-KB/s',
 'node6_PROC_pswitch',
 'node7_NET_ib0-read-KB/s',
 'node7_NET_ib0-write-KB/s',
 'node7_PROC_pswitch',
 'node8_NET_ib0-read-KB/s',
 'node8_NET_ib0-write-KB/s',
 'node8_PROC_pswitch',
 'node5_NET_ib0-read-KB/s']


class CustomDataset(Dataset):
    """
    A custom PyTorch dataset class.

    Variables:
    ----------
    data : list
        The data to be used by the dataset.

    Methods:
    -------
    __len__():
        Returns the length of the dataset.
    __getitem__(idx):
        Returns the item at the given index.
    """
    def __init__(self, data):
        super(CustomDataset, self).__init__()
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        record = self.data[idx]
        return record


class BucketBatchingSampler(Sampler):
    """
    A custom PyTorch sampler class that batches data into buckets.

    Variables:
    ----------
    data_source : Dataset
        The dataset to be used by the sampler.
    config : dict
        A dictionary containing configuration parameters.

    Methods:
    -------
    __iter__():
        Returns an iterator over the batches.
    __len__():
        Returns the number of batches.
    """
    def __init__(self, data_source: Dataset, config: dict):
        super(BucketBatchingSampler, self).__init__(data_source)
        self.data_source = data_source      
        self.config = config      

        ids = list(range(len(self.data_source)))
        self.bins = [ids[i:i + self.config['batch_size']] for i in range(0, len(ids), self.config['batch_size'])]
                
    def __iter__(self):
        for ids in self.bins:
            yield ids

    def __len__(self):
        return len(self.bins)


class Custom_collate_fn(object):
    """
    A custom PyTorch collate function.

    Variables:
    ----------
    config : dict
        A dictionary containing configuration parameters.

    Methods:
    -------
    __call__(batch):
        Processes a batch of data.
    """
    def __init__(self, config: dict):
        self.config = config
    
    def __call__(self, batch):
        return batch[0]


class CosineAnnealingWarmUpRestarts(_LRScheduler):
    """
    Cosine annealing scheduler with restarts and warmup.
    """
    def __init__(self, optimizer, T_0, T_mult=1, eta_max=0.1, T_up=0, gamma=1., last_epoch=-1):
        """
        Args:
            optimizer (torch.optim.Optimizer): Wrapped optimizer.
            T_0 (int): Initial period.
            T_mult (int): Factor for increasing the period after restarts.
            eta_max (float): Maximum learning rate.
            T_up (int): Warmup period.
            gamma (float): Multiplicative factor for reducing the maximum learning rate.
            last_epoch (int): Index of last epoch (default: -1).
        """
        if T_0 <= 0 or not isinstance(T_0, int):
            raise ValueError("Expected positive integer T_0, but got {}".format(T_0))
        if T_mult < 1 or not isinstance(T_mult, int):
            raise ValueError("Expected integer T_mult >= 1, but got {}".format(T_mult))
        if T_up < 0 or not isinstance(T_up, int):
            raise ValueError("Expected positive integer T_up, but got {}".format(T_up))
        self.T_0 = T_0
        self.T_mult = T_mult
        self.base_eta_max = eta_max
        self.eta_max = eta_max
        self.T_up = T_up
        self.T_i = T_0
        self.gamma = gamma
        self.cycle = 0
        self.T_cur = last_epoch
        super(CosineAnnealingWarmUpRestarts, self).__init__(optimizer, last_epoch)
    
    def get_lr(self):
        if self.T_cur == -1:
            return self.base_lrs
        elif self.T_cur < self.T_up:
            return [(self.eta_max - base_lr)*self.T_cur / self.T_up + base_lr for base_lr in self.base_lrs]
        else:
            return [base_lr + (self.eta_max - base_lr) * (1 + math.cos(math.pi * (self.T_cur-self.T_up) / (self.T_i - self.T_up))) / 2
                    for base_lr in self.base_lrs]

    def step(self, epoch=None):
        if epoch is None:
            epoch = self.last_epoch + 1
            self.T_cur = self.T_cur + 1
            if self.T_cur >= self.T_i:
                self.cycle += 1
                self.T_cur = self.T_cur - self.T_i
                self.T_i = (self.T_i - self.T_up) * self.T_mult + self.T_up
        else:
            if epoch >= self.T_0:
                if self.T_mult == 1:
                    self.T_cur = epoch % self.T_0
                    self.cycle = epoch // self.T_0
                else:
                    n = int(math.log((epoch / self.T_0 * (self.T_mult - 1) + 1), self.T_mult))
                    self.cycle = n
                    self.T_cur = epoch - self.T_0 * (self.T_mult ** n - 1) / (self.T_mult - 1)
                    self.T_i = self.T_0 * self.T_mult ** (n)
            else:
                self.T_i = self.T_0
                self.T_cur = epoch
                
        self.eta_max = self.base_eta_max * (self.gamma**self.cycle)
        self.last_epoch = math.floor(epoch)
        for param_group, lr in zip(self.optimizer.param_groups, self.get_lr()):
            param_group['lr'] = lr     


def iter_preprocessing(df, iter, std, col_list, method, limit_direction):
    """
    Preprocess a DataFrame by iteratively interpolating missing values in selected columns.
    
    Args:
        df (dataframe): the DataFrame to preprocess
        iter (int): the number of iterations to perform
        std (float): a scalar value representing the number of standard deviations above which to consider values "missing"
        col_list (list): a list of column names to preprocess
        method (str): the interpolation method to use (e.g. 'linear', 'quadratic', etc.)
        limit_direction (str): the direction in which to limit the number of consecutive NaN values (e.g. 'forward', 'backward', 'both')
    Returns:
        Interpolated data.
    """
        
    for i in range(iter):
        # 모든 변수 중 각 변수에 대해서 보간 수행
        for col in col_list:
            # 모든 인스턴수 중 각각의 인스턴스에 대해서 보간 수행
            for app in df.app_id.unique():
                df_index = df[df.app_id == app].index
                
                # std 가 0인 경우는 보간 x
                if df[col].loc[df_index].std() == 0.0:
                    continue
                
                # 한 스텝 이동할 때 값의 차이(양수영역 및 음수영역)가 n-std diff 이상인 경우 해당 영역 선택            
                index_positive = df.loc[df_index][df[col].loc[df_index].diff(periods=1) >= df[col].loc[df_index].diff(periods=1).std()*std].index
                index_negative = df.loc[df_index][abs(df[col].loc[df_index].diff(periods=1)) >= abs(df[col].loc[df_index].diff(periods=1).std()*std)].index

                # 해당 영역 보간 수행. 양수, 음수 영역으로 나눠서 하지 않을 경우 보간 값이 뭉쳐짐 -> 양수 보간 후 음수 영역 보간 수행            
                df[col].iloc[index_positive] = np.nan
                df[col].iloc[index_negative] = np.nan
                
                # 모든 데이터가 NaN으로 변하는 것을 막음
                if len(df[col]) - df[col].isna().sum() == 0:
                    continue
                
                try:
                    df[col].loc[df_index] = df[col].loc[df_index].interpolate(method=method, limit_direction=limit_direction)
                except:
                    df[col].loc[df_index].fillna(method='ffill', inplace=True)
                    
    return df


def criterion(real, w1, w2, w3, n):
    
    loss1 = 1/n*torch.mean((real-w1)**2) + (1-1/n)*torch.mean((real-w3)**2)
    loss2 = 1/n*torch.mean((real-w2)**2) - (1-1/n)*torch.mean((real-w3)**2)

    return loss1, loss2


def scaling(config, data, scale_col):
    """
    Scales the specified columns of data using the min-max scaling method.
    
    Args:
        config (dictionary): hyperparameter dictionary.
        data (dataframe): To be scaled dataframe.
        scale_col (list): 전처리될 변수 리스트.
    Returns:
        Scaled data.
    """
    min_max_dict = {instance: None for instance in data.app_id.unique()}
        
    if config['phase'] == 'Train':
        for instance in data.app_id.unique():
            df_index = data[data.app_id == instance][scale_col].index
            
            max_arr = data[scale_col].iloc[df_index].max().values
            min_arr = data[scale_col].iloc[df_index].min().values
                
            # 사전에 변수별 최대, 최소값 저장
            min_max_dict[instance] = {scale_col[i]:[min_arr[i], max_arr[i]] for i in range(len(scale_col))}

            # min-max scaling
            for col, (col_min, col_max) in min_max_dict[instance].items():
                
                if col_max == col_min:
                    # Handle equal col_max and col_min values
                    data[col].iloc[df_index] = 1  # Assign a specific value
                    
                elif col_max == 0 and col_min == 0:
                    # Handle zero col_max and col_min values
                    epsilon = 1e-6  # Small positive value
                    data[col].iloc[df_index] = data[col].iloc[df_index] - epsilon
                    data[col].iloc[df_index] = data[col].iloc[df_index] / epsilon
                    
                else:
                    data[col].iloc[df_index] = data[col].iloc[df_index] - col_min
                    data[col].iloc[df_index] = data[col].iloc[df_index] / (col_max-col_min)
        
        with open(config['scaler_path'], 'wb') as fw:
            pickle.dump(min_max_dict, fw)
        print("scaler is saved at {}".format(config['scaler_path']))
    
    else:
        print("scaler is loaded at {}".format(config['scaler_path']))
        with open(config['scaler_path'], 'rb') as fr:
            min_max_dict = pickle.load(fr)
            
        for instance in data.app_id.unique():
            df_index = data[data.app_id == instance][scale_col].index
    
            # min-max scaling
            for col, (col_min, col_max) in min_max_dict[instance].items():
                
                if col_max == col_min:
                    # Handle equal col_max and col_min values
                    data[col].iloc[df_index] = 1  # Assign a specific value
                    
                elif col_max == 0 and col_min == 0:
                    # Handle zero col_max and col_min values
                    epsilon = 1e-6  # Small positive value
                    data[col].iloc[df_index] = data[col].iloc[df_index] - epsilon
                    data[col].iloc[df_index] = data[col].iloc[df_index] / epsilon
                    
                else:
                    data[col].iloc[df_index] = data[col].iloc[df_index] - col_min
                    data[col].iloc[df_index] = data[col].iloc[df_index] / (col_max-col_min)
        
    return data


def dummy_and_add_feature(x):
    second = int(x.split(" ")[-1].split(":")[2])
    minute = int(x.split(" ")[-1].split(":")[1])
    hour = int(x.split(" ")[-1].split(":")[0])
    day = int(x.split(" ")[0].split("-")[-1])
    month = int(x.split(" ")[0].split("-")[-2])
        
    sin_second = np.sin((2*np.pi*second)/(60))
    cos_second = np.cos((2*np.pi*second)/(60))
    sin_minute = np.sin((2*np.pi*minute*60)/(60*60))
    cos_minute = np.cos((2*np.pi*minute*60)/(60*60))
    sin_hour = np.sin((2*np.pi*hour*60*60)/(24*60*60))
    cos_hour = np.cos((2*np.pi*hour*60*60)/(24*60*60))
    sin_day = np.sin((2*np.pi*day*24*60*60)/(31*24*60*60))
    cos_day = np.cos((2*np.pi*day*24*60*60)/(31*24*60*60))
    sin_month = np.sin((2*np.pi*month*31*24*60*60)/(12*31*24*60*60))
    cos_month = np.cos((2*np.pi*month*31*24*60*60)/(12*31*24*60*60))
    
    return sin_second, cos_second, sin_minute, cos_minute, sin_hour, cos_hour, sin_day, cos_day, sin_month, cos_month


def F_1_calc(data, score):

    percentile_95 = np.percentile(score, 95)
    Normal_indices = np.where(score < percentile_95)[0]

    predict_label = pd.DataFrame(score).apply(lambda x: [1 if y >= percentile_95 else 0 for y in x])
    cblof_class_report = classification_report(data['Anomaly'], predict_label, target_names=['Normal', 'Abnormal'], output_dict=True)

    print('precision : ', cblof_class_report['macro avg']['precision'])
    print('recall : ', cblof_class_report['macro avg']['recall'])
    print('f1 : ', cblof_class_report['macro avg']['f1-score'])
    
    return Normal_indices, cblof_class_report