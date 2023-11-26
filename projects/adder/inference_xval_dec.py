"""
cd /mnt/petrelfs/linweixiong/num_rep/minGPT/projects/adder && \
srun -p medai --job-name multi_inference_xval --gres=gpu:1 --quotatype=spot \
python inference_xval_dec.py
"""

import os
import sys
import json

import torch
from torch.nn import functional as F
from torch.utils.data import Dataset
from torch.utils.data.dataloader import DataLoader

from mingpt.model_xval_dec import GPT
from mingpt.trainer import Trainer
from mingpt.utils import set_seed, setup_logging, CfgNode as CN
import argparse

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--resume-path', type=str, action='append')
    args = parser.parse_args()
    return args

args = parse_args()

# -----------------------------------------------------------------------------

def get_config():

    C = CN()

    # system
    C.system = CN()
    C.system.seed = 3407
    C.system.work_dir = './out/multi_xval_dec'

    # data
    C.data = MultipDataset.get_default_config()

    # model
    C.model = GPT.get_default_config()
    C.model.model_type = 'gpt-mini'
    # C.model.model_type = 'gpt-nano'

    # trainer
    C.trainer = Trainer.get_default_config()
    C.trainer.learning_rate = 5e-4 # the model we're using is so small that we can go a bit faster

    C.resume_path = '/mnt/petrelfs/linweixiong/num_rep/minGPT/projects/adder/out/multi_xval_dec/model.pt'
    # C.resume_path = '/mnt/petrelfs/linweixiong/num_rep/minGPT/projects/adder/out/nano/multi_xval_v3/model.pt'

    return C


# -----------------------------------------------------------------------------
class MultipDataset(Dataset):
    """
    Creates n-digit addition problems. For example, if n=2, then an example
    addition problem would be to add 85 + 50 = 135. This problem would be
    represented as the following string for the GPT:

    "8550531"

    This is because:
    - we are discarding the + and =, which are not necessary. We just encode the digits
      of the input numbers concatenated together.
    - the result 135 is encoded backwards to make the addition easier to learn for the
      GPT model, because of how the addition algorithm works.

    As one more example, the problem 6 + 39 = 45 would be encoded as:

    "0639054"

    where you will notice that we are padding with zeros to make sure that we always
    produce strings of the exact same size: n + n + (n + 1). When n=2, this is 7.
    At test time, we will feed in an addition problem by giving the first 2n digits,
    and hoping that the GPT model completes the sequence with the next (n+1) digits
    correctly.
    """

    @staticmethod
    def get_default_config():
        C = CN()
        C.ndigit = 2
        return C

    def __init__(self, config, split):
        self.config = config
        self.split = split # train/test

        # split up all addition problems into either training data or test data
        ndigit = self.config.ndigit
        assert ndigit <= 3, "the lines below would be very memory inefficient, in future maybe refactor to support"
        num = (10**ndigit)**2 # total number of possible addition problems with ndigit numbers
        rng = torch.Generator()
        rng.manual_seed(1337)
        perm = torch.randperm(num, generator=rng)
        num_test = min(int(num*0.2), 500) # 20% of the whole dataset, or only up to 500
        self.ixes = perm[:num_test] if split == 'test' else perm[num_test:]

    def get_vocab_size(self):
        return 10 # digits 0..9

    def get_block_size(self):
        # a,b,a*b, and +1 due to potential carry overflow,
        # but then also -1 because very last digit doesn't ever plug back
        # as there is no explicit <EOS> token to predict, it is implied
        return (1+1+2)*self.config.ndigit - 1

    def __len__(self):
        return self.ixes.nelement()

    def __getitem__(self, idx):
        ndigit = self.config.ndigit
        # given a problem index idx, first recover the associated a + b
        idx = self.ixes[idx].item()
        nd = 10**ndigit

        a = idx // nd
        astr = f'%0{ndigit}d' % a
        # b = idx %  nd
        b = 0
        bstr = f'%0{ndigit}d' % b
        c = a
        # c = a * b
        cstr = (f'%0{2 * ndigit}d' % c)[::-1] # reverse c to make addition easier

        x_idx = torch.tensor([int(s) for s in astr + bstr], dtype=torch.long)
        return x_idx, torch.ones([1], dtype=torch.float), torch.tensor(c, dtype=torch.float)

# -----------------------------------------------------------------------------

if __name__ == '__main__':

    # get default config and overrides from the command line, if any
    config = get_config()
    config.merge_from_args(sys.argv[1:])
    print(config)
    setup_logging(config)
    set_seed(config.system.seed)

    # construct train and test datasets
    train_dataset = MultipDataset(config.data, split='train')
    test_dataset  = MultipDataset(config.data, split='test')

    # construct the model
    config.model.vocab_size = train_dataset.get_vocab_size()
    config.model.block_size = train_dataset.get_block_size()
    model = GPT(config.model)
    model.load_state_dict(torch.load(config.resume_path))
    model.eval()

    # construct the trainer object
    trainer = Trainer(config.trainer, model, train_dataset)

    ndigit = config.data.ndigit
    # helper function for the evaluation of a model
    factors = torch.tensor([[10**i for i in range(ndigit+1)][::-1]]).to(trainer.device)

    x = torch.tensor([
        [9,1,0,0], [9,2,0,0], [9,3,0,0], [9,4,0,0], [9,5,0,0],
        [9,0,0,0], [2,3,0,0],
    ]).to(trainer.device)
    x_num = torch.tensor([[1],[1],[1],[1],[1],[1],[1],]).to(trainer.device)
    # isolate the first two digits of the input sequence alone
    d1d2 = x[:, :ndigit*2]
    d1d2d3_idx, d1d2d3_num = model.generate(d1d2, x_num, 1, do_sample=False) # using greedy argmax, not sampling
    raise RuntimeError(d1d2d3_idx, d1d2d3_num)

    # ===========================================================
    # loader = DataLoader(train_dataset, batch_size=100, num_workers=0, drop_last=False)
    loader = DataLoader(test_dataset, batch_size=100, num_workers=0, drop_last=False)
    avg_rel_mse_loss = torch.tensor(0.0).to(trainer.device)
    avg_rel_l1_loss = torch.tensor(0.0).to(trainer.device)
    for b, batch in enumerate(loader):
        batch = [t.to(trainer.device) for t in batch]
        x_idx, y_idx, x_num, y_num = batch
        d1d2 = x_idx[:, :2]
        d1d2d3_idx, d1d2d3_num = model.generate(d1d2, x_num, 1, do_sample=False) # using greedy argmax, not sampling
        d3i_pred = d1d2d3_num[:, -1]
        d3i_gt = y_num[:, -1]

        raise RuntimeError(d3i_gt, d3i_pred)
        # 
        relative_mse_loss = F.mse_loss(d3i_gt, d3i_pred) / torch.abs(d3i_gt)
        relative_mse_loss_mean = relative_mse_loss.mean()
        avg_rel_mse_loss += relative_mse_loss_mean
        # 
        relative_l1_loss = torch.abs((d3i_pred - d3i_gt) / d3i_gt)
        relative_l1_loss_mean = relative_l1_loss.mean()

        ind = relative_l1_loss.sort().indices
        print(f'\033[32m111\033[0m: {relative_l1_loss[ind]}')
        print(f'\033[32m222\033[0m: {d3i_gt[ind].tolist()}')
        # raise RuntimeError('00')
        avg_rel_l1_loss += relative_l1_loss_mean
        # print(f'\033[32mrelative_mse_loss\033[0m: {relative_mse_loss_mean}')

    raise RuntimeError(avg_rel_mse_loss / len(loader), avg_rel_l1_loss / len(loader))
    # ===========================================================


    # let the model sample the rest of the sequence
    d1d2d3 = model.generate(d1d2, ndigit+1, max_new_tokens=1, do_sample=False) # using greedy argmax, not sampling
    # isolate the last digit of the sampled sequence
    d3 = d1d2d3[:, -(ndigit+1):]
    d3 = d3.flip(1) # reverse the digits to their "normal" order
    # decode the integers from individual digits
    d1i = (d1d2[:,:ndigit] * factors[:,1:]).sum(1)
    d2i = (d1d2[:,ndigit:ndigit*2] * factors[:,1:]).sum(1)
    d3i_pred = (d3 * factors).sum(1)
    d3i_gt = d1i + d2i # manually calculate the ground truth
    # evaluate the correctness of the results in this batch
    correct = (d3i_pred == d3i_gt).cpu() # Software 1.0 vs. Software 2.0 fight RIGHT on this line haha

    print(f'\033[32md3i_pred\033[0m: {d3i_pred}')
    print(f'\033[32md3i_gt\033[0m: {d3i_gt}')

