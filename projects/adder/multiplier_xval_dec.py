"""
xVal with decoder modification only.

```
cd /mnt/petrelfs/linweixiong/num_rep/minGPT/projects/adder && \
srun -p medai --job-name multi --gres=gpu:1 --quotatype=spot \
python multiplier_xval_dec.py

```
"""

import os
import sys
import json

import torch
from torch.utils.data import Dataset
from torch.utils.data.dataloader import DataLoader

from mingpt.model_xval_dec import GPT
from mingpt.trainer_xval_dec import Trainer_xval, BATCH_SIZE
from mingpt.utils import set_seed, setup_logging, CfgNode as CN

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
    C.trainer = Trainer_xval.get_default_config()
    C.trainer.learning_rate = 5e-4 # the model we're using is so small that we can go a bit faster

    C.wandb_log = False
    # C.wandb_log = True
    C.trainer.wandb_log = C.wandb_log

    # C.resume_path = None
    C.resume_path = '/mnt/petrelfs/linweixiong/num_rep/minGPT/projects/adder/out/multi_xval_dec/model.pt'

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
        x_num = torch.tensor([a, b])
        y_num = torch.tensor(c, dtype=torch.float)
        return x_idx, x_num, y_num
        # return x_idx, torch.ones([1], dtype=torch.float), torch.tensor(c, dtype=torch.float)

# -----------------------------------------------------------------------------

if __name__ == '__main__':

    # get default config and overrides from the command line, if any
    config = get_config()
    config.merge_from_args(sys.argv[1:])
    print(config)
    setup_logging(config)

    # wandb logging
    ddp_rank = 0
    master_process = ddp_rank == 0
    if config.wandb_log and master_process:
        import wandb
        wandb_logger = wandb.init(project='multi', name='multi_xval', config=config)
    else:
        wandb_logger = None

    set_seed(config.system.seed)
    # construct train and test datasets
    train_dataset = MultipDataset(config.data, split='train')
    # raise RuntimeError(len(train_dataset), train_dataset[0], train_dataset[-1])
    test_dataset  = MultipDataset(config.data, split='test')

    # construct the model
    config.model.vocab_size = train_dataset.get_vocab_size()
    config.model.block_size = train_dataset.get_block_size()
    model = GPT(config.model)
    if config.resume_path is not None:
        model.load_state_dict(torch.load(config.resume_path))

    # construct the trainer object
    trainer = Trainer_xval(config.trainer, model, train_dataset, wandb_logger)



    # helper function for the evaluation of a model
    def eval_split(trainer, split, max_batches=None):
        dataset = {'train':train_dataset, 'test':test_dataset}[split]
        ndigit = config.data.ndigit
        results = []
        mistakes_printed_already = 0
        factors = torch.tensor([[10**i for i in range(ndigit+1)][::-1]]).to(trainer.device)
        loader = DataLoader(dataset, batch_size=BATCH_SIZE, num_workers=0, drop_last=False)
        for b, batch in enumerate(loader):
            batch = [t.to(trainer.device) for t in batch]
            x_idx, x_num, y_num = batch
            y_idx = None
            # x_idx = x_idx.to(trainer.device)
            # isolate the first two digits of the input sequence alone
            d1d2 = x_idx
            # let the model sample the rest of the sequence
            d1d2d3_idx, d1d2d3_num = model.generate(d1d2, x_num, 1, do_sample=False) # using greedy argmax, not sampling
            d3i_pred = d1d2d3_num[:, -1]
            d3i_gt = y_num
            # raise RuntimeError(d3i_pred[:3], d3i_gt[:3])
            # raise RuntimeError((d3i_pred - d3i_gt).shape, d3i_gt.shape)
            results.append(torch.abs((d3i_pred - d3i_gt) / d3i_gt))
            # correct = (d3i_pred == d3i_gt).cpu() # Software 1.0 vs. Software 2.0 fight RIGHT on this line haha
            # for i in range(x.size(0)):
            #     results.append(int(correct[i]))
            #     if not correct[i] and mistakes_printed_already < 5: # only print up to 5 mistakes to get a sense
            #         mistakes_printed_already += 1
            #         print("GPT claims that %d + %d = %d but gt is %d" % (d1i[i], d2i[i], d3i_pred[i], d3i_gt[i]))
            # if max_batches is not None and b+1 >= max_batches:
            #     break
        rt = torch.cat(results)
        # rt = torch.cat(results, dtype=torch.float)
        # raise RuntimeError(rt.shape)
        print("%s: BatchNum: %d, MseMean = %.2f%%" % (split, len(results), 100*rt.mean()))
        return rt.sum()

    # iteration callback
    top_score = 0
    def batch_end_callback(trainer):
        global top_score

        if trainer.iter_num % 10 == 0:
            log_str = f'iter_dt {trainer.iter_dt * 1000:.2f}ms; iter {trainer.iter_num}: train loss {trainer.loss.item():.5f}, '
            # raise RuntimeError(trainer.loss, trainer.token_loss, trainer.num_loss)
            if trainer.token_loss is not None:
                log_str += f'token_loss {trainer.token_loss.item():.5f}, '
            if trainer.num_loss is not None:
                log_str += f'num_loss {trainer.num_loss.item():.5f}'
            print(log_str)
            # raise RuntimeError(trainer.loss, trainer.token_loss, trainer.num_loss)

        if trainer.iter_num % 500 == 0:
            # evaluate both the train and test score
            train_max_batches = {1: None, 2: None, 3: 5}[config.data.ndigit] # if ndigit=2 we can afford the whole train set, ow no
            model.eval()
            with torch.no_grad():
                train_score = eval_split(trainer, 'train', max_batches=train_max_batches)
                test_score  = eval_split(trainer, 'test',  max_batches=None)
            # score = train_score + test_score
            # save the model if this is the best score we've seen so far
            # if score > top_score:
            #     top_score = score
            # print(f"saving model with new top score of {score}")
            ckpt_path = os.path.join(config.system.work_dir, "model.pt")
            torch.save(model.state_dict(), ckpt_path)
            # revert model to training mode
            model.train()

    trainer.set_callback('on_batch_end', batch_end_callback)

    # run the optimization
    trainer.run()
