from option import args
import torch
import utility
import data
import model
from trainer import Trainer


if __name__ == '__main__':
    torch.cuda.set_device(args.n_GPUs)
    torch.manual_seed(args.seed)
    checkpoint = utility.checkpoint(args)

    if checkpoint.ok:
        loader = data.Data(args)
        model = model.Model(args, checkpoint)
        t = Trainer(args, loader, model, checkpoint)
        while not t.terminate():
            t.test()

        checkpoint.done()
