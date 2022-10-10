import torch
import time
from rich import print
from lib.common import gState, TrainArg
from lib.utils import AverageMeter, accuracy
from lib.comm import is_main_process, reduce_tensor, to_python_float, get_world_size, get_device


@torch.no_grad()
def validate(val_loader, model, criterion, args: TrainArg):
    batch_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    # switch to evaluate mode
    model.eval()

    end = time.time()

    for i, data in enumerate(val_loader):
        if args.cifar:
            input, target = data
            input, target = input.to(get_device()), target.to(get_device())
        else:
            input = data[0]["data"]
            target = data[0]["label"].squeeze().to(get_device()).long()
        # compute output

        gState.all_label = gState.all_label + list(target.cpu().numpy())

        with torch.no_grad():
            output = model(input)
            loss = criterion(output, target)

        # measure accuracy and record loss
        prec1, prec5 = accuracy(output.data, target, topk=(1, 5))

        reduced_loss = reduce_tensor(loss.data)
        prec1 = reduce_tensor(prec1)
        prec5 = reduce_tensor(prec5)

        losses.update(to_python_float(reduced_loss), input.size(0))
        top1.update(to_python_float(prec1), input.size(0))
        top5.update(to_python_float(prec5), input.size(0))

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if is_main_process() and i % args.print_freq == 0:
            print('Test: [{0}/{1}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Speed {2:.3f} ({3:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  '@1 {top1.val:.3f} ({top1.avg:.3f})\t'
                  '@5 {top5.val:.3f} ({top5.avg:.3f})'.format(
                      i, gState.val_loader_len,
                      get_world_size() * args.batch_size / batch_time.val,
                      get_world_size() * args.batch_size / batch_time.avg,
                      batch_time=batch_time, loss=losses,
                      top1=top1, top5=top5))
    print('[red]NUM_CLASSES={}'.format(args.num_classes))
    print('[green] * Prec@1 {top1.avg:.3f} Prec@5 {top5.avg:.3f}'
          .format(top1=top1, top5=top5))

    return [top1.avg, top5.avg]


if __name__ == "__main__":
    import argparse
    from torch import nn
    from typing import List
    from lib.loader import from_args
    from lib.loss import CrossEntropyLabelSmooth
    from lib.utils import add_dataclass_arguments, any_import
    parser = argparse.ArgumentParser()
    add_dataclass_arguments(parser, TrainArg)
    parser.add_argument("--dump", type=str, default="")
    args: TrainArg = parser.parse_args()
    args.evaluate = True
    gState.args = args

    fn = any_import(args.from_, args.import_)
    model: nn.Module = fn(num_classes=args.num_classes)
    if args.evaluate_model != 'ignore':
        checkpoint: dict = torch.load(args.evaluate_model, 'cpu')
        if "state_dict" in checkpoint:
            checkpoint = checkpoint["state_dict"]
        keys: List[str] = list(checkpoint.keys())
        for k in keys:
            if k.startswith("module."):
                checkpoint[k[7:]] = checkpoint.pop(k)
        stat = model.load_state_dict(checkpoint, False)
        print("[red]missing:",stat.missing_keys)

    model = model.cuda()

    criterion = CrossEntropyLabelSmooth().cuda()

    _, val_loader = from_args(args)

    validate(val_loader, model, criterion, args)

    if args.dump:
        gState.all_attention_weights["_labels"] = gState.all_label
        torch.save(gState.all_attention_weights, args.dump)
