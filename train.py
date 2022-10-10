import time
import os
import os.path
import sys
import math
import shutil
import torch
from torch import nn
from rich import print
from lib.utils import AverageMeter, accuracy
from lib.common import TrainArg, gState
from lib.comm import reduce_tensor, to_python_float
from lib.common import time_stamp
from lib.utils import setup_seed, any_import
from lib.comm import get_local_rank, get_rank, synchronize, is_main_process, get_world_size


def save_checkpoint(state, is_best, work_dir='./', filename='checkpoint.pth.tar'):
    torch.save(state, os.path.join(work_dir, filename))
    if is_best:
        shutil.copyfile(os.path.join(work_dir, filename),
                        os.path.join(work_dir, 'model_best.pth.tar'))


def get_annealing(anneal_type=None,
                  max_anneal=1.,
                  anneal_epochs=0.2,
                  cyc_epoch=0.25,
                  cut=0):
    args: TrainArg = gState.args

    anneal_type = anneal_type or args.kl_type
    total = gState.total_steps - cut
    current = gState.global_step - cut
    if anneal_type == "mono":
        beta = max_anneal * min(
            current / (total * anneal_epochs), 1.0)
    elif anneal_type == "cyc":
        cyc_t = int(total * cyc_epoch)
        full_kl_step = cyc_t // 2
        beta = max_anneal * min(
            (current % cyc_t+1) / full_kl_step, 1.)
    elif anneal_type == "nokla":
        return 1.
    else:
        raise ValueError("Unsupported anneal type: {}"
                         .format(anneal_type))
    return beta


def train(train_loader, model: nn.Module, criterion,
          optimizer, epoch, logger, scheduler, args: TrainArg):
    if args.opt_level is not None:
        from apex import amp
    losses = AverageMeter()
    model.train()
    for i, data in enumerate(train_loader):
        gState.global_step = epoch * gState.train_loader_len + i
        if args.cifar:
            input, target = data
            input, target = input.cuda(), target.cuda()
        else:
            input = data[0]["data"]
            target = data[0]["label"].squeeze().cuda().long()
        scheduler.step()

        def closure():
            output = model(input)
            assert output.shape[1] == args.num_classes
            loss = criterion(output, target)
            return loss

        optimizer.zero_grad()
        loss = closure()
        if args.opt_level is not None:
            with amp.scale_loss(loss, optimizer) as scaled_loss:
                scaled_loss.backward()
        else:
            loss.backward()

        if args.clip_grad:
            nn.utils.clip_grad.clip_grad_value_(
                filter(lambda p: p.requires_grad, model.parameters()), 50.)
        optimizer.step(closure)

        if (i+1) % args.print_freq == 0 or i == gState.train_loader_len - 1:
            reduced_loss = reduce_tensor(loss.data)
            losses.update(to_python_float(reduced_loss), input.size(0))
            torch.cuda.synchronize()

            if is_main_process():
                print('Epoch: [{0}] Loss {loss.avg:.4f}'.format(
                    epoch, loss=losses))
                logger.add_scalar('Train/loss', losses.val,
                                  global_step=gState.global_step)
                logger.add_scalar(
                    'Meta/lr', optimizer.param_groups[0]['lr'], global_step=gState.global_step)

    return losses.avg


def parse() -> TrainArg:
    import argparse
    import os.path

    from lib.utils import add_dataclass_arguments
    parser = argparse.ArgumentParser(
        description='Main Script',
        usage=(
            "\n\tresnet50:  3 4 6 3\n"
            "\tresnet101: 3 4 23 3\n"
            "\tresnet152: 3 8 36 3\n"
        ))
    add_dataclass_arguments(parser, TrainArg)
    args: TrainArg = parser.parse_args()
    gState.local_rank = args.local_rank
    args.rho_init = - abs(args.rho_init)
    if args.disable_apex:
        args.opt_level = None

    if args.work_dir.endswith("XX"):
        # auto resume, no timestamp
        pass
    else:
        args.work_dir = os.path.join(args.work_dir, time_stamp + args.arch)

    if args.resume:
        # already specify ckpt
        pass
    else:
        # auto resume
        ckpt = os.path.join(args.work_dir, "checkpoint.pth.tar")
        try:
            if os.path.isfile(ckpt):
                args.resume = ckpt
        except Exception as e:
            print(e)

    args.total_batch_size = get_world_size() * args.batch_size
    gState.args = args
    gState.sam = args.sam
    return args


def calc_params(model: nn.Module, title=None):
    params = 0
    buffer = 0
    for p in model.parameters():
        if p.requires_grad:
            params += p.numel()
        else:
            buffer += p.numel()

    print("[green]", "="*20)
    if title:
        print("[green]", title)
    print("[green]", f"Params: {params / 1e6:.2f}M")
    print("[green]", f"Buffer: {buffer / 1e6:.2f}M")
    params *= 4
    buffer *= 4
    print("[green]", f"Params: {params / (2**20):.2f}MB")
    print("[green]", f"Buffer: {buffer / (2**20):.2f}MB")
    print("[green]", "="*20)


def main(args: TrainArg):
    if args.sync_bn or args.opt_level:
        from apex.parallel import DistributedDataParallel as DDP
        from apex import amp, optimizers, parallel
        assert torch.backends.cudnn.enabled, "Amp requires cudnn backend to be enabled."
    if args.opt_level is None:
        from torch.nn.parallel import DistributedDataParallel as DDP

    best_prec1 = 0
    torch.set_printoptions(precision=10)
    setup_seed(0)

    from torch.distributed import init_process_group
    init_process_group(backend='nccl')
    torch.cuda.set_device(get_local_rank())
    sys.stdout.write(r"""local_rank = {}  rank = {}  world_size = {}""".format(
        get_local_rank(), get_rank(), get_world_size())+"\n")

    print("""[green]
    opt_level           = {}
    keep_batchnorm_fp32 = {}
    loss_scale          = {}
    CUDNN VERSION       = {}""".format(
        args.opt_level, args.keep_batchnorm_fp32, args.loss_scale, torch.backends.cudnn.version()
    ))

    assert args.from_ and args.import_
    model_fn = any_import(args.from_, args.import_)
    if args.pretrained:
        model = model_fn(pretrained=args.pretrained,
                         num_classes=args.num_classes)
    else:
        model = model_fn(num_classes=args.num_classes)
    calc_params(model)
    print("[green]", f"=> from {args.from_} import {args.import_}")

    if args.sync_bn:
        print("[red]using apex synced BN")
        model = parallel.convert_syncbn_model(model)

    if hasattr(torch, 'channels_last') and hasattr(torch, 'contiguous_format'):
        if args.channels_last:
            memory_format = torch.channels_last
        else:
            memory_format = torch.contiguous_format
        model = model.cuda().to(memory_format=memory_format)
    else:
        model = model.cuda()

    if args.specify_lr is None:
        args.lr = args.lr * float(args.batch_size * get_world_size()) / 256.
    else:
        args.lr = args.specify_lr

    from optimizer import model_params
    if args.sam:
        from optimizer import SAMSGD as SGD
    else:
        from optimizer import SGD
    optimizer = SGD(
            model_params(model, args),
            args.lr,
            weight_decay=args.weight_decay,
            amsgrad=True)

    # synchronize()
    print("Sync ok")
    if args.opt_level is not None:
        model, optimizer = amp.initialize(
            model, optimizer,
            opt_level=args.opt_level,
            keep_batchnorm_fp32=args.keep_batchnorm_fp32,
            loss_scale=args.loss_scale
        )
        model = DDP(model, delay_allreduce=True)
    else:
        model = DDP(model,
                    device_ids=[get_local_rank()],
                    output_device=get_local_rank(),
                    find_unused_parameters=args.find_unused_params)

    if args.resume:
        def resume():
            global best_prec1
            assert os.path.isfile(args.resume),\
                f"`{args.resume}` isn't a file."
            print("[green]=> loading checkpoint '{}'".format(args.resume))
            checkpoint = torch.load(
                args.resume, map_location=lambda storage, loc: storage.cuda(get_local_rank()))
            gState.checkpoint = checkpoint
            args.start_epoch = checkpoint['epoch']
            gState.global_epoch = checkpoint['epoch']
            setup_seed(gState.global_epoch+args.seed)
            best_prec1 = checkpoint['best_prec1']
            model.load_state_dict(checkpoint['state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            print("[green]=> loaded checkpoint '{}' (epoch {})"
                  .format(args.resume, args.start_epoch))

        resume()
        if args.start_epoch >= args.epochs:
            print("[red]finished")
            return

    if (args.evaluate or args.finetune) and args.evaluate_model != "ignore":
        assert args.evaluate_model is not None
        print(
            "[green]=> loading checkpoint '{args.evaluate_model}' for eval/finetune")
        checkpoint = torch.load(
            args.evaluate_model,
            map_location=lambda storage, loc: storage.cuda(get_local_rank()))

        def state_filter(s: dict):
            return s
            keep = {}
            for k in s:
                if "se_module" not in k:
                    keep[k] = s[k]
            return keep

        if 'state_dict' in checkpoint.keys():
            incompataible = model.load_state_dict(
                state_filter(checkpoint['state_dict']), False)
        else:
            state_dict_with_module = {}
            for k, v in checkpoint.items():
                if k.startswith("module."):
                    state_dict_with_module = checkpoint
                    break
                state_dict_with_module['module.'+k] = v
            incompataible = model.load_state_dict(
                state_filter(state_dict_with_module), False)
        if incompataible.missing_keys:
            print("[red]missing keys {}".format(incompataible.missing_keys))
            for k in incompataible.missing_keys:
                assert "se_module" in k, "Missing {}".format(k)
        if incompataible.unexpected_keys:
            print("[red]unexpected keys {}".format(
                incompataible.unexpected_keys))
            for k in incompataible.unexpected_keys:
                assert "se_module" in k, "Unexpected {}".format(k)

    if args.evaluate:
        print("[red]please use evaluate script")
        return
    else:
        if is_main_process():
            os.makedirs(args.work_dir, exist_ok=True)
        from lib.comm import DistSummaryWriter
        logger = DistSummaryWriter(args.work_dir)
        gState.summary = logger
    from lib.loader import from_args
    from lib.loss import CrossEntropyLabelSmooth
    train_loader, val_loader = from_args(args)
    criterion_ = CrossEntropyLabelSmooth().cuda()

    def criterion(*args, **kwargs):
        loss = criterion_(*args, **kwargs)
        gState.custom_scalars["Train/ce_loss"] = to_python_float(loss)
        loss = loss + gState.external_loss
        gState.custom_scalars["Train/external_loss"] = to_python_float(
            gState.external_loss)
        gState.external_loss = 0
        return loss

    from lib.sched import create_scheduler
    scheduler = create_scheduler(optimizer, args)

    logger.add_text("argv", "%r" % sys.argv, 1)
    logger.add_text("args", "%r" % args, 1)

    for epoch in range(args.start_epoch, args.epochs):
        start_time = time.perf_counter()
        gState.global_epoch = epoch
        if get_local_rank() == 0:
            os.system("ipcs -m")
            os.system("free -h")
        print("[red]NUM_CLASSES={}".format(args.num_classes))
        setup_seed(epoch+args.seed)

        # train
        lossavg = train(
            train_loader, model,
            criterion, optimizer, epoch,
            logger, scheduler, args)
        torch.cuda.empty_cache()
        if math.isnan(lossavg):
            synchronize()
            break

        # val
        from evaluate import validate
        [prec1, prec5] = validate(val_loader, model, criterion, args)
        logger.add_scalar('Val/prec1', prec1, global_step=epoch)
        logger.add_scalar('Val/prec5', prec5, global_step=epoch)

        end_time = time.perf_counter()

        if is_main_process():
            is_best = prec1 > best_prec1
            best_prec1 = max(prec1, best_prec1)
            save_checkpoint({
                'epoch': epoch + 1,
                'arch': args.arch,
                'state_dict': model.state_dict(),
                'best_prec1': best_prec1,
                'scheduler': scheduler.state_dict(),
                'optimizer': optimizer.state_dict()
            }, is_best, work_dir=args.work_dir)
            if epoch == args.epochs - 1:
                print('[green]##Best Top-1 {0} Time {1:.1f}s'.format(
                    best_prec1, end_time - start_time))
                with open(os.path.join(args.work_dir, 'res.txt'), 'w') as f:
                    f.write('best_prec1 {1}\n {2}\n'.format(
                        args.arch + args.note, best_prec1, args))
                    f.write("%r\n" % sys.argv)
                    f.write("%r\n" % " ".join(sys.argv))

        train_loader.reset()
        val_loader.reset()

    print("finished.")
    time.sleep(3)


if __name__ == "__main__":
    args = parse()
    main(args)
