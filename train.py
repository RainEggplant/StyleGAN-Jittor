import argparse
import math
import random
from pathlib import Path

import jittor as jt
import jittor.transform as transforms
from jittor import nn, optim, grad
from tqdm import tqdm

from dataset import MultiResolutionDataset
from model import StyledGenerator, Discriminator
# jt.flags.log_silent = True


def requires_grad(model: nn.Module, flag=True):
    for p in model.parameters():
        p.requires_grad = flag


def accumulate(model1: nn.Module, model2: nn.Module, decay=0.999):
    par1 = dict(model1.named_parameters())
    par2 = dict(model2.named_parameters())

    for k in par1.keys():
        # # numpy inplace operation
        # par1[k].data *= decay
        # par1[k].data += (1 - decay) * par2[k].data
        par1[k].update(par1[k] * decay + (1 - decay) * par2[k].detach())


def sample_data(root, transform, batch_size, image_size=4):
    res_root = Path(root, str(image_size))
    dataset = MultiResolutionDataset(root, transform, resolution=image_size)
    loader = dataset.set_attrs(shuffle=True, batch_size=batch_size, num_workers=args.n_workers, drop_last=True)
    return loader


# TODO: check validity
def adjust_lr(optimizer, lr):
    for group in optimizer.param_groups:
        mult = group.get('mult', 1)
        group['lr'] = lr * mult


def sample_and_save(generator, step, alpha, latent, n_row, n_col, bs, save_name):
    n_iter = n_row * n_col // bs
    results = []
    for i_s in range(n_iter):
        results.append(generator(latent[i_s * bs:(i_s + 1) * bs], step=step, alpha=alpha).data)
    n_left = n_row * n_col - n_iter * bs
    if n_left > 0:
        results.append(generator(latent[-n_left:], step=step, alpha=alpha).data)

    jt.save_image(
        jt.concat(results, 0),
        save_name,
        nrow=n_row,
        normalize=True,
        range=(-1, 1),
    )


class SynchronizedOpFix:
    """
    Temporary fix of Jittor's bug.
    See https://github.com/Jittor/jittor/issues/197
    """
    def __enter__(self):
        if jt.in_mpi:
            jt.sync_all()

    def __exit__(self, exc_type, exc_val, exc_tb):
        if jt.in_mpi:
            jt.sync_all()


def train(args, transform, generator, discriminator):
    step = int(math.log2(args.init_size)) - 2
    resolution = 4 * 2 ** step
    loader = sample_data(
        args.path, transform, args.batch.get(resolution, args.batch_default), resolution
    )
    data_iter = iter(loader)

    adjust_lr(g_optimizer, args.lr.get(resolution, 0.001))
    adjust_lr(d_optimizer, args.lr.get(resolution, 0.001))

    pbar = range(3_000_000)
    if jt.rank == 0:
        pbar = tqdm(pbar, dynamic_ncols=True)

    requires_grad(generator, False)
    requires_grad(discriminator, True)

    disc_loss_val = 0
    gen_loss_val = 0
    grad_loss_val = 0

    alpha = 0
    used_sample = jt.array(0)

    max_step = int(math.log2(args.max_size)) - 2
    final_progress = False

    if jt.rank == 0:
        # fix sample latent code
        gen_i, gen_j = args.gen_sample.get(resolution, (10, 5))
        sample_z = jt.randn((gen_i * gen_j, code_size))
        sample_dir = Path(args.save_dir, 'sample')
        checkpoint_dir = Path(args.save_dir, 'checkpoint')
        (sample_dir / 'random').mkdir(parents=True, exist_ok=True)
        (sample_dir / 'fixed').mkdir(parents=True, exist_ok=True)
        checkpoint_dir.mkdir(parents=True, exist_ok=True)

    for i in pbar:
        if jt.in_mpi:
            used_sample.sync()
            jt.sync_all()

        alpha = min(1, 1 / args.phase * (used_sample + 1))

        if (resolution == args.init_size and args.ckpt is None) or final_progress:
            alpha = 1

        if used_sample > args.phase * 2:
            used_sample = jt.array(0)
            step += 1

            if step > max_step:
                step = max_step
                final_progress = True
                ckpt_step = step + 1

            else:
                alpha = 0
                ckpt_step = step

            resolution = 4 * 2 ** step

            batch_size = args.batch.get(resolution, args.batch_default)
            with SynchronizedOpFix():
                loader.terminate()
                loader = sample_data(
                    args.path, transform, batch_size, resolution
                )

            data_iter = iter(loader)
            learning_rate = args.lr.get(resolution, 0.001)

            if jt.rank == 0:
                print('\nsaving model...')
                jt.save(
                    {
                        'generator': generator.state_dict(),
                        'discriminator': discriminator.state_dict(),
                        'g_optimizer': g_optimizer.state_dict(),
                        'd_optimizer': d_optimizer.state_dict(),
                        'g_running': g_running.state_dict(),
                    },
                    checkpoint_dir / f'train_step-{ckpt_step}.model'
                )
                print(f'\nAdvancing from step {step} ==> {step + 1}: '
                      f'resolution {resolution}, batch size {batch_size}, lr {learning_rate:.3}')

            adjust_lr(g_optimizer, learning_rate)
            adjust_lr(d_optimizer, learning_rate)

        try:
            real_image = next(data_iter)

        except (OSError, StopIteration):
            data_iter = iter(loader)
            real_image = next(data_iter)

        used_sample += real_image.shape[0] * jt.world_size

        b_size = real_image.size(0)

        d_loss = 0
        if args.loss == 'wgan-gp':
            real_predict = discriminator(real_image, step=step, alpha=alpha)
            real_predict = real_predict.mean() - 0.001 * (real_predict ** 2).mean()
            d_loss += -real_predict

        elif args.loss == 'r1':
            real_image.requires_grad = True
            real_scores = discriminator(real_image, step=step, alpha=alpha)
            real_predict = nn.softplus(-real_scores).mean()
            d_loss += real_predict

            grad_real = grad(real_scores.sum(), real_image)
            grad_penalty = (
                    grad_real.view(grad_real.size(0), -1).norm(2, dim=1) ** 2
            ).mean()
            grad_penalty = 10 / 2 * grad_penalty
            d_loss += grad_penalty
            if i % 10 == 0:
                grad_loss_val = grad_penalty

        if args.mixing and random.random() < 0.9:
            gen_in11, gen_in12, gen_in21, gen_in22 = jt.randn(
                (4, b_size, code_size)).chunk(4, 0)
            gen_in1 = [gen_in11.squeeze(0), gen_in12.squeeze(0)]
            gen_in2 = [gen_in21.squeeze(0), gen_in22.squeeze(0)]

        else:
            gen_in1, gen_in2 = jt.randn((2, b_size, code_size)).chunk(2, 0)
            gen_in1 = gen_in1.squeeze(0)
            gen_in2 = gen_in2.squeeze(0)

        fake_image = generator(gen_in1, step=step, alpha=alpha)
        fake_predict = discriminator(fake_image, step=step, alpha=alpha)

        if args.loss == 'wgan-gp':
            fake_predict = fake_predict.mean()
            d_loss += fake_predict

            eps = jt.rand((b_size, 1, 1, 1))
            x_hat = eps * real_image.data + (1 - eps) * fake_image.data
            x_hat.requires_grad = True
            hat_predict = discriminator(x_hat, step=step, alpha=alpha)
            grad_x_hat = grad(hat_predict.sum(), x_hat)
            grad_penalty = (
                    (grad_x_hat.view(grad_x_hat.size(0), -1).norm(2, dim=1) - 1) ** 2
            ).mean()
            grad_penalty = 10 * grad_penalty
            d_loss += grad_penalty
            if i % 10 == 0:
                grad_loss_val = grad_penalty
                disc_loss_val = (-real_predict + fake_predict)

        elif args.loss == 'r1':
            fake_predict = nn.softplus(fake_predict).mean()
            d_loss += fake_predict
            if i % 10 == 0:
                disc_loss_val = (real_predict + fake_predict)

        with SynchronizedOpFix():
            d_optimizer.step(d_loss)

        if (i + 1) % n_critic == 0:
            requires_grad(generator, True)
            requires_grad(discriminator, False)

            fake_image = generator(gen_in2, step=step, alpha=alpha)

            predict = discriminator(fake_image, step=step, alpha=alpha)

            if args.loss == 'wgan-gp':
                loss = -predict.mean()

            elif args.loss == 'r1':
                loss = nn.softplus(-predict).mean()

            if i % 10 == 0:
                gen_loss_val = loss

            with SynchronizedOpFix():
                g_optimizer.step(loss)

            accumulate(g_running, generator)

            requires_grad(generator, False)
            requires_grad(discriminator, True)

        if (i + 1) % 100 == 0 and jt.rank == 0:
            sample_batch = real_image.shape[0]
            with jt.no_grad():
                sample_and_save(g_running, step, alpha, jt.randn(gen_i * gen_j, code_size),
                                gen_i, gen_j, sample_batch, sample_dir / f'random/{i + 1:06}.png')
                sample_and_save(g_running, step, alpha, sample_z,
                                gen_i, gen_j, sample_batch, sample_dir / f'fixed/{i + 1:06}.png')

        if (i + 1) % 10000 == 0 and jt.rank == 0:
            print('\nsaving inference model...')
            jt.save(
                g_running.state_dict(), checkpoint_dir / f'{i + 1:06}.model'
            )

        # will cause dead lock. I don't know how to solve this, so just use data in rank 0
        # if jt.in_mpi:
        #     gen_loss_val = gen_loss_val.mpi_all_reduce()
        #     disc_loss_val = disc_loss_val.mpi_all_reduce()
        #     grad_loss_val = grad_loss_val.mpi_all_reduce()

        if jt.rank == 0:
            state_msg = (
                f'Size: {4 * 2 ** step}; G: {gen_loss_val.item():.3f}; D: {disc_loss_val.item():.3f};'
                f' Grad: {grad_loss_val.item():.3f}; Alpha: {alpha:.5f}'
            )
            pbar.set_description(state_msg)


if __name__ == '__main__':
    code_size = 512
    batch_size = 16
    n_critic = 1

    parser = argparse.ArgumentParser(description='Progressive Growing of GANs')

    parser.add_argument('path', type=str, help='path of specified dataset')
    parser.add_argument(
        '--n_workers', type=int, default=16, help="batch sizes for each gpus"
    )
    parser.add_argument(
        '--phase',
        type=int,
        default=600_000,
        help='number of samples used for each training phases',
    )
    parser.add_argument('--lr', default=0.001, type=float, help='learning rate')
    parser.add_argument('--sched', action='store_true', help='use lr scheduling')
    parser.add_argument('--init_size', default=8, type=int, help='initial image size')
    parser.add_argument('--max_size', default=1024, type=int, help='max image size')
    parser.add_argument(
        '--ckpt', default=None, type=str, help='load from previous checkpoints'
    )
    parser.add_argument(
        '--no_from_rgb_activate',
        action='store_true',
        help='use activate in from_rgb (original implementation)',
    )
    parser.add_argument(
        '--mixing', action='store_true', help='use mixing regularization'
    )
    parser.add_argument(
        '--loss',
        type=str,
        default='wgan-gp',
        choices=['wgan-gp', 'r1'],
        help='class of gan loss',
    )
    parser.add_argument('--save_dir', type=str, default='saved')
    parser.add_argument('--cpu', action='store_true', help='use CPU only')

    args = parser.parse_args()

    jt.flags.use_cuda = 0 if args.cpu else 1
    generator = StyledGenerator(code_size)
    discriminator = Discriminator(from_rgb_activate=True)

    g_running = StyledGenerator(code_size)
    g_running.eval()

    g_optimizer = optim.Adam(
        generator.generator.parameters(), lr=args.lr, betas=(0.0, 0.99)
    )
    g_optimizer.add_param_group(
        {
            'params': generator.style.parameters(),
            'lr': args.lr * 0.01,
            'mult': 0.01,
        }
    )
    d_optimizer = optim.Adam(discriminator.parameters(), lr=args.lr, betas=(0.0, 0.99))

    accumulate(g_running, generator, 0)

    if args.ckpt is not None:
        ckpt = jt.load(args.ckpt)

        generator.load_state_dict(ckpt['generator'])
        discriminator.load_state_dict(ckpt['discriminator'])
        g_running.load_state_dict(ckpt['g_running'])
        g_optimizer.load_state_dict(ckpt['g_optimizer'])
        d_optimizer.load_state_dict(ckpt['d_optimizer'])

    transform = transforms.Compose(
        [
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.ImageNormalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ]
    )

    if args.sched:
        args.lr = {128: 0.0015, 256: 0.002, 512: 0.003, 1024: 0.003}
        args.batch = {4: 512, 8: 256, 16: 128, 32: 64, 64: 32, 128: 32, 256: 32}

    else:
        args.lr = {}
        args.batch = {}

    args.gen_sample = {512: (8, 4), 1024: (4, 2)}

    args.batch_default = 32

    train(args, transform, generator, discriminator)
