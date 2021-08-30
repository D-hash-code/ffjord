import argparse
import os
import time
import numpy as np
import pandas as pd

import torch
import torch.optim as optim
import torchvision.datasets as dset
import torchvision.transforms as tforms
from torchvision.utils import save_image

###### DISCRIMINATOR IMPORTS #########
import logging

import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

from lib.networks import Discriminator

######################################

import lib.layers as layers
import lib.utils as utils
import lib.odenvp as odenvp
import lib.multiscale_parallel as multiscale_parallel

from train_misc import standard_normal_logprob
from train_misc import set_cnf_options, count_nfe, count_parameters, count_total_time
from train_misc import add_spectral_norm, spectral_norm_power_iteration
from train_misc import create_regularization_fns, get_regularization, append_regularization_to_log


'''
python train_cnf_gan.py
--ganify True
--learning_objective hybrid
--colab_mode True


> Arguments to configure
--dims 
--strides
--num_blocks
--conv 
--layer_type
--divergence_fn
--nonlinearity
--solver
--atol
--rtol
--step_size
--test_solver
--test_atol
--test_rtol
--image_size
--alpha
--time_length
--train_T
--num_epochs
--batch_size
--batch_size_schedule
--test_batch_size
--lr
--warmup_iters
--weight_decay
--spectral_norm_niter
--add_noise
--batch_norm
--residual
--autoencode
--rademacher
--spectral_norm
--multiscale
--parralel
--l1int
--l2int
--dl2int
--JFrobint
--JdiagFrobint
--JoffdiagFrobint
--time_penalty
--max_grad_norm
--begin_epoch
--resume
--save
--val_freq
--log_freq


>Arguments to Add
- n_critics - number of discriminator iterations
- f-divergence
- prior for generator
- alpha value for applying logits
- learning rate decay rate
- minimum learning rate allowed on decay
- regularisation parameter for adversarial training
- (real_nvp or nice)
- number of layers: number of units between input and output in the m function for coupling layer
- hidden layers: size of hidden layers (only applicable for nice)
- like-reg: regulizing factor for likelihood vs adversarial losses for hybrid
- df_dim: dimension depth for discriminator


'''
# go fast boi!!
torch.backends.cudnn.benchmark = True

## Args Parser
if True:
    SOLVERS = ["dopri5", "bdf", "rk4", "midpoint", 'adams', 'explicit_adams']
    parser = argparse.ArgumentParser("Continuous Normalizing Flow")
    parser.add_argument("--ganify",type=eval, default=True,choices=[True,False])
    parser.add_argument("--learning_objective",choices=['adversarial','hybrid','max_likelihood'],type=str,default='adversarial')
    parser.add_argument("--colab_mode",type=eval, default=False,choices=[True,False])
    parser.add_argument("--data", choices=["mnist", "svhn", "cifar10", 'lsun_church'], type=str, default="mnist")
    parser.add_argument("--dims", type=str, default="64,64,64")
    parser.add_argument("--strides", type=str, default="1,1,1,1")
    parser.add_argument("--num_blocks", type=int, default=2, help='Number of stacked CNFs.')

    parser.add_argument("--conv", type=eval, default=True, choices=[True, False])
    parser.add_argument(
        "--layer_type", type=str, default="concat",
        choices=["ignore", "concat", "concat_v2", "squash", "concatsquash", "concatcoord", "hyper", "blend"]
    )
    parser.add_argument("--divergence_fn", type=str, default="approximate", choices=["brute_force", "approximate"])
    parser.add_argument(
        "--nonlinearity", type=str, default="softplus", choices=["tanh", "relu", "softplus", "elu", "swish"]
    )
    parser.add_argument('--solver', type=str, default='dopri5', choices=SOLVERS)
    parser.add_argument('--atol', type=float, default=1e-5)
    parser.add_argument('--rtol', type=float, default=1e-5)
    parser.add_argument("--step_size", type=float, default=None, help="Optional fixed step size.")

    parser.add_argument('--test_solver', type=str, default=None, choices=SOLVERS + [None])
    parser.add_argument('--test_atol', type=float, default=None)
    parser.add_argument('--test_rtol', type=float, default=None)

    parser.add_argument("--imagesize", type=int, default=None)
    parser.add_argument("--alpha", type=float, default=1e-6)
    parser.add_argument('--time_length', type=float, default=1.0)
    parser.add_argument('--train_T', type=eval, default=True)

    parser.add_argument("--num_epochs", type=int, default=1000)
    parser.add_argument("--batch_size", type=int, default=200)
    parser.add_argument(
        "--batch_size_schedule", type=str, default="", help="Increases the batchsize at every given epoch, dash separated."
    )
    parser.add_argument("--test_batch_size", type=int, default=200)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--warmup_iters", type=float, default=1000)
    parser.add_argument("--weight_decay", type=float, default=0.0)
    parser.add_argument("--spectral_norm_niter", type=int, default=10)

    parser.add_argument("--add_noise", type=eval, default=True, choices=[True, False])
    parser.add_argument("--batch_norm", type=eval, default=False, choices=[True, False])
    parser.add_argument('--residual', type=eval, default=False, choices=[True, False])
    parser.add_argument('--autoencode', type=eval, default=False, choices=[True, False])
    parser.add_argument('--rademacher', type=eval, default=True, choices=[True, False])
    parser.add_argument('--spectral_norm', type=eval, default=False, choices=[True, False])
    parser.add_argument('--multiscale', type=eval, default=True, choices=[True, False])
    parser.add_argument('--parallel', type=eval, default=False, choices=[True, False])

    # Regularizations
    parser.add_argument('--l1int', type=float, default=None, help="int_t ||f||_1")
    parser.add_argument('--l2int', type=float, default=None, help="int_t ||f||_2")
    parser.add_argument('--dl2int', type=float, default=None, help="int_t ||f^T df/dt||_2")
    parser.add_argument('--JFrobint', type=float, default=None, help="int_t ||df/dx||_F")
    parser.add_argument('--JdiagFrobint', type=float, default=None, help="int_t ||df_i/dx_i||_F")
    parser.add_argument('--JoffdiagFrobint', type=float, default=None, help="int_t ||df/dx - df_i/dx_i||_F")

    parser.add_argument("--time_penalty", type=float, default=0, help="Regularization on the end_time.")
    parser.add_argument(
        "--max_grad_norm", type=float, default=1e10,
        help="Max norm of graidents (default is just stupidly high to avoid any clipping)"
    )

    parser.add_argument("--begin_epoch", type=int, default=1)
    parser.add_argument("--resume", type=str, default=None)
    parser.add_argument("--save", type=str, default="experiments/cnf")
    parser.add_argument("--val_freq", type=int, default=1)
    parser.add_argument("--log_freq", type=int, default=10)

    args = parser.parse_args()


## File Logger
if True:
    unique_file_code = np.random.choice(100000)
    # logger
    if args.colab_mode:
        args.save = '/content/drive/MyDrive/Thesis_Colab_Files/results'
    utils.makedirs(args.save)
    logger = utils.get_logger(logpath=os.path.join(args.save, f'{unique_file_code}logs'), filepath=os.path.abspath(__file__))

    if args.layer_type == "blend":
        logger.info("!! Setting time_length from None to 1.0 due to use of Blend layers.")
        args.time_length = 1.0

    logger.info(args)


def add_noise(x):
    """
    [0, 1] -> [0, 255] -> add noise -> [0, 1]
    """
    if args.add_noise:
        noise = x.new().resize_as_(x).uniform_()
        x = x * 255 + noise
        x = x / 256
    return x


def update_lr(optimizer, itr):
    iter_frac = min(float(itr + 1) / max(args.warmup_iters, 1), 1.0)
    lr = args.lr * iter_frac
    for param_group in optimizer.param_groups:
        param_group["lr"] = lr


def get_train_loader(train_set, epoch):
    if args.batch_size_schedule != "":
        epochs = [0] + list(map(int, args.batch_size_schedule.split("-")))
        n_passed = sum(np.array(epochs) <= epoch)
        current_batch_size = int(args.batch_size * n_passed)
    else:
        current_batch_size = args.batch_size
    train_loader = torch.utils.data.DataLoader(
        dataset=train_set, batch_size=current_batch_size, shuffle=True, drop_last=True, pin_memory=True
    )
    logger.info("===> Using batch size {}. Total {} iterations/epoch.".format(current_batch_size, len(train_loader)))
    return train_loader


def get_dataset(args):
    trans = lambda im_size: tforms.Compose([tforms.Resize(im_size), tforms.ToTensor(), add_noise])

    if args.data == "mnist":
        im_dim = 1
        im_size = 28 if args.imagesize is None else args.imagesize
        train_set = dset.MNIST(root="./data", train=True, transform=trans(im_size), download=True)
        test_set = dset.MNIST(root="./data", train=False, transform=trans(im_size), download=True)

    data_shape = (im_dim, im_size, im_size)
    if not args.conv: ## What is this for ??
        data_shape = (im_dim * im_size * im_size,)

    test_loader = torch.utils.data.DataLoader(
        dataset=test_set, batch_size=args.test_batch_size, shuffle=False, drop_last=True
    )
    return train_set, test_loader, data_shape


def compute_bits_per_dim(x, model):
    zero = torch.zeros(x.shape[0], 1).to(x)

    # Don't use data parallelize if batch size is small.
    # if x.shape[0] < 200:
    #     model = model.module

    z, delta_logp = model(x, zero)  # run model forward

    logpz = standard_normal_logprob(z).view(z.shape[0], -1).sum(1, keepdim=True)  # logp(z)
    logpx = logpz - delta_logp

    logpx_per_dim = torch.sum(logpx) / x.nelement()  # averaged over batches
    bits_per_dim = -(logpx_per_dim - np.log(256)) / np.log(2)

    return bits_per_dim


def create_model(args, data_shape, regularization_fns):
    hidden_dims = tuple(map(int, args.dims.split(",")))
    strides = tuple(map(int, args.strides.split(",")))

    if args.multiscale:
        model = odenvp.ODENVP(
            (args.batch_size, *data_shape),
            n_blocks=args.num_blocks,
            intermediate_dims=hidden_dims,
            nonlinearity=args.nonlinearity,
            alpha=args.alpha,
            cnf_kwargs={"T": args.time_length, "train_T": args.train_T, "regularization_fns": regularization_fns},
        )
    elif args.parallel:
        model = multiscale_parallel.MultiscaleParallelCNF(
            (args.batch_size, *data_shape),
            n_blocks=args.num_blocks,
            intermediate_dims=hidden_dims,
            alpha=args.alpha,
            time_length=args.time_length,
        )
    else:
        def build_cnf():
                diffeq = layers.ODEnet(
                    hidden_dims=hidden_dims,
                    input_shape=data_shape,
                    strides=strides,
                    conv=args.conv,
                    layer_type=args.layer_type,
                    nonlinearity=args.nonlinearity,
                )
                odefunc = layers.ODEfunc(
                    diffeq=diffeq,
                    divergence_fn=args.divergence_fn,
                    residual=args.residual,
                    rademacher=args.rademacher,
                )
                cnf = layers.CNF(
                    odefunc=odefunc,
                    T=args.time_length,
                    train_T=args.train_T,
                    regularization_fns=regularization_fns,
                    solver=args.solver,
                )
                return cnf

        chain = [layers.LogitTransform(alpha=args.alpha)] if args.alpha > 0 else [layers.ZeroMeanTransform()]
        chain = chain + [build_cnf() for _ in range(args.num_blocks)]
        if args.batch_norm:
            chain.append(layers.MovingBatchNorm2d(data_shape[0]))
        model = layers.SequentialFlow(chain)
    return model


if __name__ == "__main__":

    # get device
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    cvt = lambda x: x.type(torch.float32).to(device, non_blocking=True)

    # load dataset
    train_set, test_loader, data_shape = get_dataset(args)

    # build model
    regularization_fns, regularization_coeffs = create_regularization_fns(args)
    model = create_model(args, data_shape, regularization_fns)

    if args.spectral_norm: add_spectral_norm(model, logger)
    set_cnf_options(args, model)

    logger.info(model)
    logger.info("Number of trainable parameters: {}".format(count_parameters(model)))

    # optimizer
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    if torch.cuda.is_available():
        model = torch.nn.DataParallel(model).cuda()

    # For visualization.
    fixed_z = cvt(torch.randn(100, *data_shape))

    time_meter = utils.RunningAverageMeter(0.97)
    loss_meter = utils.RunningAverageMeter(0.97)
    steps_meter = utils.RunningAverageMeter(0.97)
    grad_meter = utils.RunningAverageMeter(0.97)
    tt_meter = utils.RunningAverageMeter(0.97)

    if args.spectral_norm and not args.resume: spectral_norm_power_iteration(model, 500)

    ####### INITIALISE DISCRIMINATOR #######

    ## Create Discriminator Model
    if args.ganify:
        n_input_channels= 1 ## For grayscale = 1, for rgb = 3
        n_features = 32 ## Number of features to be used in discriminator
        netD = Discriminator(n_input_channels, n_features).to(device)

        ## loss function
        criterion = nn.BCELoss()

        ## Discriminator optimizer
        discriminator_lr = 0.0002 ## Discriminator learning rate
        optimizerD = optim.Adam(netD.parameters(), lr=discriminator_lr,weight_decay=args.weight_decay)

        ## Initialise other variables
        real_label=1
        fake_label=0

    ########################################
    best_loss = float("inf")
    with torch.no_grad():
        fig_filename = os.path.join(args.save, f"{unique_file_code}figs", "before.jpg")
        utils.makedirs(os.path.dirname(fig_filename))
        generated_samples = model(fixed_z, reverse=True).view(-1, *data_shape)
        save_image(generated_samples, fig_filename, nrow=10)
    train_hist_df = pd.DataFrame(columns=['Epoch','Step','D_loss','G_loss','D(x)','D(G(x1))','D(G(x2))','ll_loss','ll_loss_av','grad_norm','grad_norm_av'])

    test_hist_df = pd.DataFrame(columns=['Epoch','D_loss','G_loss','D(x)','D(G(x1))','D(G(x2))','ll_loss'])
    itr = 0
    for epoch in range(args.begin_epoch, args.num_epochs + 1):
        model.train() ## Set model to train model
        if args.ganify: netD.train()
        train_loader = get_train_loader(train_set, epoch)

        if epoch == args.begin_epoch:num_batches = len(train_loader)
        
        for _, (x, y) in enumerate(train_loader): ## is x,y a single image? or a batch?
            # cast data and move to device
            x = cvt(x)
            

            #######- Adversarial Training-#########

            ## Train Discriminator
            if args.ganify:
    
                bs = x.shape[0]

                netD.zero_grad()
                label = torch.full((bs,),real_label,device=device,dtype=torch.float32)
                
                #Real Training
                output = netD(x)
                lossD_real = criterion(output,label)
                lossD_real.backward()
                D_x = output.mean().item()

                #Fake Training
                noise = cvt(torch.randn(200, *data_shape))
                fake_images = model(noise, reverse=True).view(-1, *data_shape)
                label.fill_(fake_label)
                output = netD(fake_images.detach())
                lossD_fake = criterion(output,label)
                lossD_fake.backward()
                D_G_z1 = output.mean().item()
                lossD = lossD_real + lossD_fake ## For printing purposes ??
                grad_norm = torch.nn.utils.clip_grad_norm_(netD.parameters(), args.max_grad_norm)
                optimizerD.step()

                start = time.time()
                ## Train Generator
                optimizer.zero_grad()
                label.fill_(real_label)
                output = netD(fake_images)
                lossG = criterion(output,label)
                lossG.backward()
                D_G_z2 = output.mean().item()
                grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
                optimizer.step()
                if args.hybrid:
                    #update_lr(optimizer, itr)
                    start = time.time()
                    optimizer.zero_grad() ## Set gradients to zero before
                    
                    # compute loss
                    loss = compute_bits_per_dim(x, model)
                    if regularization_coeffs:
                        reg_states = get_regularization(model, regularization_coeffs)
                        reg_loss = sum(
                            reg_state * coeff for reg_state, coeff in zip(reg_states, regularization_coeffs) if coeff != 0
                        )
                        loss = loss + reg_loss
                    total_time = count_total_time(model)
                    loss = loss + total_time * args.time_penalty

                    loss.backward()
                    grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
                    optimizer.step()
            
            #########################################
            else:
                update_lr(optimizer, itr)
                start = time.time()
                optimizer.zero_grad() ## Set gradients to zero before
                
                # compute loss
                loss = compute_bits_per_dim(x, model)
                if regularization_coeffs:
                    reg_states = get_regularization(model, regularization_coeffs)
                    reg_loss = sum(
                        reg_state * coeff for reg_state, coeff in zip(reg_states, regularization_coeffs) if coeff != 0
                    )
                    loss = loss + reg_loss
                total_time = count_total_time(model)
                loss = loss + total_time * args.time_penalty

                loss.backward()
                grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
                optimizer.step()

            if args.spectral_norm and not args.ganify: spectral_norm_power_iteration(model, args.spectral_norm_niter)

            ## Add logging tool
            if args.ganify and not args.hybrid:
                pass
            else:
                time_meter.update(time.time() - start)
                loss_meter.update(loss.item())
                steps_meter.update(count_nfe(model))
                grad_meter.update(grad_norm)
                tt_meter.update(total_time)

            if itr % args.log_freq == 0:
                if args.ganify:
                    log_message = ('Epoch [{}/{}], step [{}/{}], d_loss: {:.4f}, g_loss: {:.4f}, D(x): {:.2f}, Discriminator - D(G(x)): {:.2f}, Generator - D(G(x)): {:.2f}'.format(epoch, args.num_epochs, 
                                                            itr, num_batches, lossD.item(), lossG.item(), D_x, D_G_z1, D_G_z2)
                    )
                    logger.info(log_message)
                    if args.hybrid:
                        log_message = ('Bit/dim {:.4f}({:.4f}) | Grad Norm {:.4f}({:.4f})'.format(loss_meter.val, loss_meter.avg,grad_meter.val, grad_meter.avg))
                    logger.info(log_message)
                    train_hist_df = train_hist_df.append({'Epoch':epoch,'Step':itr,'D_loss':lossD.item(), 'G_loss':lossG.item(), 'D(x)':D_x, 'D(G(x1))':D_G_z1, 'D(G(x2))':D_G_z2, 'll_loss':loss_meter.val, 'll_loss_av':loss_meter.avg,'grad_norm':grad_meter.val, 'grad_norm_av':grad_meter.avg},ignore_index=True)
                    utils.makedirs(args.save)
                    train_hist_df.to_csv(os.path.join(args.save, f'{unique_file_code}train_hist.csv'))

                else:
                    log_message = (
                        "Iter {:04d} | Time {:.4f}({:.4f}) | Bit/dim {:.4f}({:.4f}) | "
                        "Steps {:.0f}({:.2f}) | Grad Norm {:.4f}({:.4f}) | Total Time {:.2f}({:.2f})".format(
                            itr, time_meter.val, time_meter.avg, loss_meter.val, loss_meter.avg, steps_meter.val,
                            steps_meter.avg, grad_meter.val, grad_meter.avg, tt_meter.val, tt_meter.avg
                        )
                    )
                    if regularization_coeffs:
                        log_message = append_regularization_to_log(log_message, regularization_fns, reg_states)
                    logger.info(log_message)

            if itr%50==0:
                with torch.no_grad():
                    fig_filename = os.path.join(args.save, f"{unique_file_code}_iter_figs", "epoch{:04d}_iter{:04d}.jpg".format(epoch,itr))
                    utils.makedirs(os.path.dirname(fig_filename))
                    generated_samples = model(fixed_z, reverse=True).view(-1, *data_shape)
                    save_image(generated_samples, fig_filename, nrow=10)



            itr += 1

        # compute test loss
        itr=0
        if args.ganify:
            model.eval()
            netD.eval()
            if epoch % args.val_freq == 0:
                with torch.no_grad():
                    start = time.time()
                    logger.info("validating...")
                    d_losses = []
                    g_losses = []
                    ll_losses = []
                    for (x, y) in test_loader:
                        x = cvt(x)
                        ll_loss = compute_bits_per_dim(x, model)
                        ll_losses.append(ll_loss.item())
                        
                        bs = x.shape[0]
                        label = torch.full((bs,),real_label,device=device,dtype=torch.float32)
                        
                        #Real Training
                        output = netD(x)
                        lossD_real = criterion(output,label)
                        D_x = output.mean().item()

                        #Fake Training
                        noise = cvt(torch.randn(200, *data_shape))
                        fake_images = model(noise, reverse=True).view(-1, *data_shape)
                        label.fill_(fake_label)
                        output = netD(fake_images.detach())
                        lossD_fake = criterion(output,label)
                        D_G_z1 = output.mean().item()

                        lossD = lossD_real + lossD_fake ## For printing purposes ??

                        ## Train Generator

                        label.fill_(real_label)
                        output = netD(fake_images)
                        lossG = criterion(output,label)
                        D_G_z2 = output.mean().item()
                        d_losses.append(lossD.item())
                        g_losses.append(lossG.item())


                    ll_loss = np.mean(ll_losses)
                    d_loss = np.mean(d_losses)
                    g_loss = np.mean(g_losses)
                    logger.info("Epoch {:04d} | Time {:.4f}, Log-Likelihood Bit/dim {:.4f}, Discriminator Loss {:.4f}, Generator Loss {:.4f}".format(epoch, time.time() - start, ll_loss,d_loss,g_loss))

                    test_hist_df = test_hist_df.append({'Epoch':epoch,'D_loss':d_loss, 'G_loss':g_loss, 'D(x)':D_x, 'D(G(x1))':D_G_z1, 'D(G(x2))':D_G_z2, 'll_loss':ll_loss},ignore_index=True)
                    utils.makedirs(args.save)
                    test_hist_df.to_csv(os.path.join(args.save, f'{unique_file_code}test_hist.csv'))
                    if ll_loss < best_loss:
                        best_loss = ll_loss
                        utils.makedirs(args.save)
                        torch.save({
                            "args": args,
                            "state_dict": model.module.state_dict() if torch.cuda.is_available() else model.state_dict(),
                            "optim_state_dict": optimizer.state_dict(),
                        }, os.path.join(args.save, f"{unique_file_code}checkpt.pth"))
        else:
            model.eval()
            if epoch % args.val_freq == 0:
                with torch.no_grad():
                    start = time.time()
                    logger.info("validating...")
                    losses = []
                    for (x, y) in test_loader:
                        if not args.conv:
                            x = x.view(x.shape[0], -1)
                        x = cvt(x)
                        loss = compute_bits_per_dim(x, model)
                        losses.append(loss)

                    loss = np.mean(losses)
                    logger.info("Epoch {:04d} | Time {:.4f}, Bit/dim {:.4f}".format(epoch, time.time() - start, loss))
                    if loss < best_loss:
                        best_loss = loss
                        utils.makedirs(args.save)
                        torch.save({
                            "args": args,
                            "state_dict": model.module.state_dict() if torch.cuda.is_available() else model.state_dict(),
                            "optim_state_dict": optimizer.state_dict(),
                        }, os.path.join(args.save, f'checkpt.pth'))

        ## Save loss and Accuracy

        
        # visualize samples and density
        with torch.no_grad():
            fig_filename = os.path.join(args.save, f"{unique_file_code}figs", "{:04d}.jpg".format(epoch))
            utils.makedirs(os.path.dirname(fig_filename))
            generated_samples = model(fixed_z, reverse=True).view(-1, *data_shape)
            save_image(generated_samples, fig_filename, nrow=10)
