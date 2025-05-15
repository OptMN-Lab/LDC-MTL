import os
import sys
import logging
import wandb
from argparse import ArgumentParser
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import random_split, DataLoader
from tqdm import trange
import copy
import time
from experiments.cityscapes.data import Cityscapes
from experiments.cityscapes.models import SegNet, SegNetMtan, FairSegNetMtan
from experiments.cityscapes.utils import ConfMatrix, delta_fn, depth_error
from experiments.utils import (
    common_parser,
    extract_weight_method_parameters_from_args,
    get_device,
    set_logger,
    set_seed,
    str2bool,
    norm_diff,
    get_model_grads,
    get_model_params,
)
from methods.weight_methods import WeightMethods

set_logger()


def calc_loss(x_pred, x_output, task_type):
    device = x_pred.device

    # binary mark to mask out undefined pixel space
    binary_mask = (torch.sum(x_output, dim=1) != 0).float().unsqueeze(1).to(device)

    if task_type == "semantic":
        # semantic loss: depth-wise cross-entropy
        loss = F.nll_loss(x_pred, x_output, ignore_index=-1)

    if task_type == "depth":
        # depth loss: l1 norm
        loss = torch.sum(torch.abs(x_pred - x_output) * binary_mask) / torch.nonzero(
            binary_mask, as_tuple=False
        ).size(0)

    return loss

def eval_smooth(w, prev_model, model, train_data, train_label, train_depth, train_aux_loader, num_pts=1):
    alpha = 0.1
    grad = eval_grad(prev_model, train_data, train_label, train_depth, train_aux_loader)
    update_size = norm_diff(get_model_params(model), \
                                  get_model_params(prev_model))
    max_smooth = -1
    new_model = copy.deepcopy(prev_model)
    
    for n, p in new_model.named_parameters():
        p.data = (1-alpha) * p.data + alpha * {n:p for n, p in model.named_parameters()}[n].data
        
    new_grad = eval_grad(new_model, train_data, train_label, train_depth, train_aux_loader)
    tensor1 = new_grad[:,1]
    tensor2 = grad[:,1]
    grad_diff = torch.linalg.vector_norm(tensor1-tensor2)
    smooth = grad_diff / (update_size * alpha)
    max_smooth = max(smooth, max_smooth)
    
    return max_smooth

def eval_grad(model, train_data, train_label, train_depth, train_aux_loader):
    train_pred, features = model(train_data, return_representation=True)
    losses = torch.stack(
        (
            calc_loss(train_pred[0], train_label, "semantic"),
            calc_loss(train_pred[1], train_depth, "depth"),
        )
    )
    #TODO: gradient norm calculation
    grad_dims = []
    for param in model.shared_parameters():
        grad_dims.append(param.data.numel())
    n_tasks = 2
    grads = torch.Tensor(sum(grad_dims), n_tasks).to(device)
    for i in range(n_tasks):
        if i < n_tasks - 1:
            losses[i].backward(retain_graph=True)
        else:
            losses[i].backward()
        # grad2vector
        grads[:,i].fill_(0.0)
        count = 0
        for param in model.shared_parameters():
            grad = param.grad
            if grad is not None:
                grad_cur = grad.data.detach().clone()
                beg = 0 if count ==0 else sum(grad_dims[:count])
                en = sum(grad_dims[:count+1])
                grads[beg: en, i].copy_(grad_cur.view(-1))
            count += 1
        # multi_task_model.zero_grad_shared_modules()
        for p in model.parameters():
            p.grad = None
    return grads


def main(path, lr, bs, device):
    # ----
    # Nets
    # ---
    model = dict(segnet=SegNet(), mtan=SegNetMtan(), ldc_mtl=FairSegNetMtan())[args.model]
    prev_model = copy.deepcopy(model)
    model = model.to(device)
    prev_model = prev_model.to(device)
    # dataset and dataloaders
    log_str = (
        "Applying data augmentation on Cityscapes."
        if args.apply_augmentation
        else "Standard training strategy without data augmentation."
    )
    logging.info(log_str)

    cityscapes_train_set = Cityscapes(
        root=path.as_posix(), train=True, augmentation=args.apply_augmentation
    )
    cityscapes_test_set = Cityscapes(root=path.as_posix(), train=False)

    train_loader = torch.utils.data.DataLoader(
        dataset=cityscapes_train_set, batch_size=bs, shuffle=True
    )
    test_loader = torch.utils.data.DataLoader(
        dataset=cityscapes_test_set, batch_size=bs, shuffle=False
    )

    # weight method
    weight_methods_parameters = extract_weight_method_parameters_from_args(args)
    weight_method = WeightMethods(
        args.method, n_tasks=2, device=device, **weight_methods_parameters[args.method]
    )

    # optimizer
    optimizer = torch.optim.Adam(
        [
            dict(params=model.parameters(), lr=lr),
        ]
    )
    
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=100, gamma=0.5)
    epochs = 200
    epoch_iter = trange(epochs)
    train_batch = len(train_loader)
    test_batch = len(test_loader)
    avg_cost = np.zeros([epochs, 12], dtype=np.float32)
    custom_step = -1
    conf_mat = ConfMatrix(model.segnet.class_nb)
    deltas = np.zeros([epochs,], dtype=np.float32)

    # some extra statistics we save during training
    loss_list = []
    norm_list = []
    

    for epoch in epoch_iter:
        cost = np.zeros(12, dtype=np.float32)
        model.train()
        time1 = time.time()
        for j, batch in enumerate(train_loader):
            custom_step += 1
            optimizer.zero_grad()
            train_data, train_label, train_depth = batch
            train_data, train_label = train_data.to(device), train_label.long().to(
                device
            )
            train_depth = train_depth.to(device)

            train_pred, features = model(train_data, return_representation=True)
            # prev_train_pred, prev_features = prev_model(train_data, return_representation=True)

            losses = torch.stack(
                (
                    calc_loss(train_pred[0], train_label, "semantic"),
                    calc_loss(train_pred[1], train_depth, "depth"),
                )
            )

            # tau = 1 with identical normalization
            weighted_loss = train_pred[2][0]*losses[0]+train_pred[2][1]*losses[1]
            diff = torch.abs(losses[0]-losses[1])
            total_loss = args.lamb * torch.sum(diff) + weighted_loss
            total_loss.backward()
            
            optimizer.step()
            loss_list.append(losses.detach().cpu())
            
            conf_mat.update(train_pred[0].argmax(1).flatten(), train_label.flatten())

            cost[0] = losses[0].item()
            cost[3] = losses[1].item()
            cost[4], cost[5] = depth_error(train_pred[1], train_depth)
            avg_cost[epoch, :6] += cost[:6] / train_batch
        time2 = time.time()
        print(f"Epoch {epoch+1} Time: {time2-time1}", flush=True)
        # epoch_iter.set_description(
        #     f"[{epoch+1}  {j+1}/{train_batch}] semantic loss: {losses[0].item():.3f}, "
        #     f"depth loss: {losses[1].item():.3f}, "
        # )

        # scheduler
        scheduler.step()
        # compute mIoU and acc
        avg_cost[epoch, 1:3] = conf_mat.get_metrics()

        # evaluating test data
        model.eval()
        conf_mat = ConfMatrix(model.segnet.class_nb)
        with torch.no_grad():  # operations inside don't track history
            test_dataset = iter(test_loader)
            for k in range(test_batch):
                test_data, test_label, test_depth = test_dataset.next()
                test_data, test_label = test_data.to(device), test_label.long().to(
                    device
                )
                test_depth = test_depth.to(device)

                test_pred = model(test_data)
                test_loss = torch.stack(
                    (
                        calc_loss(test_pred[0], test_label, "semantic"),
                        calc_loss(test_pred[1], test_depth, "depth"),
                    )
                )

                conf_mat.update(test_pred[0].argmax(1).flatten(), test_label.flatten())

                cost[6] = test_loss[0].item()
                cost[9] = test_loss[1].item()
                cost[10], cost[11] = depth_error(test_pred[1], test_depth)
                avg_cost[epoch, 6:] += cost[6:] / test_batch

            # compute mIoU and acc
            avg_cost[epoch, 7:9] = conf_mat.get_metrics()

            # Test Delta_m
            test_delta_m = delta_fn(
                avg_cost[epoch, [7, 8, 10, 11]]
            )
            deltas[epoch] = test_delta_m

            # print results
            print(
                f"\nLOSS FORMAT: SEMANTIC_LOSS MEAN_IOU PIX_ACC | DEPTH_LOSS ABS_ERR REL_ERR ", flush=True
            )
            print(
                f"Epoch: {epoch:04d} | TRAIN: {avg_cost[epoch, 0]:.4f} {avg_cost[epoch, 1]:.4f} {avg_cost[epoch, 2]:.4f} "
                f"| {avg_cost[epoch, 3]:.4f} {avg_cost[epoch, 4]:.4f} {avg_cost[epoch, 5]:.4f} | {avg_cost[epoch, 6]:.4f} "
                f"TEST: {avg_cost[epoch, 7]:.4f} {avg_cost[epoch, 8]:.4f} {avg_cost[epoch, 9]:.4f} | "
                f"{avg_cost[epoch, 10]:.4f} {avg_cost[epoch, 11]:.4f}"
                f"| {test_delta_m:.3f}", flush=True
            )

            if wandb.run is not None:
                wandb.log({"Train Semantic Loss": avg_cost[epoch, 0]}, step=epoch)
                wandb.log({"Train Mean IoU": avg_cost[epoch, 1]}, step=epoch)
                wandb.log({"Train Pixel Accuracy": avg_cost[epoch, 2]}, step=epoch)
                wandb.log({"Train Depth Loss": avg_cost[epoch, 3]}, step=epoch)
                wandb.log({"Train Absolute Error": avg_cost[epoch, 4]}, step=epoch)
                wandb.log({"Train Relative Error": avg_cost[epoch, 5]}, step=epoch)

                wandb.log({"Test Semantic Loss": avg_cost[epoch, 6]}, step=epoch)
                wandb.log({"Test Mean IoU": avg_cost[epoch, 7]}, step=epoch)
                wandb.log({"Test Pixel Accuracy": avg_cost[epoch, 8]}, step=epoch)
                wandb.log({"Test Depth Loss": avg_cost[epoch, 9]}, step=epoch)
                wandb.log({"Test Absolute Error": avg_cost[epoch, 10]}, step=epoch)
                wandb.log({"Test Relative Error": avg_cost[epoch, 11]}, step=epoch)
                wandb.log({"Test ∆m": test_delta_m}, step=epoch)



            keys = [
                "Train Semantic Loss",
                "Train Mean IoU",
                "Train Pixel Accuracy",
                "Train Depth Loss",
                "Train Absolute Error",
                "Train Relative Error",

                "Test Semantic Loss",
                "Test Mean IoU",
                "Test Pixel Accuracy",
                "Test Depth Loss",
                "Test Absolute Error",
                "Test Relative Error",
            ]

            name = f"{args.method}_sd{args.seed}"

            torch.save({
                "delta_m": deltas,
                "keys": keys,
                "avg_cost": avg_cost,
                "losses": loss_list,
                "norms": norm_list,
            }, f"./save/{name}.stats")


if __name__ == "__main__":
    parser = ArgumentParser("Cityscapes", parents=[common_parser])
    parser.set_defaults(
        data_path=os.path.join(os.getcwd(), "experiments/cityscapes/dataset/cityscapes"),
        lr=3e-4,
        n_epochs=200,
        batch_size=8,
        lamb = 0.1,
    )
    parser.add_argument(
        "--model",
        type=str,
        default="ldc_mtl",
        choices=["segnet", "mtan", "ldc_mtl"],
        help="model type",
    )
    parser.add_argument(
        "--apply-augmentation", type=str2bool, default=True, help="data augmentations"
    )
    parser.add_argument("--wandb_project", type=str, default=None, help="Name of Weights & Biases Project.")
    parser.add_argument("--wandb_entity", type=str, default=None, help="Name of Weights & Biases Entity.")
    args = parser.parse_args()

    # set seed
    set_seed(args.seed)

    if args.wandb_project is not None:
        wandb.init(project=args.wandb_project, entity=args.wandb_entity, config=args)

    device = get_device(gpus=args.gpu)
    main(path=args.data_path, lr=args.lr, bs=args.batch_size, device=device)

    if wandb.run is not None:
        wandb.finish()
