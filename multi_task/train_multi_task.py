import sys
import torch
import click
import json
import datetime
from timeit import default_timer as timer

import numpy as np
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torch.utils import data
import torchvision
import types
import itertools
import random
from tqdm import tqdm
from tensorboardX import SummaryWriter

import losses
import datasets
import metrics
import model_selector
from min_norm_solvers import MinNormSolver, gradient_normalizers




def write_file(paraString, values) -> object:
    os.makedirs('./out/out', exist_ok=True)
    np.savetxt('./out/out/' + paraString, values, fmt='%.5f')


@click.command()
@click.option('--param_file', default='params.json', help='JSON parameters file')
def train_multi_task(param_file):
    with open('configs.json') as config_params:
        configs = json.load(config_params)

    with open(param_file) as json_params:
        params = json.load(json_params)

    total_values_L = []
    total_values_R = []

    exp_identifier = []
    for (key, val) in params.items():
        if 'tasks' in key:
            continue
        exp_identifier+= ['{}={}'.format(key,val)]

    exp_identifier = '_'.join(exp_identifier)
    params['exp_id'] = exp_identifier
    paraString = str(params['exp_id'] + ".txt")

    writer = SummaryWriter(log_dir='runs/0907_test'.format(str(params['exp_id'])))

    train_loader, train_dst, val_loader, val_dst, train_loader_2,val_loader_2 = datasets.get_dataset(params, configs)
    loss_fn = losses.get_loss(params)
    metric = metrics.get_metrics(params)

    model = model_selector.get_model(params)
    model_params = []
    for m in model:
        model_params += model[m].parameters()

    optimizer = torch.optim.SGD(model_params, lr=params['lr'])

    tasks = params['tasks']
    all_tasks = configs[params['dataset']]['all_tasks']
    print('Starting training with parameters \n \t{} \n'.format(str(params)))

    if 'mgda' in params['algorithm']:
        approximate_norm_solution = params['use_approximation']
        if approximate_norm_solution:
            print('Using approximate min-norm solver')
        else:
            print('Using full solver')
    n_iter = 0
    loss_init = {}
    NUM_ITERATIONS = params['iteration']
    for iteration in tqdm(range(NUM_ITERATIONS)):
        iteration_values = []
        iteration_values_L = []
        iteration_values_R = []
        start = timer()
        print('Epoch {} Started'.format(iteration))


        for m in model:
            model[m].train()

        #for batch in train_loader:
        random_index = random.randint(0, len(train_loader) - 1)
        batch = next(itertools.islice(train_loader, random_index, None))


        if 'VR' in params['optimizer']:
            if iteration % 20 == 0:
                random_index = random.randint(0, len(train_loader) - 1)
                batch = next(itertools.islice(train_loader, random_index, None))
            else:
                random_index = random.randint(0, len(train_loader) - 1)
                batch = next(itertools.islice(train_loader_2, random_index, None))


        n_iter += 1
        # First member is always images
        images = batch[0]
        images = Variable(images.cuda())

        labels = {}
        # Read all targets of all tasks
        for i, t in enumerate(all_tasks):
            if t not in tasks:
                continue
            labels[t] = batch[i+1]
            labels[t] = Variable(labels[t].cuda())

        # Scaling the loss functions based on the algorithm choice
        loss_data = {}
        loss_data_old = {}
        grads = {}
        grads_batch_old={}
        grads_vr_new = {}

        scale = {}
        mask = None
        masks = {}

        # Will use our MGDA_UB if approximate_norm_solution is True. Otherwise, will use MGDA

        if approximate_norm_solution:
            optimizer.zero_grad()
            # First compute representations (z)
            images_volatile = Variable(images.data, volatile=True)
            rep, mask = model['rep'](images_volatile, mask)
            # As an approximate solution we only need gradients for input
            if isinstance(rep, list):
                # This is a hack to handle psp-net
                rep = rep[0]
                rep_variable = [Variable(rep.data.clone(), requires_grad=True)]
                list_rep = True
            else:
                rep_variable = Variable(rep.data.clone(), requires_grad=True)
                list_rep = False

            # Compute gradients of each loss function wrt z
            for t in tasks:
                optimizer.zero_grad()
                out_t, masks[t] = model[t](rep_variable, None)
                loss = loss_fn[t](out_t, labels[t])
                loss_data[t] = loss.data
                loss.backward()
                grads[t] = []
                if list_rep:
                    grads[t].append(Variable(rep_variable[0].grad.data.clone(), requires_grad=False))
                    rep_variable[0].grad.data.zero_()
                else:
                    grads[t].append(Variable(rep_variable.grad.data.clone(), requires_grad=False))
                    rep_variable.grad.data.zero_()
            if 'VR' in params['optimizer'] or 'VRM' in params['optimizer'] or 'VRMP' in params['optimizer'] or 'VRP' in params['optimizer']:
                if iteration%3==0:
                    grads_vr_old= grads
                    model_old=model
                else:
                    for t in tasks:
                        optimizer.zero_grad()
                        out_t, masks[t] = model_old[t](rep_variable, None)
                        loss_old = loss_fn[t](out_t, labels[t])
                        loss_data_old[t] = loss_old.data
                        loss_old.backward()
                        grads_batch_old[t] = []
                        if list_rep:
                            grads_batch_old[t].append(Variable(rep_variable[0].grad.data.clone(), requires_grad=False))
                            rep_variable[0].grad.data.zero_()
                        else:
                            grads_batch_old[t].append(Variable(rep_variable.grad.data.clone(), requires_grad=False))
                            rep_variable.grad.data.zero_()
                    for t in tasks:
                        grads_vr_new[t] = []
                        grads_vr_new[t]=grads_vr_old[t] + grads[t]+(-1 * grads_batch_old[t])
                    grads_vr_old = grads_vr_new
                    model_old = model
                    grads = grads_vr_new

        else:
            # This is MGDA
            for t in tasks:
                # Comptue gradients of each loss function wrt parameters
                optimizer.zero_grad()
                rep, mask = model['rep'](images, mask)
                out_t, masks[t] = model[t](rep, None)
                loss = loss_fn[t](out_t, labels[t])
                loss_data[t] = loss.data
                loss.backward()
                grads[t] = []
                for param in model['rep'].parameters():
                    if param.grad is not None:
                        grads[t].append(Variable(param.grad.data.clone(), requires_grad=False))



        # Normalize all gradients, this is optional and not included in the paper.
        gn = gradient_normalizers(grads, loss_data, params['normalization_type'])
        for t in tasks:
            for gr_i in range(len(grads[t])):
                grads[t][gr_i] = grads[t][gr_i] / gn[t]

        # Frank-Wolfe iteration to compute scales.
        sol, min_norm = MinNormSolver.find_min_norm_element([grads[t] for t in tasks])
        for i, t in enumerate(tasks):
            scale[t] = float(sol[i])


        # Scaled back-propagation
        optimizer.zero_grad()
        rep, _ = model['rep'](images, mask)
        for i, t in enumerate(tasks):
            out_t, _ = model[t](rep, masks[t])
            loss_t = loss_fn[t](out_t, labels[t])
            loss_data[t] = loss_t.data
            if i > 0:
                loss = loss + scale[t]*loss_t
            else:
                loss = scale[t]*loss_t
        loss.backward()
        #optimizer.step()

        lr = params['lr']



        if 'MOCO' in params['optimizer'] or 'VRM' in params['optimizer'] or 'VRMP' in params['optimizer']:
            old_parameters_dict = {}
            for idx, param in enumerate(model_params):  # Enumerate to get index
                # Use index as key to get corresponding old parameter from the dictionary
                old_parameter = old_parameters_dict.get(idx, None)

                if old_parameter is None:  # If this is the first iteration
                    old_parameters_dict[idx] = param.data.clone()  # Store a clone of the data, not the tensor itself

                if param.grad is not None:
                    # Get old parameter corresponding to current parameter using index as key
                    old_parameter = old_parameters_dict[idx]

                    # Check if the shapes match to avoid size mismatch errors
                    if old_parameter.shape != param.data.shape:
                        raise ValueError(f"Size mismatch between old and new parameter at index {idx}")

                    # Update parameter data
                    param.data = param.data - lr * param.grad.data + 0.1 * ( old_parameter- param.data )

                    # Update old parameter data for the next iteration
                    old_parameters_dict[idx] = param.data.clone()  # Store a clone of the updated data



        if 'VR' in params['optimizer'] or 'MSGDA' in params['optimizer'] or 'MGDA' in params['optimizer'] or 'VRP' in params['optimizer']:
            for param in model_params:
                if param.grad is not None:
                    param.data =param.data- lr * param.grad.data






        writer.add_scalar('training_loss', loss.data, n_iter)
        for t in tasks:
            writer.add_scalar('training_loss_{}'.format(t), loss_data[t], n_iter)

        for m in model:
            model[m].eval()

        tot_loss = {}
        tot_loss['all'] = 0.0
        met = {}
        for t in tasks:
            tot_loss[t] = 0.0
            met[t] = 0.0

        num_val_batches = 0
        for batch_val in val_loader:
            val_images = Variable(batch_val[0].cuda(), volatile=True)
            labels_val = {}

            for i, t in enumerate(all_tasks):
                if t not in tasks:
                    continue
                labels_val[t] = batch_val[i+1]
                labels_val[t] = Variable(labels_val[t].cuda(), volatile=True)

            val_rep, _ = model['rep'](val_images, None)
            for t in tasks:
                out_t_val, _ = model[t](val_rep, None)
                loss_t = loss_fn[t](out_t_val, labels_val[t])
                tot_loss['all'] += loss_t.data
                tot_loss[t] += loss_t.data
                metric[t].update(out_t_val, labels_val[t])
            num_val_batches+=1

        for t in tasks:
            writer.add_scalar('validation_loss_{}'.format(t), tot_loss[t]/num_val_batches, n_iter)
            metric_results = metric[t].get_result()
            for metric_key in metric_results:
                writer.add_scalar('metric_{}_{}'.format(metric_key, t), metric_results[metric_key], n_iter)
            metric[t].reset()
        writer.add_scalar('validation_loss', tot_loss['all']/len(val_dst), n_iter)
        iteration_values_L.append(tot_loss['L']/len(val_dst))
        iteration_values_R.append(tot_loss['R']/len(val_dst))

        total_values_L.append(iteration_values_L)
        total_values_R.append(iteration_values_R)





        end = timer()
        print('Epoch ended in {}s'.format(end - start))
        os.makedirs('./out/out_L', exist_ok=True)
        os.makedirs('./out/out_R', exist_ok=True)

        # total_values_result=[i[0].cpu().numpy() for i in total_values]
        total_values_result_L = [i[0].cpu().numpy() for i in total_values_L]
        total_values_result_R = [i[0].cpu().numpy() for i in total_values_R]

        # np.savetxt('./out/out/' + paraString, total_values_result ,fmt='%.5f')
        np.savetxt('./out/out_L/' + paraString, total_values_result_L, fmt='%.5f')
        np.savetxt('./out/out_R/' + paraString, total_values_result_R, fmt='%.5f')
        end = timer()
        # print(total_values)
        # print(np.array(total_values[0]))
        print('Epoch ended in {}s'.format(end - start))
        print(tot_loss['all'] / len(val_dst))
        print(tot_loss)


if __name__ == '__main__':
    train_multi_task()
