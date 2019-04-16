#!/usr/bin/python
# -*- coding: utf-8 -*-
from dependencies import *


def train_model(model, optimizer, scheduler, num_epochs=25):
    best_model_wts = copy.deepcopy(model.state_dict())
    best_dice = 0

    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)

        since = time.time()

        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:
            if phase == 'train':
                scheduler.step()
                for param_group in optimizer.param_groups:
                    print("LR", param_group['lr'])

                model.train()  # Set model to training mode
            else:
                model.eval()   # Set model to evaluate mode

            metrics = defaultdict(float)
            epoch_samples = 0
            dice = 0

            for inputs, labels in tqdm.tqdm(dataloaders[phase]):
                inputs = inputs.to(device)
                labels = labels.to(device)

                # zero the parameter gradients
                optimizer.zero_grad()
                # with SummaryWriter(comment='U-Net') as w:
                #     w.add_graph(model, (inputs,))

                # forward
                # track history if only in train
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    loss = calc_loss(outputs, labels, metrics)
                    dice += cal_dice(outputs, labels)

                    # backward + optimize only if in training phase
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                # statistics
                epoch_samples += inputs.size(0)
            metrics['dice'] = dice
            print_metrics(metrics, epoch_samples, phase)
            epoch_dice = metrics['dice'] / epoch_samples

            # deep copy the model
            if phase == 'val' and metrics['dice'] > best_dice:
                print("saving best model")
                best_dice = epoch_dice
                best_model_wts = copy.deepcopy(model.state_dict())
                torch.save(model.state_dict(), PATH_MODEL_SAVE)
                # torch.save()

        time_elapsed = time.time() - since
        print('{:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))

    print('Best val loss: {:4f}'.format(best_dice))

    # load best model weights
    model.load_state_dict(best_model_wts)
    return model
# use the same transformations for train/val in this example


if __name__ == '__main__':
    df = process_data("数据集1/*/*/*.dcm")
    df_, df_test = train_test_split(
        df, test_size=0.1, shuffle=True, random_state=SEED)
    df_train, df_val = train_test_split(
        df_, test_size=0.15, shuffle=True, random_state=SEED)
    df_train = df_train.reset_index(drop=True)
    df_val = df_val.reset_index(drop=True)
    train_set = Data(df_train['path_dcm'],
                     df_train['path_mask'], transform=transform)  # TODO
    val_set = Data(df_val['path_dcm'], df_train['path_mask'], transform=None)

    # image_datasets = {
    #     'train': train_set, 'val': val_set
    # }
    dataloaders = {
        'train': DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True, num_workers=0),
        'val': DataLoader(val_set, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
    }

    net = UNet(1, 1).to(device)

    if train:
        if TRAIN_CONTINUE:
            net.load_state_dict(torch.load(PATH_MODEL_SAVE))
            optimizer_ft = optim.Adam(
                filter(lambda p: p.requires_grad, net.parameters()), lr=1e-4)
            exp_lr_scheduler = lr_scheduler.StepLR(
                optimizer_ft, step_size=30, gamma=0.1)
            net = train_model(net, optimizer_ft,
                              exp_lr_scheduler, num_epochs=60)
        else:

            optimizer_ft = optim.Adam(
                filter(lambda p: p.requires_grad, net.parameters()), lr=1e-4)
            exp_lr_scheduler = lr_scheduler.StepLR(
                optimizer_ft, step_size=30, gamma=0.1)
            net = train_model(net, optimizer_ft,
                              exp_lr_scheduler, num_epochs=60)
    else:
        net.load_state_dict(torch.load(
            PATH_MODEL_TEST))
        metrics = defaultdict(float)
        with torch.no_grad():
            net.eval()
            dice = 0
            epoch_samples = 0
            for inputs, labels in tqdm.tqdm(dataloaders['val']):
                inputs = inputs.to(device)
                labels = labels.to(device)

                # forward
                # track history if only in train
                outputs = net(inputs)
                dice += cal_dice(outputs, labels)

                # statistics
                epoch_samples += inputs.size(0)

                # print_metrics(metrics, epoch_samples, 'test')
            epoch_dice = dice / epoch_samples
            print('dice: {}'.format(epoch_dice))
            #
            #     outputs = net(inputs)
            #     loss = calc_loss(outputs, labels, metrics)
            #
            #     # statistics
            #     epoch_samples += inputs.size(0)
            #
            # print_metrics(metrics, epoch_samples, 'test')
            # epoch_loss = metrics['loss'] / epoch_samples
