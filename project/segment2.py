#!/usr/bin/python
# -*- coding: utf-8 -*-

# import os
# import sys
# parentdir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
# sys.path.append(parentdir)
from traning.lr import *
from librarys import *


from models.Unet34 import Unet_scSE_hyper as Net

def train_model(model, optimizer, scheduler, epoch, num_epochs=30, batchs=None, best_dice=None):
    # best_model_wts = copy.deepcopy(model.state_dict())

    while epoch <= num_epochs:
    # for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs))
        print('-' * 20)

        since = time.time()
        for phase in ['train', 'val']:
            if phase == 'train':
                # TODO
                if scheduler is not None:
                    # lr = scheduler(epoch)
                    lr = scheduler(batchs)

                    change_learning_rate(optimizer, lr)
                    lr = get_learning_rate(optimizer)
                #
                    print("LR: ", lr)

                model.train()  # Set model to training mode
                epoch += 1
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

                with torch.set_grad_enabled(phase == 'train'):
                    logits = model(inputs)
                    # TODO
                    # loss = model.criterion3(logits, labels)
                    loss = model.criterion3(logits, labels) + model.focal_loss(logits, labels, 1.0, 0.5, 0.25)
                    # loss = calc_loss(logits, labels, metrics)
                    dice += cal_dice(logits, labels)
                    metrics['loss'] += loss.detach().item() * labels.size(0)

                    # backward + optimize only if in training phase
                    if phase == 'train':
                        # TODO
                        # if scheduler is not None:
                        #     lr = scheduler(batchs)
                        #     adjust_learning_rate(optimizer, lr)


                        batchs += 1
                        loss.backward()
                        optimizer.step()

                epoch_samples += inputs.size(0)

            metrics['dice'] = dice
            print_metrics(metrics, epoch_samples, phase)


            if phase == 'val':
                # torch.save(net.state_dict(), PATH_CHECKPOINT + 'checkpoint_%d_model.pth' % epoch)
                # torch.save({
                #     'optimizer': optimizer.state_dict(),
                #     'batchs': batchs,
                #     'epoch': epoch,
                #     'best_dice': max([best_dice, (metrics['dice'] / epoch_samples)])
                # }, PATH_CHECKPOINT + 'checkpoint_%d_optim.pth' % epoch)
                if (metrics['dice'] / epoch_samples) > best_dice:
                    print("saving best model")
                    best_dice = metrics['dice'] / epoch_samples

                    torch.save( model.state_dict(), PATH_MODEL_BEST)

        time_elapsed = time.time() - since
        print('{:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))

    print('Best val loss: {:4f}'.format(best_dice))

def prepare_dataLoader():
    df = process_data("数据集1/*/*/*.dcm")
    df_, df_test = train_test_split(
        df, test_size=0.1, shuffle=True, random_state=SEED)
    df_train, df_val = train_test_split(
        df_, test_size=0.1, shuffle=True, random_state=SEED)
    df_train = df_train.reset_index(drop=True)
    df_val = df_val.reset_index(drop=True)
    df_test = df_test.reset_index(drop=True)
    train_set = Data(df_train['path_dcm'],
                     df_train['path_mask'], transform=None)  # TODO
    val_set = Data(df_val['path_dcm'], df_val['path_mask'], transform=None)
    test_set = Data(df_test['path_dcm'], df_test['path_mask'], transform=None)

    # image_datasets = {
    #     'train': train_set, 'val': val_set
    # }
    dataloaders = {
        'train': DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True, num_workers=0),
        'val': DataLoader(val_set, batch_size=BATCH_SIZE, shuffle=True, num_workers=0),
        'test': DataLoader(test_set, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
    }
    return dataloaders

def save_test(path, predict, mask, i):
    if not os.path.exists(path):
        os.mkdir(path)
    cv2.imwrite(path+'%d_predict.png'%i, predict.astype('uint8')*255)
    cv2.imwrite(path + '%d_mask.png' % i, mask.astype('uint8')*255)


def record_count(logits, lables):
    pred, lables = logits_to_pred(logits, lables)
    return np.array([pred.any(), lables.any()])

if __name__ == '__main__':
    dataloaders = prepare_dataLoader()

    net = Net().to(device)

    if TRAIN:
        # TODO
        # scheduler = lambda x: (0.01 / 2) * (np.cos(PI * (np.mod(x - 1, 15*797) / (15*797))) + 1)
        # optimizer = optim.SGD(filter(lambda p: p.requires_grad, net.parameters()),
        #                       lr=0.01, momentum=0.9, weight_decay=0.0001)
        scheduler = lambda x: 0.0001* (0.1 **(x // 10))
        optimizer = optim.Adam(
            filter(lambda p: p.requires_grad, net.parameters()), lr=1e-4)

        if initial_checkpoint is not None:
            net.load_state_dict(torch.load(initial_checkpoint, map_location=lambda storage, loc: storage))

            checkpoint = torch.load(initial_checkpoint)
            epoch = checkpoint['epoch']
            batchs = checkpoint['batchs']
            optimizer.load_state_dict(checkpoint['optimizer'])

            # 加载best_dice
            best_dice = checkpoint['best_dice']

            train_model(net, optimizer,
                              scheduler, epoch=epoch, num_epochs=30, batchs=batchs, best_dice=best_dice)
        else:

            train_model(net, optimizer,
                              scheduler, epoch=1, num_epochs=30, batchs=1, best_dice=0)
    else:
        # TODO 测试时batchsize只能设置为1
        net.load_state_dict(torch.load(
            PATH_MODEL_TEST))
        metrics = defaultdict(float)
        with torch.no_grad():
            net.eval()
            dice, epoch_samples, record = 0, 0, np.empty((1, 2))
            # epoch_samples = 0
            # record =
            for i, (inputs, labels) in tqdm.tqdm(enumerate(dataloaders['test'])):
                # TODO
                if labels.max().item() >= 0.:


                    inputs = inputs.to(device)
                    labels = labels.to(device)

                    outputs = net(inputs)
                    dice += cal_dice(outputs, labels)
                    # save_test('./test/', outputs.sigmoid().squeeze().cpu().numpy()>0.5, labels.squeeze().cpu().numpy(), i)
                    # record = np.concatenate([record, record_count(outputs, labels)], axis=0)
                    # dice += dice_accuracy(outputs, labels, is_average=False)

                    # statistics
                    epoch_samples += inputs.size(0)

            epoch_dice = dice / epoch_samples
            print('dice: {}'.format(epoch_dice))
            pass

