#!/usr/bin/python
# -*- coding: utf-8 -*-


from model.unet_2 import *
from preprocessing import *


class Data(Dataset):

    def __init__(self, dcm_series, labels, transform=None):
        self.transorm = transform
        # self.df = df
        self.dcm_series = dcm_series
        self.labels = labels

    def __len__(self):
        return len(self.dcm_series)

    def __getitem__(self, item):
        # X = torch.FloatTensor(np.expand_dims(np.clip((itk_read(
        #     (self.dcm_series[item])) - MIN_BOUND) / (MAX_BOUND - MIN_BOUND), 0.,
        #                                              1.) , 0))
        # X = np.clip((itk_read(
        #     (self.dcm_series[item])) - MIN_BOUND) / (MAX_BOUND - MIN_BOUND), 0.,
        #                                              1.)
        #
        # y = np.array(read_mask(self.labels[item]))/255
        # Generate data
        # X, y = self.__data_generation(list_IDs_temp)

        X = torch.FloatTensor(np.expand_dims(np.clip((itk_read(
            (self.dcm_series[item])) - MIN_BOUND) / (MAX_BOUND - MIN_BOUND), 0.,
            1.), 0))

        y = torch.FloatTensor(np.expand_dims(
            np.array(read_mask(self.labels[item])) / 255, 0))
        if self.transorm is not None:

            seed = np.random.rand()
            # random.seed(seed)
            # tmp = self.transorm(image=X, mask=y)
            # X = tmp['image']
            # y = tmp['mask']
            # random.seed(seed)
            # y = self.transorm(y)
            random.seed(seed)
            X = self.transorm(X)
            random.seed(seed)
            y = self.transorm(y)
        # else:
        #     X, y = torch.FloatTensor(X), torch.FloatTensor(y)
        return X, y


def train_model(model, optimizer, scheduler, num_epochs=25):
    best_model_wts = copy.deepcopy(model.state_dict())
    best_loss = 1e10
    iter_count = 1

    #
    # loss_win = viz.line(np.arange(10))
    # acc_win = viz.line(X=np.column_stack((np.array(0), np.array(0))),
    #                    Y=np.column_stack((np.array(0), np.array(0))))
    #
    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)

        since = time.time()

        # Each epoch has a training and validation phase
        for phase in (['train', 'val']):
            if phase == 'train':
                # scheduler.step()
                for param_group in optimizer.param_groups:
                    print("LR", param_group['lr'])

                model.train()  # Set model to training mode
            else:
                model.eval()   # Set model to evaluate mode

            metrics = defaultdict(float)
            epoch_samples = 0

            for inputs, labels in tqdm.tqdm(dataloaders[phase]):
                inputs = inputs.to(device)
                labels = labels.to(device)
                # if phase == 'train':
                #     for param_group in optimizer.param_groups:
                #         param_group['lr'] = scheduler(iter_count)
                #         iter_count += 1
                optimizer.zero_grad()  # TODO

                # zero the parameter gradients

                # with SummaryWriter(comment='U-Net') as w:
                #     w.add_graph(model, (inputs,))

                # forward
                # track history if only in train
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    loss = lovasz_hinge(outputs, labels, metrics)

                    # backward + optimize only if in training phase
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                # statistics
                epoch_samples += inputs.size(0)
                # viz.line(Y=np.array([metrics['loss']]), X=np.array([iter_count]), update='replace', win=loss_win)
                # viz.line(Y=np.column_stack((np.array([tr_acc]), np.array([ts_acc]))),
                #          X=np.column_stack((np.array([iter_count]), np.array([iter_count]))),
                #          win=acc_win, update='replace',
                #          opts=dict(legned=['Train_acc', 'Val_acc']))

            print_metrics(metrics, epoch_samples, phase)
            epoch_loss = metrics['loss'] / epoch_samples

            # deep copy the model
            if phase == 'val' and epoch_loss < best_loss:
                print("saving best model")
                best_loss = epoch_loss
                best_model_wts = copy.deepcopy(model.state_dict())
                torch.save(model.state_dict(), 'model.h5')
                # torch.save()

        time_elapsed = time.time() - since
        print('{:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))

    print('Best val loss: {:4f}'.format(best_loss))

    # load best model weights
    model.load_state_dict(best_model_wts)
    return model


# use the same transformations for train/val in this example
trans = transforms.Compose([
    transforms.ToPILImage(),
    # transforms.RandomCrop(),
    transforms.RandomApply([transforms.RandomAffine(
        90., [0.2, 0.2], scale=(0.9, 1.15))], 0.8),
    # transforms.Lambda()
    transforms.RandomVerticalFlip(0.5),
    transforms.RandomHorizontalFlip(0.5),
    # transforms.TenCrop(), # TODO
    # transforms.RandomRotation(90.),
    transforms.ToTensor()
    # transforms.ToTensor(),
    # transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])  # imagenet
])


if __name__ == '__main__':
    df = process_data("数据集1/*/*/*.dcm")
    df_train, df_val = train_test_split(df, test_size=0.15)
    df_train = df_train.reset_index(drop=True)
    df_val = df_val.reset_index(drop=True)
    train_set = Data(df_train['path_dcm'], df_train['path_mask'], trans)
    val_set = Data(df_val['path_dcm'], df_train['path_mask'])

    # image_datasets = {
    #     'train': train_set, 'val': val_set
    # }
    dataloaders = {
        'train': DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True, num_workers=0),
        'val': DataLoader(val_set, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
    }

    net = UNet().to(device)
    # net = ResNetUNet(1).to(device)
    # summary(net, input_size=(1, 512, 512))
    # dummy_input = Variable(torch.randn(1, 1, 512, 512))
    # y = net(dummy_input)
    # from torchviz import make_dot
    # g = make_dot(y)
    # g.view()

    if train:

        optimizer_ft = optim.Adam(
            filter(lambda p: p.requires_grad, net.parameters()), lr=1e-4)
        # lr_scheduler.LambdaLR(optimizer_ft, )
        # exp_lr_scheduler = lr_scheduler.StepLR(optimizer_ft, step_size=30, gamma=0.1)
        CYCLE = 6000
        LR_INIT = 0.1
        LR_MIN = 0.001
        exp_lr_scheduler = lr_scheduler.StepLR(
            optimizer_ft, step_size=30, gamma=0.1)

        def scheduler(x): return ((LR_INIT - LR_MIN) / 2) * \
            (np.cos(np.pi * (np.mod(x - 1, CYCLE) / (CYCLE))) + 1) + LR_MIN
        net = train_model(net, optimizer_ft, exp_lr_scheduler,
                          num_epochs=60)  # TODO
    else:
        net.load_state_dict(torch.load('/home/zhang/下载/model_1.h5'))
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
