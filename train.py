import os
import torch
from utils.pair_dataset import pairDataset
from utils.abstract_dataset import DeepfakeAbstractBaseDataset
from utils.tools import setup_logger, _seed_everything, save_checkpoint, AverageMeter, evaluate
import random
import yaml
import argparse
from trainer import Trainer

# config
gpu_ids = [0]
_seed_everything()
loss_freq = 40
parser = argparse.ArgumentParser()
parser.add_argument('--detector_path', type=str, 
            default='./config/dpl.yaml',
            help='path to detector YAML file')

def prepare_training_data(config):
    train_set = pairDataset(config)  # Only use the pair dataset class in training
    train_data_loader = \
        torch.utils.data.DataLoader(
            dataset=train_set,
            batch_size=config['train_batchSize'] // 2,
            shuffle=True, 
            num_workers=int(config['workers']),
            drop_last=True,
            collate_fn=train_set.collate_fn,
            )
    return train_data_loader

def prepare_valid_data(config):
    def get_valid_data_loader(config, valid_name):
        # update the config dictionary with the specific valid dataset
        config = config.copy()  # create a copy of config to avoid altering the original one
        config['valid_dataset'] = [valid_name]  # specify the current valid dataset
        valid_set = DeepfakeAbstractBaseDataset(
                config=config,
                mode='val', 
            )
        valid_data_loader = \
            torch.utils.data.DataLoader(
                dataset=valid_set, 
                batch_size=config['test_batchSize'],
                shuffle=False, 
                num_workers=int(config['workers']),
                collate_fn=valid_set.collate_fn,
            )
        return valid_data_loader

    valid_data_loaders = {}
    for one_valid_name in config['valid_dataset']:
        valid_data_loaders[one_valid_name] = get_valid_data_loader(config, one_valid_name)
    return valid_data_loaders[list(valid_data_loaders.keys())[0]]

if __name__ == "__main__":
    opt = parser.parse_args()
    with open(opt.detector_path, 'r') as f:
        config = yaml.safe_load(f)
    input_size = config['resolution']
    batch_size = config['train_batchSize']
    max_epoch = config['nEpochs']
    dataset_path_prefix = config['dataset_path_prefix']
    quality_genres_num = config['quality_genres_num']
    compress_num = config['compress_num']
    train_stage = config['train_stage']
    checkpoint_path = config['checkpoint_path']
    # create logger
    logger_path = config['log_dir']
    os.makedirs(logger_path, exist_ok=True)
    logger = setup_logger(logger_path, config['running_name'] + f'_training_Stage{train_stage}.log', 'logger')

    # prepare the training data loader
    train_data_loader = prepare_training_data(config)
    len_dataloader = train_data_loader.__len__()
    # prepare the valid data loader
    valid_data_loader = prepare_valid_data(config)
    # train
    model = Trainer(gpu_ids, quality_genres_num, train_stage)
    if train_stage == 2:
        checkpoint = torch.load(checkpoint_path)
        model.model.load_state_dict(checkpoint['model'], strict=True)
        if checkpoint['policy'] is not None:
            model.model.fsm.policy.load_state_dict(checkpoint['policy'], strict=True)
            model.model.fsm.policy_old.load_state_dict(checkpoint['policy'], strict=True)
    model.total_steps = 0
    epoch = 0
    valid_count = 0
    while epoch < max_epoch:
        logger.info(f'No {epoch}')
        for iteration, data_dict in enumerate(train_data_loader):
            model.total_steps += 1
            # get data
            data_list, label = data_dict['image'], data_dict['label']
            # manually shuffle
            idx = list(range(data_list[0].shape[0]))
            random.shuffle(idx)
            for index in range(len(data_list)):
                data_list[index] = data_list[index][idx]
            label = label[idx]
            for index in range(len(data_list)):
                data_list[index] = data_list[index].detach()
            label = label.detach()
            model.set_input(data_list, label.repeat(compress_num))
            reward_list = [AverageMeter() for _ in range(quality_genres_num - 1)]
            if train_stage == 1:
                loss1, loss_cls, entropy_loss = model.optimize_weight(reward_list, epoch)
                if model.total_steps % loss_freq == 0:
                    logger.info(f'loss1: {loss1}, loss_cls: {loss_cls}, entropy_loss: {entropy_loss} at step: {model.total_steps}')
            elif train_stage == 2:
                loss2, loss_actor, loss_critic, _reward, reward_w = model.optimize_weight(reward_list, epoch)
                if model.total_steps % loss_freq == 0:
                    logger.info(f'loss2: {loss2}, loss_actor: {loss_actor}, loss_critic: {loss_critic}, reward: {_reward}, reward_w: {reward_w} at step: {model.total_steps}')
            
            # valid
            if (iteration + 1) % (len_dataloader // config['valid_time']) == 0:
                valid_count += 1
                model.model.eval()
                if train_stage != 1:
                    model.model.fsm.policy.eval()
                    model.model.fsm.policy_old.eval()

                auc, acc = evaluate(model, valid_data_loader)
                logger.info(f'(Val @ epoch {epoch}, valid_count {valid_count}) auc: {auc}, acc:{acc}')

                model.model.train()
                if train_stage != 1:
                    model.model.fsm.policy.train()
                    model.model.fsm.policy_old.train()
                
                save_checkpoint({
                    'model': model.model.state_dict(),
                    'policy': model.model.fsm.policy.state_dict() if model.model.fsm else None
                }, f'./Result/weights/bz16_LC_v1_epoch{epoch}_validcount{valid_count}_weight_Stage{train_stage}.pth')

        epoch += 1
                