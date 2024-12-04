from utils.tools import evaluate, setup_logger, _seed_everything
import torch
import yaml
from utils.abstract_dataset import DeepfakeAbstractBaseDataset
from trainer import Trainer
import random

_seed_everything()
gpu_ids = [0]
ckpt_path = './Result/test_result'
compress_range_list = [(30, 99), (13, 29), (8, 12)]

def prepare_testing_data(config, jpeg_compress_factors):
    def get_test_data_loader(config, test_name, jpeg_compress_factor):
        # update the config dictionary with the specific testing dataset
        config = config.copy()  # create a copy of config to avoid altering the original one
        config['test_dataset'] = [test_name]  # specify the current test dataset
        test_set = DeepfakeAbstractBaseDataset(
                config=config,
                mode='test', 
                jpeg_compress_factor=jpeg_compress_factor
            )
        test_data_loader = \
            torch.utils.data.DataLoader(
                dataset=test_set, 
                batch_size=config['test_batchSize'],
                shuffle=False, 
                num_workers=int(config['workers']),
                collate_fn=test_set.collate_fn,
            )
        return test_data_loader

    test_data_loaders = {}
    for one_test_name in config['test_dataset']:
        for jpeg_compress_factor in jpeg_compress_factors:
            test_data_loaders[f'{one_test_name}_jpeg_factor_{jpeg_compress_factor}'] = get_test_data_loader(config, one_test_name, jpeg_compress_factor)
    return test_data_loaders

if __name__ == "__main__":

    # load config
    with open('./config/dpl.yaml', 'r') as f:
        config = yaml.safe_load(f)
    
    logger = setup_logger(ckpt_path, config['running_name']+'_test.log', 'logger')
    
    #config
    quality_genres_num = config['quality_genres_num']
    compress_num = config['compress_num']
    train_stage = config['train_stage']
    running_name = config['running_name']

    jpeg_compress_factors = [100]
    for st, en in compress_range_list:
        random_factor = random.randint(st, en)
        jpeg_compress_factors.append(random_factor)

    # prepare the testing data loader
    test_data_loaders = prepare_testing_data(config, jpeg_compress_factors)

    model = Trainer(gpu_ids, quality_genres_num, train_stage)

    save_dir = f'./Result/weights/bz16_LC_v1_epoch4_validcount5_weight_Stage1.pth'
    checkpoint = torch.load(save_dir)
    model.model.load_state_dict(checkpoint['model'], strict=True)
    if train_stage != 1:
        model.model.fsm.policy.load_state_dict(checkpoint['policy'], strict=True)
        model.model.fsm.policy_old.load_state_dict(model.model.ppo.policy.state_dict())
        model.model.fsm.policy.eval()
        model.model.fsm.policy_old.eval()

    model.model.eval()
    logger.info(f'start Test(bz16_LC_v1_epoch4_validcount5_weight_Stage1)')

    # testing for all test data
    auc_sum = 0.
    acc_sum = 0.
    single_dataset_auc_sum = 0.
    single_dataset_acc_sum = 0.
    keys = test_data_loaders.keys()
    key_idx = 0
    for idx, key in enumerate(keys):
        auc, acc_rf = evaluate(model, test_data_loaders[key])
        logger.debug(f'(Test on {key}) auc: {auc}, acc: {acc_rf}')
        auc_sum += auc
        acc_sum += acc_rf
        single_dataset_auc_sum += auc
        single_dataset_acc_sum += acc_rf
        if (idx + 1) % (len(jpeg_compress_factors)) == 0:
            dataset_name = config['test_dataset'][key_idx]
            logger.debug(f'(Test on {dataset_name}) auc: {single_dataset_auc_sum / (len(jpeg_compress_factors))}, acc: {single_dataset_acc_sum / (len(jpeg_compress_factors))}')
            single_dataset_auc_sum = 0
            single_dataset_acc_sum = 0
            key_idx += 1

    logger.debug(f'auc_avg: {auc_sum / len(keys)}, acc_avg: {acc_sum / len(keys)}')