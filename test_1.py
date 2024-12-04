from utils.tools import evaluate, setup_logger, _seed_everything
import torch
import yaml
from utils.abstract_dataset import DeepfakeAbstractBaseDataset
from trainer import Trainer

_seed_everything()
gpu_ids = [0]
ckpt_path = './Result/test_result'

def prepare_testing_data(config):
    def get_test_data_loader(config, test_name):
        # update the config dictionary with the specific testing dataset
        config = config.copy()  # create a copy of config to avoid altering the original one
        config['val_dataset'] = [test_name]  # specify the current test dataset
        test_set = DeepfakeAbstractBaseDataset(
                config=config,
                mode='val', 
            )
        test_data_loader = \
            torch.utils.data.DataLoader(
                dataset=test_set, 
                batch_size=config['eval_batchSize'],
                shuffle=False, 
                num_workers=int(config['workers']),
                collate_fn=test_set.collate_fn,
            )
        return test_data_loader

    test_data_loaders = {}
    for one_test_name in config['val_dataset']:
        test_data_loaders[one_test_name] = get_test_data_loader(config, one_test_name)
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
    # prepare the testing data loader
    test_data_loaders = prepare_testing_data(config)

    model = Trainer(gpu_ids, quality_genres_num, train_stage)
    for i in range(1, 6):
        save_dir = f'./Result/weights/bz16_LC_v1_epoch{i - 1}_validcount{i}_weight_Stage{train_stage}.pth'
        checkpoint = torch.load(save_dir)
        model.model.load_state_dict(checkpoint['model'], strict=True)
        if train_stage != 1:
            model.model.fsm.policy.load_state_dict(checkpoint['policy'], strict=True)
            model.model.fsm.policy_old.load_state_dict(model.model.fsm.policy.state_dict())
            model.model.fsm.policy.eval()
            model.model.fsm.policy_old.eval()

        model.model.eval()

        logger.info(f'start Test({running_name}_epoch{i-1}_validcount{i}_weight_Stage{train_stage})')

        # testing for all test data
        keys = test_data_loaders.keys()
        for key in keys:
            auc, acc_rf = evaluate(model, test_data_loaders[key])
            logger.info(f'(Test on {key}) auc: {auc}, acc: {acc_rf}')