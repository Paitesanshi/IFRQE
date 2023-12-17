import logging
import time
from logging import getLogger

import torch
import pickle

from IFQRE.config import Config
from IFQRE.data import create_dataset, data_preparation, save_split_dataloaders, load_split_dataloaders
from IFQRE.trainer.trainer import GameTrainer
from IFQRE.utils import init_logger, get_model, get_trainer, init_seed, set_color

def objective_function(config_dict=None, config_file_list=None, saved=True):
    r""" The default objective_function used in HyperTuning

    Args:
        config_dict (dict, optional): Parameters dictionary used to modify experiment parameters. Defaults to ``None``.
        config_file_list (list, optional): Config files used to modify experiment parameters. Defaults to ``None``.
        saved (bool, optional): Whether to save the model. Defaults to ``True``.
    """

    config = Config(config_dict=config_dict, config_file_list=config_file_list)
    init_seed(config['seed'], config['reproducibility'])
    logging.basicConfig(level=logging.ERROR)
    dataset = create_dataset(config)
    train_data, valid_data, test_data = data_preparation(config, dataset)
    init_seed(config['seed'], config['reproducibility'])
    model = get_model(config['model'])(config, train_data.dataset).to(config['device'])
    trainer = get_trainer(config['MODEL_TYPE'], config['model'])(config, model)
    best_valid_score, best_valid_result = trainer.fit(train_data, valid_data, verbose=False, saved=saved)
    test_result = trainer.evaluate(test_data, load_best_model=saved)

    return {
        'best_valid_score': best_valid_score,
        'valid_score_bigger': config['valid_metric_bigger'],
        'best_valid_result': best_valid_result,
        'test_result': test_result
    }


def load_data_and_model(model_file):
    r"""Load filtered dataset, split dataloaders and saved model.

    Args:
        model_file (str): The path of saved model file.

    Returns:
        tuple:
            - config (Config): An instance object of Config, which record parameter information in :attr:`model_file`.
            - model (AbstractRecommender): The model load from :attr:`model_file`.
            - dataset (Dataset): The filtered dataset.
            - train_data (AbstractDataLoader): The dataloader for training.
            - valid_data (AbstractDataLoader): The dataloader for validation.
            - test_data (AbstractDataLoader): The dataloader for testing.
    """
    checkpoint = torch.load(model_file)
    config = checkpoint['config']
    init_seed(config['seed'], config['reproducibility'])
    init_logger(config)
    logger = getLogger()
    logger.info(config)

    dataset = create_dataset(config)
    logger.info(dataset)
    train_data, valid_data, test_data = data_preparation(config, dataset)

    init_seed(config['seed'], config['reproducibility'])
    model = get_model(config['model'])(config, train_data.dataset).to(config['device'])
    model.load_state_dict(checkpoint['state_dict'])
    model.load_other_parameter(checkpoint.get('other_parameter'))

    return config, model, dataset, train_data, valid_data, test_data

def run_IFQRE(model=None, dataset=None, config_file_list=None, config_dict=None, saved=True):
    r""" A fast running api, which includes the complete process of
    training and testing a model on a specified dataset

    Args:
        model (str, optional): Model name. Defaults to ``None``.
        dataset (str, optional): Dataset name. Defaults to ``None``.
        config_file_list (list, optional): Config files used to modify experiment parameters. Defaults to ``None``.
        config_dict (dict, optional): Parameters dictionary used to modify experiment parameters. Defaults to ``None``.
        saved (bool, optional): Whether to save the model. Defaults to ``True``.
    """
    # configurations initialization
    config = Config(model=model, dataset=dataset, config_file_list=config_file_list, config_dict=config_dict)
    init_seed(config['seed'], config['reproducibility'])
    # logger initialization
    init_logger(config)
    logger = getLogger()
    
    logger.info(config)

    # dataset filtering
    avg_valid_result = None
    avg_test_result = None
    #if config['method_type']=='check_T':

    dataset = create_dataset(config)
    logger.info(dataset)

    # dataset splitting
    dataset.inter_feat.sample(frac=1)
    train_data, valid_data, test_data = data_preparation(config, dataset)
    # model loading and initialization
    init_seed(config['seed'], config['reproducibility'])
    model = get_model(config['model'])(config, train_data.dataset).to(config['device'])
    logger.info(model)
    torch.save(model.state_dict(), config['model'] + config.dataset + 'model.pth')
    # trainer loading and initialization
    # trainer = get_trainer(config['MODEL_TYPE'], config['model'])(config, model)
    
    ot=list()
    gt=list()
    lossT=list()
    cntT=[0 for x in range(10)]
    train_st=time.time()
    #rates=[0.1/(x+1) for x in range(int(config['M']))]
    rates=[0.15,0.08,0.07,0.06,0.05,0.1,0.01,0.0001,0.0005,0.001]
    #rates=[0.01,0.05,0.3,0.3,0.3]
    for i in range(int(config['M'])):
        st=time.time()
        if config['method_type']=='game':
            model.distribution.init_distribution()
        trainer = GameTrainer(config, model)
        # model training
        # for i in range(config['T']):
        
        #mask_rate=rates[i]
        mask_rate=rates[i]
        ot,gt,lossT,cntT=trainer.update_distribution(train_data, valid_data,ot,gt,lossT,cntT,mask_rate, show_progress=config['show_progress'])
        if config['method_type'].find('valid_loss')!=-1:
            return
        ed=time.time()
        logger.info(set_color('update_distribution Time', 'yellow') + f': {int(ed)-int(st)}')
        best_valid_score, best_valid_result = trainer.fit(
            train_data, valid_data, saved=saved, show_progress=config['show_progress']
        )
        ed=time.time()
        logger.info(set_color('total train Time', 'yellow') + f': {int(ed)-int(st)}')
        # model evaluation
        test_result = trainer.evaluate(test_data, load_best_model=saved, show_progress=config['show_progress'])
        ed=time.time()
        for j in range(len(cntT)):
            message='Total Anchor{} used {}'.format(j+1,cntT[j])
            logger.info(message)
        logger.info(set_color('M Time', 'yellow') + f': {int(ed)-int(st)}')
        logger.info(set_color('Total Time', 'yellow') + f': {int(ed)-int(train_st)}')
        if i == 0:
            avg_valid_result = best_valid_result
            avg_test_result = test_result
        else:
            for k in best_valid_result.keys():
                avg_valid_result[k] += best_valid_result[k]
                avg_test_result[k] += test_result[k]
        logger.info(set_color('best valid ', 'yellow') + f': {best_valid_result}')
        logger.info(set_color('test result', 'yellow') + f': {test_result}')

    for k in avg_valid_result.keys():
        avg_valid_result[k] /= int(config['M'])
        avg_test_result[k] /= int(config['M'])

    logger.info(set_color('avg valid ', 'yellow') + f': {avg_valid_result}')
    logger.info(set_color('avg result', 'yellow') + f': {avg_test_result}')

    return {
        'best_valid_score': best_valid_score,
        'valid_score_bigger': config['valid_metric_bigger'],
        'avg_valid_result': avg_valid_result,
        'avg_result': avg_test_result
    }


def tune_run_IFQRE(config=None, train_data=None, valid_data=None, test_data=None, logger=None, saved=True):
    r""" A fast running api, which includes the complete process of
    training and testing a model on a specified dataset

    Args:
        model (str, optional): Model name. Defaults to ``None``.
        dataset (str, optional): Dataset name. Defaults to ``None``.
        config_file_list (list, optional): Config files used to modify experiment parameters. Defaults to ``None``.
        config_dict (dict, optional): Parameters dictionary used to modify experiment parameters. Defaults to ``None``.
        saved (bool, optional): Whether to save the model. Defaults to ``True``.
    """
    # configurations initialization
    # config = Config(model=model, dataset=dataset, config_file_list=config_file_list, config_dict=config_dict)
    # init_seed(config['seed'], config['reproducibility'])
    # # logger initialization
    # init_logger(config)
    # logger = getLogger()
    #
    # logger.info(config)
    #
    # # dataset filtering
    # dataset = create_dataset(config)
    # logger.info(dataset)

    # dataset splitting
    # train_data, valid_data, test_data = data_preparation(config, dataset)

    # model loading and initialization
    init_seed(config['seed'], config['reproducibility'])
    model = get_model(config['model'])(config, train_data.dataset).to(config['device'])
    logger.info(model)

    # trainer loading and initialization
    # trainer = get_trainer(config['MODEL_TYPE'], config['model'])(config, model)
    trainer = GameTrainer(config, model)
    # model training
    # for i in range(config['T']):
    st = time.time()
    trainer.update_distribution(train_data, valid_data, show_progress=config['show_progress'])
    ed = time.time()
    print((ed - st) / 60)
    best_valid_score, best_valid_result = trainer.fit(
        train_data, valid_data, saved=saved, show_progress=config['show_progress']
    )

    # model evaluation
    test_result = trainer.evaluate(test_data, load_best_model=saved, show_progress=config['show_progress'])

    logger.info(set_color('best valid ', 'yellow') + f': {best_valid_result}')
    logger.info(set_color('test result', 'yellow') + f': {test_result}')

    return {
        'best_valid_score': best_valid_score,
        'valid_score_bigger': config['valid_metric_bigger'],
        'best_valid_result': best_valid_result,
        'test_result': test_result
    }
