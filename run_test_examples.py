import traceback
from time import time
from IFQRE.quick_start import run_IFQRE
from argparse import ArgumentParser



test_examples = {
    'Test MF': {
        'model': 'MF',
        'dataset': 'ml-100k',
    },
    'Test NeuMF': {
        'model': 'NeuMF',
        'dataset': 'ml-100k',
    },
    'Test LightGCN': {
        'model': 'LightGCN',
        'dataset': 'ml-100k',
    },

}


def run_test_examples():
    parser = ArgumentParser()
    parser.add_argument('--model', '-m', type=str, default='MF', help='name of models')
    parser.add_argument('--dataset', '-d', type=str, default='ml-100k', help='name of datasets')
    parser.add_argument('--epochs', '-e', type=int, default=30, help='num of running epochs')
    parser.add_argument('--learning_rate', '-r', type=float, default=0.001, help='learning rate')
    parser.add_argument('--batch_size', '-b', type=int, default=512, help='batch size')
    args = parser.parse_args()
    example = 'Test '
    example = example + args.model
    config_dict = test_examples[example]
    config_dict['dataset'] = args.dataset
    config_dict['epochs'] = args.epochs
    config_dict['learning_rate'] = args.learning_rate
    config_dict['train_batch_size'] = args.batch_size
    test_start_time = time()
    success_examples, fail_examples = [], []
    try:
        run_IFQRE(config_dict=config_dict, saved=True)
        success_examples.append(example)
    except Exception:
        print(traceback.format_exc())
        fail_examples.append(example)
    test_end_time = time()
    print('total test time: ', test_end_time - test_start_time)
    print('success examples: ', success_examples)
    print('fail examples: ', fail_examples)
    print('\n')


if __name__ == '__main__':
    run_test_examples()