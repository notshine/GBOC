import pandas as pd
import os
import argparse
import numpy as np
from torch.backends import cudnn
from sklearn.preprocessing import MinMaxScaler
from evaluation.metrics import get_metrics
from utils.slidingWindows import find_length_rank
from HP_list import Optimal_Uni_algo_HP_dict,Optimal_Multi_algo_HP_dict
import random, argparse, time, os, re, logging

from model_wrapper import *
from data_func import *

import time
import torch
import random
import numpy as np
import warnings
warnings.filterwarnings('ignore')

# seeding
seed = 2024
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
np.random.seed(seed)
random.seed(seed)
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True

print("CUDA available: ", torch.cuda.is_available())
print("cuDNN version: ", torch.backends.cudnn.version())


def configure_logger(filename):
    logger = logging.getLogger('my_logger')
    logger.setLevel(logging.INFO)

    file_handler = logging.FileHandler(filename)
    file_handler.setLevel(logging.INFO)

    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)

    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    file_handler.setFormatter(formatter)
    console_handler.setFormatter(formatter)

    logger.addHandler(file_handler)
    logger.addHandler(console_handler)

    return logger


def mkdir(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)

def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


def append_result_csv(args, evaluation_result, csv_path='result.csv'):
    dataset_name = os.path.splitext(os.path.basename(args.filename))[0]
    row = {
        'dataset': dataset_name,
        'win_size': args.win_size,
        'lr': args.lr,
        'hidden_dim': args.hidden_dim,
        'num_layers': args.num_layers,
        'alpha': args.alpha,
        'epochs': args.epochs,
    }
    row.update(evaluation_result)

    new_df = pd.DataFrame([row])
    if os.path.exists(csv_path) and os.path.getsize(csv_path) > 0:
        old_df = pd.read_csv(csv_path)
        all_cols = list(dict.fromkeys(list(old_df.columns) + list(new_df.columns)))
        old_df = old_df.reindex(columns=all_cols)
        new_df = new_df.reindex(columns=all_cols)
        result_df = pd.concat([old_df, new_df], ignore_index=True)
    else:
        result_df = new_df

    result_df.to_csv(csv_path, index=False)

def main(args):
    
    data, label = load_dataset(args)

    slidingWindow = find_length_rank(data, rank=1)

    train_index = int(args.filename.split('.')[0].split('_')[-3])
    data_train = data[:train_index, :]

    print(">>>", np.shape(data), train_index)

    # ===============================================================
    if "TSB-AD-U" in args.data_direc:
        Optimal_Det_HP = Optimal_Uni_algo_HP_dict['GBOC']
    else:
        Optimal_Det_HP = Optimal_Multi_algo_HP_dict['GBOC']
    
    # If parameters are specified on the command line, the configuration in HP_list.py will be overridden.
    if args.win_size is not None:
        Optimal_Det_HP['win_size'] = args.win_size
    if args.lr is not None:
        Optimal_Det_HP['lr'] = args.lr
    if args.hidden_dim is not None:
        Optimal_Det_HP['hidden_dim'] = args.hidden_dim
    if args.num_layers is not None:
        Optimal_Det_HP['num_layers'] = args.num_layers
    if args.alpha is not None:
        Optimal_Det_HP['alpha'] = args.alpha
    if args.epochs is not None:
        Optimal_Det_HP['epochs'] = args.epochs

    start_time = time.time()

    output = run_GBOC(data_train, data, **Optimal_Det_HP)

    end_time = time.time()
    run_time = end_time - start_time

    if isinstance(output, np.ndarray):
        logging.info(f'Success at {args.filename} using {args.AD_Name} | Time cost: {run_time:.3f}s at length {len(label)}')
        np.save(args.target_dir+'/'+args.filename.split('.')[0]+'.npy', output)
    else:
        logger.error(f'At {args.filename}: '+output)

    output = MinMaxScaler(feature_range=(0,1)).fit_transform(output.reshape(-1,1)).ravel()
    evaluation_result = get_metrics(output, label, slidingWindow=slidingWindow, pred=output > (np.mean(output)+3*np.std(output)))
    print('Evaluation Result: ', evaluation_result)

    list_w = list(evaluation_result.values())
    print(list_w)

    # visualization
    from plot_ab_sco import AnomalyScoreVisualizer, interactive_plot
    
    # Prepare data
    if "TSB-AD-U" in args.data_direc:
        index = 0
    else:
        index = 0
    
    if len(data.shape) > 1:
        data_to_plot = data[:, index]
    else:
        data_to_plot = data
    
    # Create save path
    save_path = os.path.join("plots_results", f"{args.filename}_{args.AD_Name}")
    
    # Create visualizer
    visualizer = AnomalyScoreVisualizer(data_to_plot, output, label, save_path)
    
    # Select visualization method:
    # 1. Interactive plot
    interactive_plot(data_to_plot, output, label, save_path)
    
    # 2. Local detailed plot (e.g., index 2000-4000)
    # visualizer.plot_local_detail(0, 1750)
    
    # 3. Downsampled plot
    # visualizer.plot_downsampled(step=50)

    return evaluation_result, list_w, run_time


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--AD_Name', type=str, default='GBOC', help='anomaly detection algorithm name')
    parser.add_argument('--data_direc', type=str, default='TSB-AD-M')
    parser.add_argument('--filename', type=str, default='166_SMAP_id_23_Sensor_tr_1113_1st_1890')
    parser.add_argument('--score_dir', type=str, help='path to save the anomaly score', default='score')
    parser.add_argument('--save_dir', type=str, help='path to save the evaluation result', default='eval')
    parser.add_argument('--save', type=bool, help='whether to save the evaluation result',  default=True)
    
    # GBOC model parameters - optional, if not specified, the configuration in HP_list.py will be used
    parser.add_argument('--win_size', type=int, default=None, help='sliding window size')
    parser.add_argument('--lr', type=float, default=None, help='learning rate')
    parser.add_argument('--hidden_dim', type=int, default=None, help='hidden dimension')
    parser.add_argument('--num_layers', type=int, default=None, help='number of LSTM layers')
    parser.add_argument('--alpha', type=float, default=None, help='alpha for loss combination')
    parser.add_argument('--epochs', type=int, default=None, help='number of training epochs')
     
    args = parser.parse_args()

    if "TSB-AD-U" in args.data_direc:
        args.score_dir = os.path.join(args.score_dir, "uni")
        args.save_dir = os.path.join(args.save_dir, "uni")
    else:
        args.score_dir = os.path.join(args.score_dir, "multi")
        args.save_dir = os.path.join(args.save_dir, "multi")

    target_dir = os.path.join(args.score_dir, args.AD_Name)
    os.makedirs(target_dir, exist_ok = True)
    os.makedirs(args.save_dir, exist_ok = True)
    logger = configure_logger(filename=f'{target_dir}/029_run_{args.AD_Name}.log') 
    args.target_dir = target_dir

    print(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()))
    print('================ Hyperparameters ===============')
    for k, v in sorted(vars(args).items()):
        print('%s: %s' % (str(k), str(v)))
    print('====================  Train  ===================')

    if (not os.path.exists("plots_data")):
        mkdir("plots_data")
    if (not os.path.exists(os.path.join("plots_results", f"{args.AD_Name}"))):
        mkdir(os.path.join("plots_results", f"{args.AD_Name}"))

    if os.path.exists(f'{args.save_dir}/{args.AD_Name}.csv'):
        df = pd.read_csv(f'{args.save_dir}/{args.AD_Name}.csv')
        write_csv =  df.values.tolist()
    else:
        write_csv = []

    evaluation_result, list_w, run_time = main(args)

    # Save a run summary: dataset name + 6 model parameters + metric results.
    append_result_csv(args, evaluation_result, csv_path='result.csv')

    list_w.insert(0, run_time)
    list_w.insert(0, args.filename)
    write_csv.append(list_w)

    col_w = list(evaluation_result.keys())
    col_w.insert(0, 'Time')
    col_w.insert(0, 'file')
    w_csv = pd.DataFrame(write_csv, columns=col_w)
    w_csv.to_csv(f'{args.save_dir}/{args.AD_Name}.csv', index=False)

