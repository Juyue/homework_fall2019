import os
import tensorflow as tf
from matplotlib import pyplot as plt

def parse_filename(filepath):
    fullfpath = os.path.split(filepath)[0].split(os.path.sep)
    return fullfpath

def analyze_experiment_result(logdir):
    for filename in os.listdir(logdir):
        if filename.startswith('events.out.'):
            filepath = os.path.join(logdir, filename)
            break
    output_result = collect_experiment_result(filepath)
    plot_experiment_result(output_result)
    
    fig_path = parse_filename(filepath)
    plt.savefig(fig_path[-1])

def plot_experiment_result(output_result):
    fig, ax = plt.subplots(3, 2)
    max_step = len(output_result['loss'])
    xaxis = list(range(max_step))
    ax[0][0].plot(xaxis, output_result['train_mean'])
    ax[0][0].title.set_text('return (train mean)')
    ax[0][1].plot(xaxis, output_result['train_std'])
    ax[0][1].title.set_text('return (train std)')
    ax[1][0].plot(xaxis, output_result['eval_mean'])
    ax[1][0].title.set_text('return (eval mean)')
    ax[1][1].plot(xaxis, output_result['eval_std'])
    ax[1][1].title.set_text('return (eval std)')
    ax[2][0].plot(xaxis, output_result['loss'])
    ax[2][0].title.set_text('loss')

def collect_experiment_result(filepath):
    result = {'Train_AverageReturn': {}, 'Train_StdReturn':{}, 'Eval_AverageReturn':{}, 'Eval_StdReturn':{}, 'Train_Loss_mean':{}}
    for summary in tf.train.summary_iterator(filepath):
        for tag in result:
            if summary.summary.value and summary.summary.value[0].tag == tag:
                result[tag][summary.step] = summary.summary.value[0].simple_value

    # turn it into a list.
    max_step = max(result['Train_AverageReturn'].keys())
    def dict2list(metrics, max_step):
        return [result[metrics][i] for i in range(max_step + 1)]
    eval_mean_return = dict2list('Eval_AverageReturn', max_step) 
    eval_std_return = dict2list('Eval_StdReturn', max_step) 
    train_mean_return = dict2list('Train_AverageReturn', max_step) 
    train_std_return = dict2list('Train_StdReturn', max_step) 
    loss_val = dict2list('Train_Loss_mean', max_step)

    output_result = {'eval_mean': eval_mean_return, 
                     'eval_std':eval_std_return, 
                     'train_mean':train_mean_return, 
                     'train_std':train_std_return, 
                     'loss':loss_val}
    return output_result

filepath = os.path.join('D:\Documents\JuyueFiles\Learning\DRL_Homework\hw1\cs285\data\dagger_test_dagger_ant_Ant-v2_20-01-2020_14-58-21')
analyze_experiment_result(filepath)