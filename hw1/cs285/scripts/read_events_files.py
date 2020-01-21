import os
import tensorflow as tf
filepath = os.path.join('D:\Dropbox\Deep_Reinforcement_Learning\DRL_Homework\hw1\cs285\data\dagger_test_dagger_ant_Ant-v2_20-01-2020_14-58-21\events.out.tfevents.1579550301.DESKTOP-RDUNHDS')

def collect_experiment_result(filepath)
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

    output_result = {'eval_mean': eval_mean_return, 'eval_std':eval_std_return, 'train_mean':train_mean_return, 'train_std':train_std_return, 'loss':loss_val}
    return output_result