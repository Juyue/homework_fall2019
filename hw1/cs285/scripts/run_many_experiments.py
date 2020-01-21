import os
import time
from read_events_files import analyze_experiment_result
from run_hw1_behavior_cloning import BC_Trainer

def main(params):
    ##################################
    ### CREATE DIRECTORY FOR LOGGING
    ##################################
    params['exp_name'] = 'l' + str(params['n_layers']) + 's' + str(params['size'])

    logdir_prefix = 'bc_'
    if params['do_dagger']:
        logdir_prefix = 'dagger_'
        assert params['n_iter'] > 1, ('DAGGER needs more than 1 iteration (n_iter>1) of training, to iteratively query the expert and train (after 1st warmstarting from behavior cloning).')
    else:
        params['n_iter'] ==1, ('Vanilla behavior cloning collects expert data just once (n_iter=1)')

    ## directory for logging
    data_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), '../data')
    if not (os.path.exists(data_path)):
        os.makedirs(data_path)
    logdir = logdir_prefix + params['exp_name'] + '_' + params['env_name'] + '_' + time.strftime("%d-%m-%Y_%H-%M-%S")
    logdir = os.path.join(data_path, logdir)
    params['logdir'] = logdir
    if not(os.path.exists(logdir)):
        os.makedirs(logdir)
        
    trainer = BC_Trainer(params)
    trainer.run_training_loop()

    # analyze the result.
    analyze_experiment_result(logdir)

ROOT_PATH = "D:\\Documents\\JuyueFiles\\Learning\\DRL_Homework\\hw1\\"
env_names = ['Ant-v2', 'Humanoid-v2', 'HalfCheetah-v2', 'Hopper-v2']
expert_policy_files = ['Ant.pkl', 'Humanoid.pkl', 'HalfCheetah.pkl', 'Hopper.pkl'] 
expert_data_files = ['Ant-v2', 'Humanoid-v2', 'HalfCheetah-v2', 'Hopper-v2'] 
do_daggers = [1, 0] 
batch_size = 1000
ep_len = None 
n_iter = 1000
# control the experiments you want to run.
for env_name, expert_policy_name, expert_data_name in zip(env_names[1:], expert_policy_files[1:], expert_data_files[1:]):
    for do_dagger in do_daggers:
        params = {}
        params['expert_policy_file'] = os.path.join(ROOT_PATH, "cs285\\policies\\experts", expert_policy_name)
        params['expert_data'] = os.path.join(ROOT_PATH, "cs285\\expert_data\\", 'expert_data_' + expert_data_name + '.pkl')
        params['env_name'] = env_name
        params['do_dagger'] = do_dagger
        params['ep_len'] = ep_len 
        params['n_layers'] = 2 
        params['size'] = 64 

        params['num_agent_train_steps_per_iter'] = 1000
        params['n_iter'] = n_iter
        params['batch_size'] = batch_size
        params['eval_batch_size'] = 200
        params['train_batch_size'] = 100 
        params['learning_rate'] = 5e-3
        params['video_log_freq'] = -1
        params['scalar_log_freq'] = 1 
        params['use_gpu'] = 0
        params['which_gpu'] = -1
        params['max_replay_buffer_size'] = 1000000
        params['seed'] = 1

        main(params)
