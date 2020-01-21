set ROOTPATH=D:\Dropbox\Deep_Reinforcement_Learning\
python cs285/scripts/run_hw1_behavior_cloning.py --expert_policy_file cs285/policies/experts/Ant.pkl --env_name Ant-v2 --exp_name test_bc_ant --n_iter 100 --do_dagger --expert_data cs285/expert_data/expert_data_Ant-v2.pkl --video_log_freq -1
