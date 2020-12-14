clc, clear;
cur_folder = pwd;
index_dir = strfind(cur_folder,'\');
par_folder = cur_folder(1:index_dir(end)-1);

test_results_folder = fullfile(par_folder, 'results');
ori_ldpc_code_folder = fullfile(par_folder, 'ori_ldpc_code');
class_order_folder   = fullfile(par_folder, 'class_order');

sd = 0.2;
ldpc_len = 256;
iter_epochs = 1000;
%%%%%%%%%%%%%%% 07-29-13-20 seed=1 yaleB %%%%%%%%%%%%%%%%

run_time = '07-29-13-20-resnet_pre-codes256-epochs50-seed1-dataset-seed0';
dataset = 'yaleB';
seed = 1;
linear_map = 3.78;


class_order_path     = fullfile(class_order_folder, 'YALEB_class_order_38.mat');
load(class_order_path);
class_order = yaleB_class_order(seed+1,:); % mat index start from 1


% 1. Linear map
create_rec_linear_map(test_results_folder, run_time, dataset, ldpc_len, linear_map)
% 2. LDPC decoding using C++ and get the dec file
% ...
% ...
% 3. After ldpc decoding, execute the following code
% dec2mat(test_results_folder, run_time, ldpc_len, dataset, sd);
% vote =  compare_vote(test_results_folder, ori_ldpc_code_folder, run_time, ldpc_len, dataset, class_order, sd);
% scores_fars_gars = calc_far_gar(test_results_folder, run_time, dataset, class_order, sd, iter_epochs);

load gong
sound(y,Fs)