function [vote] = compare_vote(test_results_folder, ori_ldpc_code_folder, run_time, ldpc_len, test_dataset, class_order, sd)

if test_dataset == 'yaleB'
    class_nums = 38;
elseif genuine_dataset == 'FEI'
    class_nums = 200;
elseif genuine_dataset == 'PIE'
    class_nums = 68;
end


ori_ldpc_code_path   = fullfile(ori_ldpc_code_folder, strcat(test_dataset, '_ori_ldpc_code_', num2str(ldpc_len), '.mat'));
dec_mat_file_folder = fullfile(test_results_folder,'dec_mat_file', run_time);


dec_mat_file_name = strcat(run_time, '-', num2str(sd),'.mat');
dec_mat_file_path = fullfile(dec_mat_file_folder, dec_mat_file_name);
vote_folder = fullfile(test_results_folder,'vote', run_time);
if ~exist(vote_folder,'dir')
    mkdir(vote_folder)
end
vote_file_path = fullfile(vote_folder, strcat(run_time,'-',num2str(sd),'-vote.mat'));

load(dec_mat_file_path);
load(ori_ldpc_code_path)

img_idx=1;
co = 1;
extended_img_nums = 64;
test_total_nums = size(preLabel,1);
test_img_nums = int16(test_total_nums / extended_img_nums);
vote=zeros(test_img_nums, class_nums);
for class_order_idx = 1:class_nums
    class_id = class_order(class_order_idx);
    if class_id == 11
        ori_nums_per_class = 50;
    elseif class_id == 12
        ori_nums_per_class = 49;
    elseif class_id == 13
        ori_nums_per_class = 50;
    elseif class_id == 15
        ori_nums_per_class = 53;
    elseif class_id == 16
        ori_nums_per_class = 52;
    elseif class_id == 17
        ori_nums_per_class = 53;
    elseif class_id == 18
        ori_nums_per_class = 53;
    else
        ori_nums_per_class = 54;
    end
    for j = 1 : ori_nums_per_class
        for k = 1 : extended_img_nums
            pre_label = preLabel(img_idx,:); % pre_label
            img_idx = img_idx+1;
            for class_idx = 1 : class_nums %
                true_label = ori_ldpc_code(class_order(class_idx),:);
                b = isequal(true_label(1,:), pre_label(1,:));
                if b == 1
                    vote(co, class_idx) = vote(co, class_idx) + 1;
                end
            end
        end
        co = co + 1;
    end
end
vote = vote / extended_img_nums;
save(vote_file_path, 'vote');
fprintf('compare and vote success\n')

end

