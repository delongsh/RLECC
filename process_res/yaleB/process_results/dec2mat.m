function dec2mat(test_results_folder, run_time, ldpc_len, test_dataset, sd)
% dec2mat convert .dec file to .mat file
% 

dec_file_folder = fullfile(test_results_folder,'dec_file', run_time);
dec_mat_file_folder = fullfile(test_results_folder,'dec_mat_file', run_time);
if ~exist(dec_mat_file_folder,'dir')
    mkdir(dec_mat_file_folder)
end
dec_file_name = strcat(run_time, '-', num2str(sd),'.dec');
dec_file_path = fullfile(dec_file_folder, dec_file_name)
dec_mat_file_name = strcat(run_time, '-', num2str(sd),'.mat');
dec_mat_file_path = fullfile(dec_mat_file_folder, dec_mat_file_name);

[a1] = textread(dec_file_path,'%s','headerlines',0);
count = size(a1,1);
preLabel=zeros(count,ldpc_len);
label=zeros(1,ldpc_len);
for i = 1 : count
    la=a1{i,1};
    for k=1:ldpc_len
        label(1,k)=str2double(la(1,k));
    end
    preLabel(i,:)=label(1,:);
end

save(dec_mat_file_path, 'preLabel');
fprintf('dec2mat success \n');
% 
% 
end

