function scores_far_gar = calc_far_gar(test_results_folder, run_time, test_dataset, class_order, sd, iter_epochs)

% calc_acc_fals calc FAR and GAR according to different matching_score
%

vote_folder = fullfile(test_results_folder,'vote', run_time);

vote_file_path = fullfile(vote_folder, strcat(run_time,'-',num2str(sd),'-vote.mat'));
load(vote_file_path);
result_file_path = fullfile(vote_folder, strcat(run_time, '-',num2str(sd), '-scores_fars_gars.mat'));
img_nums = size(vote,1);
epoch_per = 1/(iter_epochs-1);

scores=zeros(iter_epochs,1);
fals=zeros(iter_epochs,1);
frrs=zeros(iter_epochs,1);
fal=0;
score=0;
dict_attack_acc=zeros(iter_epochs,1);

test_img_nums = size(vote,1);
class_nums = size(vote,2);
% 
% for s = 1 : iter_epochs
%     count=0;
%     for img_idx = 1:test_img_nums
%         for class_idx = 1:class_nums
%             a=vote(img_idx, class_idx);
%             if (a>=score)
%                 count=count+1;
%             end
%         end  
%     end
%     dict_attack_acc(s,1)=count/double(img_nums * class_nums);
%     scores(s,1)=score;
%     score=score+epoch_per;
% end

for s = 1 : iter_epochs
    co=1;
    count=0;
    fal=0;

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
            a=vote(co, class_order_idx);
            if (a>=score)
                count=count+1;
            end       
            for k = 1:class_nums
                if(k~=class_order_idx)
                    b=vote(co,k);
                    if (b>=score)
                        fal=fal+1;
                    end
               end
           end
           co=co+1;
        end  
    end
    acc(s,1)=count/double(img_nums);
    frr=fal/(double(img_nums)*(class_nums-1));

    fals(s,1)=frr;
    scores(s,1)=score;
    score=score+epoch_per;
end

scores_far_gar = struct('acc', acc, 'fals', fals, 'scores', scores);
save(result_file_path, 'scores_far_gar');

fprintf('calc fars and gars success \n');
end

