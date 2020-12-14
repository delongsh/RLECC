function create_rec_linear_map(test_results_folder, run_time, test_dataset, ldpc_len, linear_map)
%create_rec_linear_map Linear map, generate .rec file to LDPC decoding
%

preOut_folder = fullfile(test_results_folder,'preOut', run_time);
rec_file_folder = fullfile(test_results_folder,'rec_file', run_time);
if ~exist(rec_file_folder,'dir')
    mkdir(rec_file_folder);
end


rec_file_name = strcat(run_time, '.rec');
preOut_name = strcat(run_time, '.mat');

fid=fopen(fullfile(rec_file_folder, rec_file_name),'wt');
out = cell2mat(struct2cell(load(fullfile(preOut_folder, preOut_name)))) / linear_map;
img_num = size(out,1);

for i=1:img_num
    cm=0.00;
    outCode=out(i,:);
    for k=1:ldpc_len-1
        a=roundn(outCode(k),-2);
        if(a>0)
            fprintf(fid,'+');
            fprintf(fid,'%0.2f ',a);
        end
        if(a==0)
            fprintf(fid,'-');
            fprintf(fid,'%0.2f ',cm);
        end
        if(a<0)
            fprintf(fid,'%0.2f ',a);
        end
    end
    b=roundn(outCode(ldpc_len),-2);
    if(b>0)
         fprintf(fid,'+');
         fprintf(fid,'%0.2f ',b);
    end
    if(b==0)
        fprintf(fid,'-');
        fprintf(fid,'%0.2f',cm);
    end
    if(b<0)
        fprintf(fid,'%0.2f',b);
    end
    fprintf(fid,'\n');
end
fclose(fid);
fprintf('create_rec_linear_map success \n');

end


