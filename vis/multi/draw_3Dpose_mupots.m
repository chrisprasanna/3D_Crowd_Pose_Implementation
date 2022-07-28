clc
clear 
close all

root_path = pwd;
save_path = './vis/';
num_joint =  17;

colorList_skeleton = [
    255/255 128/255 0/255;
    255/255 153/255 51/255;
    255/255 178/255 102/255;
    230/255 230/255 0/255;
    
    255/255 153/255 255/255;
    153/255 204/255 255/255;
    
    255/255 102/255 255/255;
    255/255 51/255 255/255;
    
    102/255 178/255 255/255;
    51/255 153/255 255/255;
    
    255/255 153/255 153/255;
    255/255 102/255 102/255;
    255/255 51/255 51/255;
    
    153/255 255/255 153/255;
    102/255 255/255 102/255;
    51/255 255/255 51/255;
    ];
colorList_joint = [
    255/255 128/255 0/255;
    255/255 153/255 51/255;
    255/255 153/255 153/255;
    255/255 102/255 102/255;
    255/255 51/255 51/255;
    153/255 255/255 153/255;
    102/255 255/255 102/255;
    51/255 255/255 51/255;
    255/255 153/255 255/255;
    255/255 102/255 255/255;
    255/255 51/255 255/255;
    153/255 204/255 255/255;
    102/255 178/255 255/255;
    51/255 153/255 255/255;
    230/255 230/255 0/255;
    230/255 230/255 0/255;
    255/255 178/255 102/255;
    
    ];
% skeleton = [ [0, 16], [1, 16], [1, 15], [15, 14], [14, 8], [14, 11], [8, 9], [9, 10], [11, 12], [12, 13], [1, 2], [2, 3], [3, 4], [1, 5], [5, 6], [6, 7] ];
% skeleton = transpose(reshape(skeleton,[2,16])) + 1;
% skeleton = [[0, 1], [1, 2], [2, 3], [0, 4], [4, 5], [5, 6], [0, 7], [7, 8],[8, 9], [9, 10], [8, 14], [14, 15], [15, 16], [8, 11], [11, 12], [12, 13]];

skeleton = [
    [0,16],...
    [1,16],...
    [1,2],...
    [2,3],...
    [3,4],...
    [1,5],...
    [5,6],...
    [6,7],...
    [1,15],...
    [15,14],...
    [14,8]...
    [8,9],...
    [9,10],...
    [14,11],...
    [11,12],...
    [12,13]
    ];
skeleton = transpose(reshape(skeleton,[2,length(skeleton)/2])) + 1;

data_dir = '../../MultiPersonTestSet/';
files = dir(data_dir);
folderNames = {files([files.isdir]).name};
folderNames = folderNames(~ismember(folderNames ,{'.','..'}));

for ii = 1:length(folderNames)
    
    data_path = fullfile(data_dir, folderNames{ii});
    results_path = fullfile('vis', folderNames{ii});
    images = dir(fullfile(data_path, '*.jpg'));
    numFiles = length(images);
    
    preds_3d_kpt = load([data_path, '/MuPots_', folderNames{ii},'_camCentric.mat']);
    preds_3d_kpt = preds_3d_kpt.data;
    
    preds_3d_kpt = permute(preds_3d_kpt, [1 2 4 3]); % switch dimension order
    
    sizex = 2048;
    sizey = 2048;
    img = ones(sizex,sizey,3);
    
    data = cell(1,numFiles);
    myresults = cell(1,numFiles);
    
    vidfile = VideoWriter(fullfile(pwd, results_path, [folderNames{ii},'.mp4']),'MPEG-4');
    open(vidfile);
    sp = figure;
    set(sp, 'visible', 'off');
    
    singeVidFile = VideoWriter(fullfile(pwd, results_path, [folderNames{ii},'_singleFig.mp4']),'MPEG-4');
    open(singeVidFile);
        
    %%
    for frame = 1:length(preds_3d_kpt)
        msg = [folderNames{ii}, ' (', num2str(ii), '/', num2str(length(folderNames)),'), frame (',...
            num2str(frame), '/', num2str(length(preds_3d_kpt)), ')'];
        disp(msg)
        
        % corresponding predictor image
        img_path = fullfile(images(frame).folder, images(frame).name);
        img = imread(img_path);
        
        % image to save to
        % img_name = [folderNames{ii}, '/img_', num2str(frame), '.jpg'];
        img_name = [folderNames{ii}, '/', images(frame).name];
        
        % draw stick figures
        f = draw_3Dskeleton(img,preds_3d_kpt,num_joint,skeleton,colorList_joint,colorList_skeleton,frame);
        
        if ~exist(strcat(save_path,folderNames{ii}), 'dir')
            mkdir(strcat(save_path,folderNames{ii}));
        end
        
        saveas(f, strcat(save_path,img_name));
        close(f);
        
        % collect data
        data{frame} = imread([data_path,'\', images(frame).name]);
        myresults{frame} = imread(strcat(save_path,img_name));
        
        % write video
        subplot(1,2,1)
        imshow(data{frame})
        subplot(1,2,2)
        imshow(myresults{frame});
        
        F = getframe(sp);
        writeVideo(vidfile, F);
        
        % write single fig video
        fig = figure;
        set(fig, 'visible', 'off');
        imshow(myresults{frame});
        F = getframe(fig);
        writeVideo(singeVidFile, F);
        title(img_name, 'Interpreter', 'none')
        close(fig)
    end
    close(vidfile)
    close(singeVidFile)
    
end
