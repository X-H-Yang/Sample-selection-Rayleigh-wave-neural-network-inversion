% Code package purpose: Accoplish the proposed method in the paper 
%                       entitled "A Sample Selection Method for 
%                       Neural-network-based Rayleigh Wave Inversion".
%
% paper status: Published - Early access 
%               (journal: IEEE Transactions on Geoscience and Remote Sensing)
% Xiao-Hui Yang, Qiang Zu, Yuanyuan Zhou, Peng Han *, Xiaofei Chen (2024). 
% A Sample Selection Method for Neural-network-based Rayleigh Wave Inversion. 
% IEEE Transactions on Geoscience and Remote Sensing, 62, pp. 1-17.
% doi: 10.1109/TGRS.2023.3341955.
%
% software version: MATLAB R2017a
%
% Acknowledgement: The forward modeling program used to generate 
%                  theoretical Rayleigh wave dispersion curves in this 
%                  code package was obtained from the  website 
%                  (https://github.com/eespr/MuLTI) provided by 
%                  Killingbeck et al. (2018); the broad learning network 
%                  codes available on the website 
%                  (https://broadlearning.ai/, Chen and Liu 2017 and Chen 
%                  et al. 2018) were also applied for the accomplishment of
%                  this code package.
%
% Killingbeck et al. (2018): Killingbeck, S. F., Livermore, P. W., 
%                            Booth, A. D., & West, L. J. (2018). Multimodal 
%                            layered transdimensional inversion of seismic 
%                            dispersion curves with depth constraints. 
%                            Geochemistry, Geophysics, Geosystems, 19(12), 
%                            4957-4971.
%
% Chen and Liu 2017: Chen, C. P., & Liu, Z. (2017). Broad learning system: 
%                    An effective and efficient incremental learning system
%                    without the need for deep architecture. IEEE 
%                    transactions on neural networks and learning systems, 
%                    29(1), 10-24.
%
% Chen et al. 2018: Chen, C. P., Liu, Z., & Feng, S. (2018). Universal 
%                   approximation capability of broad learning system and 
%                   its structural variations. IEEE transactions on neural 
%                   networks and learning systems, 30(4), 1191-1204.
%
% Variables about training samples:
%           train_x_pool: all input samples (5000 samples) in sample pool
%                         (the sample selection is conducted based on all
%                         the samples, namely, the sample pool)
%           train_y_pool: all output samples (5000 samples) in sample pool
%
%           train_x: selected input training samples by proposed sample
%                    selection method
%           train_y: selected output training samples by proposed sample 
%                    selection method
%           train_x_random: selected input training samples by random
%                           sample selection
%           train_y_random: selected output training samples by random
%                           sample selection
%           train_x_bad: selected input training samples by worst sample
%                        selection
%           train_y_bad: selected output training samples by worst sample 
%                        selection
%
% Date: 2023/10/22
%
% Developed by: Xiao-Hui Yang, Currently working at 
%               Chengdu University of Information Technology
%
% Email: yangxh@cuit.edu.cn / xiao-hui.yang@hotmail.com
%
% Note: the specific descriptions of the proposed sample selection method
%       for Rayleigh wave inversion can refer to the paper entitiled
%      "A Sample Selection Method for Neural-network-based Rayleigh Wave
%       Inversion"; the users can cite this paper for scientific research.
% 
% Xiao-Hui Yang, Qiang Zu, Yuanyuan Zhou, Peng Han *, Xiaofei Chen (2024). 
% A Sample Selection Method for Neural-network-based Rayleigh Wave Inversion. 
% IEEE Transactions on Geoscience and Remote Sensing, 62, pp. 1-17. doi: 10.1109/TGRS.2023.3341955.

clear;
clc;
close all;

myFontSize = 20;
myMarkerSize = 20;

%% Reset the seed by clock for random number generation
rand('seed',sum(100*clock))
randn('seed',sum(100*clock))

%% Raw data (a numerical example, measured dispersion curves)
curve_00 = xlsread('numerical_fun.xls'); % fundamental mode
curve_01 = xlsread('numerical_1st.xls'); % 1st higher mode
curve_02 = xlsread('numerical_2nd.xls'); % 2nd higher mode
curve_03 = xlsread('numerical_3rd.xls'); % 3rd higher mode

% fundamental mode
f_00_original = curve_00(:,1)'; f_00_original = f_00_original(:)';
dispersion_00_original = curve_00(:,2); 
dispersion_00_original = dispersion_00_original(:)';
% first higher mode
f_01_original = curve_01(:,1)'; f_01_original = f_01_original(:)';
dispersion_01_original = curve_01(:,2); 
dispersion_01_original = dispersion_01_original(:)';
% second higher mode
f_02_original = curve_02(:,1); f_02_original = f_02_original(:)';
dispersion_02_original = curve_02(:,2); 
dispersion_02_original = dispersion_02_original(:)';
% third higher mode
f_03_original = curve_03(:,1); f_03_original = f_03_original(:)';
dispersion_03_original = curve_03(:,2); 
dispersion_03_original = dispersion_03_original(:)';

% interpolation process - frequency and dispersion values for invertion use
f_00_min = 6; f_00_max = 48;
f_01_min = 31.5; f_01_max = 69.5;
f_02_min = 29.5; f_02_max = 49;
f_03_min = 64; f_03_max = 100;
df = 0.5;
f_00 = f_00_min:df:f_00_max;
f_01 = f_01_min:df:f_01_max;
f_02 = f_02_min:df:f_02_max;
f_03 = f_03_min:df:f_03_max;
dispersion_00 = interp1(f_00_original,dispersion_00_original,f_00);
dispersion_01 = interp1(f_01_original,dispersion_01_original,f_01);
dispersion_02 = interp1(f_02_original,dispersion_02_original,f_02);
dispersion_03 = interp1(f_03_original,dispersion_03_original,f_03);

dispersion_all_cell = cell(1,4);
dispersion_all_cell{1} = dispersion_00;
dispersion_all_cell{2} = dispersion_01;
dispersion_all_cell{3} = dispersion_02;
dispersion_all_cell{4} = dispersion_03;

f = 6:df:100;
index_vec_all = cell(1,4);
index_vec_all{1} = [find(f == f_00_min) find(f == f_00_max)];
index_vec_all{2} = [find(f == f_01_min) find(f == f_01_max)];
index_vec_all{3} = [find(f == f_02_min) find(f == f_02_max)];
index_vec_all{4} = [find(f == f_03_min) find(f == f_03_max)];

modes_num_vec = [1 2 3 4]; % available modes of dispersion curves
index_vec = index_vec_all(modes_num_vec);

dispersions_R_true = []; % integrate different modes of dispersion curves
for jj = 1:1:length(modes_num_vec)
    temp = modes_num_vec(jj);
    dispersions_R_true = [dispersions_R_true dispersion_all_cell{temp}];
end

validation_x = dispersions_R_true;

%% Actual model parameters (a numerical example)
h_true = [2 4 5 5];
h_true2 = [h_true 0];
Vs_true = [400 200 300 500 650];
Vp_true = [700 300 500 900 1100]; % primary wave velocity (all layers)
den_true = [1.9 1.7 1.8 2.0 2.1]; % density (all layers)

layers_num = length(Vp_true);

Vs_true_profile = [h_true Vs_true];

Vs_profile_lower = [1.2 2.4 3.0 3.0 240 120 180 300 390]; % search space
Vs_profile_upper = [2.8 5.6 7.0 7.0 560 350 420 700 780]; % search space

%% Hyperparameters of the broad learning network
Fea_vec = 4:2:10;
Win_vec = 4:2:20;
Enhan_vec = 4:2:30;

tic % start to record time
%% create a pool of 5000 samples
train_samples_N_pool = 5000; % number of samples in the sample pool
disp('Waiting several minutes for generating a training sample pool ...')
% the generation of 5000 samples in the pool
[train_x_pool,train_y_pool] = getSamples_RW_fieldData_sub_2(...
    train_samples_N_pool,modes_num_vec,index_vec_all,f,Vp_true,den_true,...
    Vs_profile_lower,Vs_profile_upper,dispersions_R_true);
disp('All samples in original sample pool were obtained!')

train_samples_N = 500; % the number of selected samples
samples_N = train_samples_N;

%% Inversion manner 1 - without sample selection - 5000 samples
% Inversion based on all 5000 samples in the pool for comparison

% Normalize data (mean=0, std=1)
[mean_x_pool,std_x_pool,train_x_norm_pool] = normalized_fun(train_x_pool);
[mean_y_pool,std_y_pool,train_y_norm_pool] = normalized_fun(train_y_pool);
validation_x_norm_pool = zeros(size(validation_x,1),size(validation_x,2));
for i = 1:1:size(validation_x,2)
    validation_x_norm_pool(:,i) = (validation_x(:,i)-mean_x_pool(i))/...
        std_x_pool(i);
end

% broad learning network - network training and complexity selection
% the network complexity selection is performed based on forward modeling
disp('Performing inversion manner 1 - without sample selection ...')
[Y_hat_pool,all_index_pool,NumFea_hat_pool,NumWin_hat_pool,...
    NumEnhan_hat_pool] = bls_regression_Y_sub_noTest(train_x_norm_pool,...
    train_y_norm_pool,Fea_vec,Win_vec,Enhan_vec,validation_x,Vp_true,...
    den_true,f,modes_num_vec,index_vec,index_vec_all,mean_y_pool,...
    std_y_pool,validation_x_norm_pool);
dispersions_R_inverted_pool = calDispersions_2(Y_hat_pool,Vp_true,...
    den_true,f,modes_num_vec,index_vec_all); % inverted dispersion curves

disp('Inverted parameters-inversion manner 1 (without sample selection:');
Y_hat_pool

% draw inverted dispersion curves
figure()
plot(curve_00(:,1),curve_00(:,2),'k.','MarkerSize',myMarkerSize);
hold on
plot(curve_01(:,1),curve_01(:,2),'k.','MarkerSize',myMarkerSize);
plot(curve_02(:,1),curve_02(:,2),'k.','MarkerSize',myMarkerSize);
plot(curve_03(:,1),curve_03(:,2),'k.','MarkerSize',myMarkerSize);
drawDispersionsCompareField_5(dispersions_R_inverted_pool,f,...
    modes_num_vec,index_vec,myFontSize) % inverted
axis([0 100 150 600]);
set(gca,'XTick',0:20:100);
set(gca,'YTick',100:100:700);
set(gca,'FontName','Times New Roman','FontSize',myFontSize);
title('Without sample selection - 5000 samples');

% draw objective function curve
figure()
plot(all_index_pool(:,1)-all_index_pool(1,1),all_index_pool(:,2),'b','Linewidth',1.5);
axis([0 100 0 20]);
set(gca,'YTick',0:5:20);
xlabel('Time [s]','FontSize',myFontSize);
ylabel('Obj','FontSize',myFontSize);
set(gca,'FontName','Times New Roman','FontSize',myFontSize);
title('Without sample selection - 5000 samples');


%% Inversion manner 2 - proposed sample selection - 500 samples
% Inversion based selected 500 samples by proposed sample selection method

%%%%%%%%%%%%%%%%%%%%%%%%%%%%% sample selection %%%%%%%%%%%%%%%%%%%%%%%%%%%%
corr_vector = zeros(1,train_samples_N_pool);

window_factor = 0.2; % recommended value of moving window factor
window_num = floor(window_factor*length(validation_x));
corr_vector_temp = zeros(1,length(validation_x)-window_num);

for j = 1:train_samples_N_pool
    
    for i = 1:length(validation_x)-window_num
        temp = train_x_pool(j,i:i+window_num);
        if sum(abs(temp-temp(1)*ones(1,length(temp)))) == 0
            temp(1) = temp(1) + 0.001; % avoid nan
        end
        cc = corrcoef(temp,validation_x(i:i+window_num));
        
        corr_vector_temp(i) = cc(2,1);
    end
    corr_vector(j) = mean(corr_vector_temp);
    
    [corr_rank,corr_index] = sort(corr_vector,'descend');
end

% selected training samples based on proposed sample selection method
train_x = train_x_pool(corr_index(1:samples_N),:);
train_y = train_y_pool(corr_index(1:samples_N),:);
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% Normalize data (mean=0, std=1)
[mean_x,std_x,train_x_norm] = normalized_fun(train_x);
[mean_y,std_y,train_y_norm] = normalized_fun(train_y);
validation_x_norm = zeros(size(validation_x,1),size(validation_x,2));
for i = 1:1:size(validation_x,2)
    validation_x_norm(:,i) = (validation_x(:,i)-mean_x(i))/std_x(i);
end

% broad learning network - network training and complexity selection
% the network complexity selection is performed based on forward modeling
disp('Performing inversion manner 2 - proposed sample selection ...')
[Y_hat,all_index,NumFea_hat,NumWin_hat,NumEnhan_hat] = ...
    bls_regression_Y_sub_noTest(train_x_norm,train_y_norm,Fea_vec,...
    Win_vec,Enhan_vec,validation_x,Vp_true,den_true,f,modes_num_vec,...
    index_vec,index_vec_all,mean_y,std_y,validation_x_norm);
dispersions_R_inverted = calDispersions_2(Y_hat,Vp_true,den_true,f,...
    modes_num_vec,index_vec_all);

disp('Inverted parameters-inversion manner 2 (proposed sample selection:');
Y_hat

% draw inverted dispersion curves
figure()
plot(curve_00(:,1),curve_00(:,2),'k.','MarkerSize',myMarkerSize);
hold on
plot(curve_01(:,1),curve_01(:,2),'k.','MarkerSize',myMarkerSize);
plot(curve_02(:,1),curve_02(:,2),'k.','MarkerSize',myMarkerSize);
plot(curve_03(:,1),curve_03(:,2),'k.','MarkerSize',myMarkerSize);
drawDispersionsCompareField_5(dispersions_R_inverted,f,...
    modes_num_vec,index_vec,myFontSize) % inverted
axis([0 100 150 600]);
set(gca,'XTick',0:20:100);
set(gca,'YTick',100:100:700);
set(gca,'FontName','Times New Roman','FontSize',myFontSize);
title('Proposed sample selection - 500 samples');

% draw objective function curve
figure()
plot(all_index(:,1)-all_index(1,1),all_index(:,2),'b','Linewidth',1.5);
axis([0 100 0 20]);
set(gca,'YTick',0:5:20);
xlabel('Time [s]','FontSize',myFontSize);
ylabel('Obj','FontSize',myFontSize);
set(gca,'FontName','Times New Roman','FontSize',myFontSize);
title('Proposed sample selection - 500 samples');

%% Inversion manner 3 - random sample selection - 500 samples
% Inversion based selected 500 samples by random sample selection
%%%%%%%%%%%%%%%%%%%%%%%%% random sample selection %%%%%%%%%%%%%%%%%%%%%%%%%
train_x_random = train_x_pool(1:samples_N,:);
train_y_random = train_y_pool(1:samples_N,:);
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% Normalize data (mean=0, std=1)
[mean_x_random,std_x_random,train_x_norm_random] = ...
    normalized_fun(train_x_random);
[mean_y_random,std_y_random,train_y_norm_random] = ...
    normalized_fun(train_y_random);
validation_x_norm_random = zeros(size(validation_x,1),...
    size(validation_x,2));
for i = 1:1:size(validation_x,2)
    validation_x_norm_random(:,i) = ...
        (validation_x(:,i)-mean_x_random(i))/std_x_random(i);
end

% broad learning network - network training and complexity selection
% the network complexity selection is performed based on forward modeling
disp('Performing inversion manner 3 - random sample selection ...')
[Y_hat_random,all_index_random,NumFea_hat_random,NumWin_hat_random,...
    NumEnhan_hat_random] = bls_regression_Y_sub_noTest(...
    train_x_norm_random,train_y_norm_random,Fea_vec,Win_vec,Enhan_vec,...
    validation_x,Vp_true,den_true,f,modes_num_vec,index_vec,...
    index_vec_all,mean_y_random,std_y_random,validation_x_norm_random);
dispersions_R_inverted_random = calDispersions_2(Y_hat_random,Vp_true,...
    den_true,f,modes_num_vec,index_vec_all);

disp('Inverted parameters-inversion manner 3 (random sample selection:');
Y_hat_random

% draw inverted dispersion curves
figure()
plot(curve_00(:,1),curve_00(:,2),'k.','MarkerSize',myMarkerSize);
hold on
plot(curve_01(:,1),curve_01(:,2),'k.','MarkerSize',myMarkerSize);
plot(curve_02(:,1),curve_02(:,2),'k.','MarkerSize',myMarkerSize);
plot(curve_03(:,1),curve_03(:,2),'k.','MarkerSize',myMarkerSize);
drawDispersionsCompareField_5(dispersions_R_inverted_random,f,...
    modes_num_vec,index_vec,myFontSize) % inverted
axis([0 100 150 600]);
set(gca,'XTick',0:20:100);
set(gca,'YTick',100:100:700);
set(gca,'FontName','Times New Roman','FontSize',myFontSize);
title('Random sample selection - 500 samples');

% draw objective function curve
figure()
plot(all_index_random(:,1)-all_index_random(1,1),all_index_random(:,2),...
    'b','Linewidth',1.5);
axis([0 100 0 20]);
set(gca,'YTick',0:5:20);
xlabel('Time [s]','FontSize',myFontSize);
ylabel('Obj','FontSize',myFontSize);
set(gca,'FontName','Times New Roman','FontSize',myFontSize);
title('Random sample selection - 500 samples');

%% Inversion manner 4 - worst sample selection - 500 samples
% Inversion based selected 500 samples by worst sample selection
%%%%%%%%%%%%%%%%%%%%%%%%% worst sample selection %%%%%%%%%%%%%%%%%%%%%%%%%%
train_x_bad = train_x_pool(corr_index(end-samples_N+1:end),:);
train_y_bad = train_y_pool(corr_index(end-samples_N+1:end),:);
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% Normalize data (mean=0, std=1)
[mean_x_bad,std_x_bad,train_x_norm_bad] = normalized_fun(train_x_bad);
[mean_y_bad,std_y_bad,train_y_norm_bad] = normalized_fun(train_y_bad);
validation_x_norm_bad = zeros(size(validation_x,1),size(validation_x,2));
for i = 1:1:size(validation_x,2)
    validation_x_norm_bad(:,i) = ...
        (validation_x(:,i)-mean_x_bad(i))/std_x_bad(i);
end

% broad learning network - network training and complexity selection
% the network complexity selection is performed based on forward modeling
disp('Performing inversion manner 4 - worst sample selection ...')
[Y_hat_bad,all_index_bad,NumFea_hat_bad,NumWin_hat_bad,NumEnhan_hat_bad]...
    = bls_regression_Y_sub_noTest(train_x_norm_bad,train_y_norm_bad,...
    Fea_vec,Win_vec,Enhan_vec,validation_x,Vp_true,den_true,f,...
    modes_num_vec,index_vec,index_vec_all,mean_y_bad,std_y_bad,...
    validation_x_norm_bad);
dispersions_R_inverted_bad = calDispersions_2(Y_hat_bad,Vp_true,...
    den_true,f,modes_num_vec,index_vec_all);

disp('Inverted parameters-inversion manner 4 (worst sample selection:');
Y_hat_bad

% draw inverted dispersion curves
figure()
plot(curve_00(:,1),curve_00(:,2),'k.','MarkerSize',myMarkerSize);
hold on
plot(curve_01(:,1),curve_01(:,2),'k.','MarkerSize',myMarkerSize);
plot(curve_02(:,1),curve_02(:,2),'k.','MarkerSize',myMarkerSize);
plot(curve_03(:,1),curve_03(:,2),'k.','MarkerSize',myMarkerSize);
drawDispersionsCompareField_5(dispersions_R_inverted_bad,f,...
    modes_num_vec,index_vec,myFontSize) % inverted
axis([0 100 150 600]);
set(gca,'XTick',0:20:100);
set(gca,'YTick',100:100:700);
set(gca,'FontName','Times New Roman','FontSize',myFontSize);
title('Worst sample selection - 500 samples');

% draw objective function curve
figure()
plot(all_index_bad(:,1)-all_index_bad(1,1),all_index_bad(:,2),...
    'b','Linewidth',1.5);
axis([0 100 0 20]);
set(gca,'YTick',0:5:20);
xlabel('Time [s]','FontSize',myFontSize);
ylabel('Obj','FontSize',myFontSize);
set(gca,'FontName','Times New Roman','FontSize',myFontSize);
title('Worst sample selection - 500 samples');
