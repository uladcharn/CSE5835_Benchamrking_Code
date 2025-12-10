import numpy as np
import math
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import GPy
import os
import pickle
import datetime
from collections import Counter
import matplotlib.ticker as ticker
import random
from sklearn import preprocessing
from scipy.stats import norm
from sklearn.ensemble import RandomForestRegressor
import matplotlib.font_manager as font_manager
import copy
from scipy.interpolate import splrep
from scipy.interpolate import interp1d

def P_rand(nn):
    x_random = np.arange(nn)
    
    M = n_top
    N = nn
    
    P = np.array([None for i in x_random])
    E = np.array([None for i in x_random])
    A = np.array([None for i in x_random])
    cA = np.array([None for i in x_random])
    
    P[0] = M / N
    E[0] = M / N
    A[0] = M / N
    cA[0] = A[0]
    

    for i in x_random[1:]:
        P[i] = (M - E[i-1]) / (N - i)
        E[i] = np.sum(P[:(i+1)])
        j = 0
        A_i = P[i]
        while j < i:
            A_i *= (1 - P[j])
            j+=1
        A[i] = A_i
        cA[i] = np.sum(A[:(i+1)])
        
    return E / M, cA

def EF(x):
        n_eval = len(x)
        TopPercent_RS = P_rand(n_eval)[0]
        
        l_EF = []
        for j in np.arange(n_eval):
            l_EF.append(x[j] / TopPercent_RS[j])
            
        return l_EF

def AF(x):
        n_eval = len(x)
        TopPercent_RS = list(np.round(P_rand(n_eval)[0].astype(np.double) / 0.005, 0) * 0.005)
    #     We check Top% at 0.005 intervals between 0 and 1.
        l_TopPercent = []
        l_AF = []
        
        x = list(np.round(x.astype(np.double) / 0.005, 0) * 0.005)
        
        TopPercent = np.arange(0, 1.005, 0.005)
        
        pointer_x = 0
        pointer_rs = 0
        for t in TopPercent:
            if t in x and t in TopPercent_RS:
                n_x = 0
                n_rs = 0
                while pointer_x < len(x):
                    if x[pointer_x] == t:
                        pointer_x += 1
                        n_x = pointer_x
                        break
                    else:
                        pointer_x += 1

                while pointer_rs < len(TopPercent_RS):
                    if TopPercent_RS[pointer_rs] == t:
                        pointer_rs += 1
                        n_rs = pointer_rs
                        break
                    else:
                        pointer_rs += 1
            
                l_TopPercent.append(t)
                
                AF = n_rs / n_x
                l_AF.append(AF)  
            
        return l_TopPercent, l_AF

# smoothing for visualization purposes
def AF_interp1d(TopPercent):
    f_med = interp1d(AF(TopPercent[0])[0], AF(TopPercent[0])[1], kind = 'linear', fill_value='extrapolate')
#     again 0.005 intervals
    xx_ = np.linspace(min(AF(TopPercent[0])[0]), 1, 201 - int(min(AF(TopPercent[0])[0])/0.005))
    f_low = interp1d(AF(TopPercent[1])[0], AF(TopPercent[1])[1], kind = 'linear', fill_value='extrapolate')
    f_high = interp1d(AF(TopPercent[2])[0], AF(TopPercent[2])[1], kind = 'linear', fill_value='extrapolate')   
    return xx_, f_med, f_low, f_high

def aggregation_(seed, n_runs, n_fold):
    
    assert math.fmod(n_runs, n_fold) == 0
    fold_size = int(n_runs / n_fold)
    
    random.seed(seed)
    
    index_runs = list(np.arange(n_runs))
    
    agg_list = []
    
    i = 0
    
    while i < n_fold:
    
        index_i = random.sample(index_runs, fold_size)
        for j in index_i:
            index_runs.remove(j)
            
        agg_list.append(index_i)
        
        i += 1
#     print(agg_list)    
    return agg_list

def avg_(x):
#     nsteps
    n_eval = len(x[0]) 
    
#     fold
    n_fold = 5
    
#     rows = # of ensembles = 50
    n_runs = len(x)
    
    assert math.fmod(n_runs, n_fold) == 0
    fold_size = int(n_runs / n_fold)
    
#     # of seeds 
    n_sets = len(seed_list)
    
    l_index_list = []
    
    for i in np.arange(n_sets):
        
        s = aggregation_(seed_list[i], n_runs, n_fold)
        l_index_list.extend(s)

#     rows in l_index_list

    assert len(l_index_list) == n_sets * n_fold

    
    l_avg_runs = []

    for i in np.arange(len(l_index_list)):
        
        avg_run = np.zeros(n_eval)
        for j in l_index_list[i]:
            
            avg_run += np.array(x[j])
            
        avg_run = avg_run/fold_size
        l_avg_runs.append(avg_run)
    

    assert n_eval == len(l_avg_runs[0])
    assert n_sets * n_fold == len(l_avg_runs)
    
    mean_ = [None for i in np.arange(n_eval)]
    std_ = [None for i in np.arange(n_eval)]
    median_ = [None for i in np.arange(n_eval)]
    low_q = [None for i in np.arange(n_eval)]
    high_q = [None for i in np.arange(n_eval)]
    

#     5th, 95th percentile, mean, median are all accessible
    for i in np.arange(len(l_avg_runs[0])):
        i_column = []
        for j in np.arange(len(l_avg_runs)):
            i_column.append(l_avg_runs[j][i])
        
        i_column = np.array(i_column)
        mean_[i] = np.mean(i_column)
        median_[i] = np.median(i_column)
        std_[i] = np.std(i_column)
        low_q[i] = np.quantile(i_column, 0.05, out=None, overwrite_input=False, interpolation='linear')
        high_q[i] = np.quantile(i_column, 0.95, out=None, overwrite_input=False, interpolation='linear')
    
    return np.array(median_), np.array(low_q), np.array(high_q), np.array(mean_), np.array(std_)

def TopPercent(x_top_count, n_top, N):
    
    x_ = [[] for i in np.arange(len(x_top_count))]
    
    for i in np.arange(len(x_top_count)):
        for j in np.arange(N):
            if j < len(x_top_count[i]):
                x_[i].append(x_top_count[i][j] / n_top)
            else:
                x_[i].append(1)

    return x_

datasets = ['Perovskite','AutoAM']

colors = ['#006d2c', "#003c6d"]

seed_list = [4295, 8508, 326, 3135, 1549, 2528, 1274, 6545, 5971, 6269, 
            2422, 4287, 9320, 4932, 951, 4304, 1745, 5956, 7620, 4545]


fig = plt.figure(figsize=(6,6))
ax0 = fig.add_subplot(111)
fig1 = plt.figure(figsize=(6,6))
ax1 = fig1.add_subplot(111)
fig2 = plt.figure(figsize=(6,6))
ax2 = fig2.add_subplot(111)

model_name = 'rf' # rf or gp
seed_num = 20

for dataset, color in zip(datasets,colors):

    raw_dataset = pd.read_csv('./datasets/' + dataset + '_dataset.csv')
    feature_name = list(raw_dataset.columns)[:-1]
    objective_name = list(raw_dataset.columns)[-1]

    ds = copy.deepcopy(raw_dataset) 
    # only P3HT/CNT, Crossed barrel, AutoAM need this line; Perovskite and AgNP do not need this line.
    ds[objective_name] = -raw_dataset[objective_name].values

    ds_grouped = ds.groupby(feature_name)[objective_name].agg(lambda x: x.unique().mean())
    ds_grouped = (ds_grouped.to_frame()).reset_index()

    N = len(ds_grouped)
    print(f"Total number of observations: {N}")
    # number of top candidates, currently using top 5% of total dataset size
    n_top = int(math.ceil(N * 0.05))
    print(f"Total number candidates: {n_top}")
    # the top candidates and their indicies
    top_indices = list(ds_grouped.sort_values(objective_name).head(n_top).index)

    model_to_vis = np.load(f'test_run_{model_name}_{dataset}_{seed_num}.npy', allow_pickle = True)

    model_to_vis = model_to_vis[3] # test_run_results

    model_percentages = avg_(TopPercent(model_to_vis, n_top, N))

    print(f'Plotting performance - {dataset}')

    if dataset == datasets[0]:
        ax0.plot(np.arange(N)+1, P_rand(N)[0],'--',color='black',label='random baseline', linewidth=3.5)

    ax0.plot(np.arange(N) + 1, np.round(model_percentages[0].astype(np.double) / 0.005, 0) * 0.005, label = f'{model_name.upper()} - {dataset}', color = color, linewidth=3) # GP M52 : LCB
    ax0.fill_between(np.arange(N) + 1, np.round(model_percentages[1].astype(np.double) / 0.005, 0) * 0.005, np.round(model_percentages[2].astype(np.double) / 0.005, 0) * 0.005, color = color, alpha=0.2)

    # the rest are for visualization purposes, please adjust for different needs
    font = font_manager.FontProperties(family='Arial', size = 18, style='normal')
    leg = ax0.legend(prop = font, borderaxespad = 0,  labelspacing = 0.3, handlelength = 1.2, handletextpad = 0.3, frameon=False, loc = (0,0.81))
    for line in leg.get_lines():
        line.set_linewidth(4)
    ax0.set_ylabel("Top%", fontname="Arial", fontsize=24, rotation='vertical')    
    ax0.hlines(0.8, 0, 480, colors='k', linestyles='--', alpha = 0.2)
    ax0.set_ylim([0, 1.05])
    ax0.set_xscale('log')
    ax0.set_xlabel('learning cycle $i$', fontsize=24, fontname = 'Arial')
    ax0.xaxis.set_tick_params(labelsize=24)
    ax0.yaxis.set_tick_params(labelsize=24)
    ax0.spines['right'].set_visible(False)
    ax0.spines['top'].set_visible(False)
    ax0.set_xticks([1, 2, 10, 100], ['1', '2','10', '10$^{\mathrm{2}}$'],fontname = 'Arial') # , '6$×$10$^{\mathrm{2}}$'

    print(f'Plotting EF - {dataset}')

    if dataset == datasets[0]:
        ax1.plot(np.linspace(1, N, N), np.ones(N),'--',color='black',label='random baseline', linewidth = 3)        

    ax1.plot(np.arange(N) + 1, EF(np.round(model_percentages[0].astype(np.double) / 0.005, 0) * 0.005), label = f'{model_name.upper()} - {dataset}', color = color, linewidth=3)
    ax1.fill_between(np.arange(N) + 1, EF(np.round(model_percentages[1].astype(np.double) / 0.005, 0) * 0.005), EF(np.round(model_percentages[2].astype(np.double) / 0.005, 0) * 0.005), color = color, alpha=0.2)

    # the rest are for visualization purposes, please adjust for different needs
    font = font_manager.FontProperties(family='Arial', size = 18, style='normal')
    leg = ax1.legend(prop = font, borderaxespad = 0,  labelspacing = 0.3, handlelength = 1.2, handletextpad = 0.3, frameon=False, loc = (0,0.81))
    ax1.set_ylabel('EF', fontsize=24, rotation = 'horizontal', fontname = 'Arial', labelpad = 10)
    ax1.set_xlabel('learning cycle $i$', fontsize=24, fontname = 'Arial')
    ax1.set_xlim([1, 100])
    ax1.set_ylim([0, 10.5])
    ax1.set_xscale('log')
    ax1.spines['right'].set_visible(False)
    ax1.spines['top'].set_visible(False)
    ax1.xaxis.set_tick_params(labelsize=24)
    ax1.yaxis.set_tick_params(labelsize=24)
    ax1.set_xticks([10, 100], ['10', '10$^{\mathsf{2}}$'],fontname = 'Arial') # , '6$×$10$^{\mathsf{2}}$' , 600

    print(f'Plotting AF - {dataset}')

    ax2.plot(np.linspace(0, 1, 200), np.ones(200),'--',color='black',label=None, linewidth=3)        

    xx_, f_med_, f_low_, f_high_ = AF_interp1d(model_percentages)
    ax2.plot(xx_, f_med_(xx_), label = f'{model_name.upper()} - {dataset}', color = color, linewidth=3)
    ax2.fill_between(xx_, f_low_(xx_), f_high_(xx_), color = color, alpha=0.2)

    # the rest are for visualization purposes, please adjust for different needs
    font = font_manager.FontProperties(family='Arial', size = 18, style='normal')
    leg = ax2.legend(prop = font, borderaxespad = 0,  labelspacing = 0.3, handlelength = 1.2, handletextpad = 0.3, frameon=False, loc = (0,0.81))
    ax2.set_ylabel('AF', fontsize = 24, rotation = 'horizontal', fontname = 'Arial', labelpad = 10)
    ax2.set_xlabel('Top%', fontsize = 24, fontname = 'Arial')
    ax2.set_ylim([0, 10.5])
    ax2.spines['right'].set_visible(False)
    ax2.spines['top'].set_visible(False)
    ax2.xaxis.set_tick_params(labelsize=24)
    ax2.yaxis.set_tick_params(labelsize=24)
    ax2.set_xticks([0, 0.2, 0.4, 0.6, 0.8, 1.0], ['0', '0.2', '0.4', '0.6', '0.8', '1.0'], fontname = 'Arial')
    ax2.set_yticks([0, 1, 2, 4, 6, 8, 10], ['0', '1', '2', '4', '6', '8', '10'], fontname = 'Arial')

fig.tight_layout()
fig.savefig(f'{model_name}_performance.png')

fig1.tight_layout()
fig1.savefig(f'{model_name}_ef.png')

fig2.tight_layout()
fig2.savefig(f'{model_name}_af.png')



