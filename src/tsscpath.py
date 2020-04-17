####=====================================================================================================================
####  0. Configure
####=====================================================================================================================
'''
import sys
sys.path.append(path+'TSSCpath/src/')
import tsscpath
'''
import numpy as np
import pandas as pd
import pickle as pkl
import scipy.optimize as op
import scanpy as sc
from fa2 import ForceAtlas2
import seaborn as sns
import matplotlib.pyplot as plt 
import h5py
import math
import matplotlib as mpl

import pickle
from sklearn.decomposition import PCA 
from sklearn.neighbors import NearestNeighbors
from scipy.stats import zscore
import scipy
import seaborn as sns
from scipy.io import mmread
import networkx as nx

import warnings
warnings.simplefilter('ignore')



##--------------------
## 1. Util functions
##--------------------
def pkl_save(data, file):
    with open(file, "wb") as fp:   #Pickling
        pickle.dump(data, fp)

def pkl_load(file):
    with open(file, "rb") as fp:   # Unpickling
        data = pickle.load(fp)
    return data

def plot_gene(Exp, nrow, ncol, genes, cmap='Reds'):
    fig = plt.figure(figsize=[4 * ncol, 4 * nrow])
    gs = plt.GridSpec(nrow, ncol)
    N = len(genes)
    for j, gene in enumerate(genes):
        c = Exp.loc[:,gene]
        ax = plt.subplot(gs[j//ncol, j%ncol])
        ax.scatter(Exp.loc[:,'x'], Exp.loc[:,'y'], s=3, color='lightgrey')
        ax.scatter(Exp.loc[:,'x'], Exp.loc[:,'y'], s=5, cmap=cmap, c=np.log(c), vmin=c.min(), vmax=c.max())
        ax.set_title(f'{gene}')
        ax.set_axis_off()
    plt.show()

def plot_UMAP_gene(Exp, nrow, ncol, genes, cmap):
    fig = plt.figure(figsize=[4 * ncol, 4 * nrow])
    gs = plt.GridSpec(nrow, ncol)
    N = len(genes)
    for j, gene in enumerate(genes):
        c = Exp.loc[:,gene]
        ax = plt.subplot(gs[j//ncol, j%ncol])
        ax.scatter(Exp.loc[:,'umap_x'], Exp.loc[:,'umap_y'], s=3, color='lightgrey')
        ax.scatter(Exp.loc[:,'umap_x'], Exp.loc[:,'umap_y'], s=5, cmap=cmap, c=np.log(c), vmin=c.min(), vmax=c.mean())
        ax.set_title(f'{gene}')
        ax.set_axis_off()
    plt.show()


####=====================================================================================================================
####  2. Process Data
####=====================================================================================================================

##-------------
## 2.1 Data QC
##-------------
def plot_QC(df, pdfnm):
    n_counts = df.sum(axis=1) # 每个cell的count
    n_genes = (df>0).sum(axis=1) # 每个cell的基因
    n_cells = (df>0).sum() # 每个基因有多少细胞
    df_MT = df.loc[:,df.columns.str.contains('MT-')]
    df_MT_UMI = df_MT.sum(axis=1)
    MT_ratio = df_MT_UMI/n_counts
    QC = pd.DataFrame({'n_counts': n_counts, 'n_genes':n_genes, 'MT_ratio':MT_ratio})
    QC_gene = pd.DataFrame({'n_cells':n_cells})
    fig, axs = plt.subplots(1, 4, figsize=(12, 3), sharey=True)
    # f = plt.figure(figsize=(16, 4))
    # ax1 = f.add_subplot(141)
    # ax2 = f.add_subplot(142)
    # ax3 = f.add_subplot(143)
    # ax4 = f.add_subplot(144)
    # plt.subplot(1,4,1)
    ax1 = sns.violinplot(y=QC.loc[:,'n_counts'], color='#fbb4ae', ax=axs[0])
#     sns.stripplot(y=QC.loc[:,'n_counts'], color='grey', jitter=0.05, size=3)
    # plt.subplot(1,4,2)
    ax2 = sns.violinplot(y=QC.loc[:,'n_genes'], color='#b3cde3', ax=axs[1])
#     sns.stripplot(y=QC.loc[:,'n_genes'], color='grey', jitter=0.05, size=3)
    # plt.subplot(1,4,3)
    ax3 = sns.violinplot(y=QC.loc[:,'MT_ratio'], color='#ccebc5', ax=axs[2])
#     sns.stripplot(y=QC.loc[:,'MT_ratio'], color='grey', jitter=0.05, size=3)
    # plt.subplot(1,4,4)
    ax3 = sns.violinplot(y=QC_gene.loc[:,'n_cells'], color='#ccebc5', ax=axs[3])
#     sns.stripplot(y=QC.loc[:,'MT_ratio'], color='grey', jitter=0.05, size=3)
    plt.savefig(pdfnm)
    print(QC.describe())

##-----------------
## 2.2 Data Filter
##-----------------
def filter_outlier_cell(df_list):
    '''剔除掉UMI异常的细胞，total UMI在mean+-2sd以外的(95%)
    Args:
        df_list: different DataFrame in a list format.
    Output:
        df_filter_list: 剔除outlier之后的DataFrame
    '''
    total_UMI = []
    df_filter_list = []
    for df in df_list: # cell*gene
        df_total_UMI = df.sum(axis=1).tolist()
        total_UMI += df_total_UMI
    total_UMI = np.array(total_UMI)
    mu = np.mean(total_UMI)
    sd = np.std(total_UMI)
    lower, upper = mu-2*sd, mu+2*sd
    for df in df_list:
        df_total_UMI = df.sum(axis=1)
        keep_idx = (df_total_UMI>=lower) & (df_total_UMI<=upper)
        df_filter = df.loc[keep_idx, ]
        df_filter_list.append(df_filter)
        print('Keep percentage: %.2f%%'%(keep_idx.sum()/len(keep_idx)*100))
    return df_filter_list    

def filter_lowgene_cells(df, min_genes=200):
    '''踢除基因表达数量小于一定数量的cell
    '''
    keep_idx = (df>0).sum(axis=1)>min_genes
    df_keep = df.loc[keep_idx, :]
    print('Keep percentage: %.2f%%(cell_filter)'%(keep_idx.sum()/len(keep_idx)*100))
    return df_keep

def filter_dead_cell(df, cutoff=0.2):
    '''剔除掉MT较高的细胞
    '''
    df_total_UMI = df.sum(axis=1)
    df_MT = df.loc[:,df.columns.str.contains('MT-')]
    df_MT_UMI = df_MT.sum(axis=1)
    keep_idx = df_MT_UMI/df_total_UMI < cutoff
    df_keep = df.loc[keep_idx, :]
    print('Keep percentage: %.2f(MT_ratio_filter)'%(sum(keep_idx)/len(keep_idx)*100))
    return df_keep

def filter_abnormalcount_cells(df, count_cutoff):
    '''踢除细胞counts大于一定数量的cell
    '''
    keep_idx = (df.sum(axis=1)<count_cutoff[1]) & (df.sum(axis=1)>count_cutoff[0])
    df_keep = df.loc[keep_idx, :]
    return df_keep

def filter_lowcell_gene(df, min_cells=3):
    '''踢除细胞表达数量小于一定数量的gene
    '''
    keep_idx = (df>0).sum()>min_cells
    df_filter = df.loc[:, keep_idx]
    print('Keep percentage: %.2f(gene_filter)'%(sum(keep_idx)/len(keep_idx)*100))
    return df_filter

def cell_filter(df, MT_ratio, min_genes, min_cells):
    df = filter_dead_cell(df, cutoff=MT_ratio)
    df = filter_lowgene_cells(df, min_genes=min_genes)
    df = filter_lowcell_gene(df, min_cells=min_cells)
    return df

##------------------
## 2.3 Data combine
##------------------
def combine_data(df_list):
    '''合并list中数据
    '''
    print('Combine data from same time point...')
    com_dfs = []
    for dfs in df_list:
        for k,df in enumerate(dfs):
            if k==0:
                genes = [i.columns.tolist() for i in dfs]
                genes = set(genes[0]).intersection(*genes)
                com_df = df.loc[:, genes]
            else:
                com_df = pd.concat([com_df, df.loc[:,genes]])
        com_dfs.append(com_df)
    return(com_dfs)

# Data normalize
def normalize_data(df_list, base=10000):
    '''对数据进行normalization,基于total count进行,normalize到0.1million
    '''
    print('Normalize data to %d count ...'%base)
    df_norm_list = []
    for df in df_list:
        df_total_UMI = df.sum(axis=1)
        df_norm = df.div(df_total_UMI, axis=0)*base
        df_norm_list.append(df_norm)
    return df_norm_list


####=====================================================================================================================
####  3. High variable gene
####=====================================================================================================================

def get_vscore(norm_data):
    '''
    '''
    min_mean = 0 # Exclude genes with average expression of this value or lower
    fit_percentile = 33 # Fit to this percentile of CV-vs-mean values
    error_wt = 1; # Exponent of fitting function. Value of 2 is least-squares (L2). Value of 1 is robust (L1).
    nBins = 50;

    # remove duplicate gene
    keepidx = ~norm_data.columns.duplicated()
    norm_data = norm_data.loc[:, keepidx]

    # PREPARE DATA FOR FITTING
    mu_gene = norm_data.mean()
    idx = mu_gene>min_mean
    mu_gene = mu_gene[idx]
    FF_gene = norm_data.var()[idx]/mu_gene

    # Perform fit on log-transformed data:
    data_x = np.log(mu_gene)
    data_y = np.log(FF_gene/mu_gene)

    def runningquantile(x, y, p, nBins):
        x = x.sort_values()
        y = y[x.sort_values().index]
        dx = (x[-1]-x[0])/nBins
        xOut = np.arange(x[0]+dx/2, x[-1]-dx/2, dx)
        yOut = []
        for k,v in enumerate(xOut):
            ind = (x>=(v-dx/2)) & (x<(v+dx/2))
            if ind.sum()>0:
                yOut.append(np.percentile(y[ind], p))
            else:
                if k>0:
                    yOut.append(yOut[k-1])
                else:
                    yOut.append(np.nan)
        yOut = np.array(yOut)
        return(xOut, yOut)

    x,y = runningquantile(data_x, data_y, fit_percentile, nBins);
    
    FFhist, FFhist_vals = np.histogram(np.log10(FF_gene),200)
    c = np.exp(FFhist_vals[FFhist.argmax()]) # z = 1/( (1+a)(1+b) )
    c = max([1,c]) # Do not allow c to fall below 1.

    # FIT c=[CV_eff]^2
    errFun = lambda b: sum(np.abs(np.log(c*np.exp(-x) + b)-y)**error_wt)
    b = op.fmin(func=errFun, x0=0.1)[0]
    a = c/(1+b) - 1

    v_scores = FF_gene/((1+a)*(1+b) + b*mu_gene)
    v_scores = v_scores.sort_values(ascending=False)
    print('a:%.3f, b:%.3f, c:%.3f'%(a, b, c))
    return(v_scores)

def get_HVG(df_list):
    '''
    '''
    min_exp = 0
    min_exp_cnt = 10
    vargene = 2000
    minGeneCor = 0.2
    excludeGeneCor = 0.4
    cell_cycle = ['MCM5','PCNA','TYMS','FEN1','MCM2','MCM4','RRM1','UNG','GINS2','MCM6','CDCA7','DTL','PRIM1','UHRF1','MLF1IP','HELLS','RFC2','RPA2','NASP','RAD51AP1','GMNN','WDR76','SLBP','CCNE2','UBR7','POLD3','MSH2','ATAD2','RAD51','RRM2','CDC45','CDC6','EXO1','TIPIN','DSCC1','BLM','CASP8AP2','USP1','CLSPN','POLA1','CHAF1B','BRIP1','E2F8','HMGB2','CDK1','NUSAP1','UBE2C','BIRC5','TPX2','TOP2A','NDC80','CKS2','NUF2','CKS1B','MKI67','TMPO','CENPF','TACC3','FAM64A','SMC4','CCNB2','CKAP2L','CKAP2','AURKB','BUB1','KIF11','ANP32E','TUBB4B','GTSE1','KIF20B','HJURP','CDCA3','HN1','CDC20','TTK','CDC25C','KIF2C','RANGAP1','NCAPD2','DLGAP5','CDCA2','CDCA8','ECT2','KIF23','HMMR','AURKA','PSRC1','ANLN','LBR','CKAP5','CENPE','CTCF','NEK2','G2E3','GAS2L3','CBX5','CENPA']
    housekeep = ['RPS18','GAPDH','PGK1','PPIA','RPL13A','RPLP0','B2M','YWHAZ','SDHA','TFRC','RPA1','RPA2','RPAIN','RPE','RPL15','RPL15','RPL22','RPL32','RPL32','RPL35A','RPL4','RPL7L1','RPN1','RPN2','RPP30','RPP38','RPRD1A','RPS19BP1','RPS6KA5','RPS6KB1','RPUSD1','RPUSD2','RPUSD4']
    dfFilter_list = []
    for df in df_list:
        # 基于threshold进行filter
        df = df.loc[:, df.apply(lambda x:sum(x>min_exp)>min_exp_cnt)]
        MTidx = df.columns.str.contains('MT-')
        df = df.loc[:, ~MTidx]
        # 基于CV进行filter
#         gene_mu = df.mean()
#         gene_std = df.std()
#         gene_cv = gene_std/gene_mu
#         var_gene = gene_cv.sort_values(ascending=False)[:vargene].index.tolist()
        vscore = get_vscore(df)
        var_gene = vscore.sort_values(ascending=False)[:vargene].index.tolist()
        # 基于min correlation进行filter
        dfCor = df.loc[:, var_gene].corr()
        minCorIdx = (dfCor.abs()>minGeneCor).sum()>1
        # 删除housekeeping和cell cycle基因
        ccIdx = ~(dfCor.columns.str.upper()).isin(cell_cycle)
        hkIdx = ~(dfCor.columns.str.upper()).isin(housekeep)
        ccCorIdx = ~((dfCor.loc[:, ~ccIdx].abs()>excludeGeneCor).sum(axis=1)>0)
        hkCorIdx = ~((dfCor.loc[:, ~hkIdx].abs()>excludeGeneCor).sum(axis=1)>0)
        # keep idx
        keepIdx = minCorIdx & ccIdx & hkIdx & ccCorIdx & hkCorIdx
        dfFilter = df.loc[:, var_gene].loc[:, keepIdx]
        dfFilter_list.append(dfFilter)
    return(dfFilter_list)

def log_transform(df_list, pseudo_count=0.1):
    """Log transform the matrix
    :param data: Counts matrix: Cells x Genes
    :return: Log transformed matrix
    """
    logtrans = []
    for df in df_list:
        logtrans.append(np.log2(df + pseudo_count))
    return logtrans


####=====================================================================================================================
####  4. Single cell connection by KNN graph
####=====================================================================================================================

##-----------------------------------
##  4.1 KNN graph for each time point
##-----------------------------------
def get_knn_innner(Exp_list_HVG, pca_n, k=100):
    ret = []
    for i,df in enumerate(Exp_list_HVG):
        # zscore
        df_zscore = df.apply(zscore)
        
        # PCA
        if not pca_n:
            n_components = min(min(df.shape), 100)
        else:
            n_components = pca_n
        pca = PCA(n_components=n_components, svd_solver='full')
        df_zscore_pca = pca.fit_transform(df_zscore)
        sample_name = np.array(['t%s_%s'%(i, j) for j in range(df.shape[0])])

        # KNN graph
        nbrs = NearestNeighbors(n_neighbors=k+1, metric='correlation', n_jobs=-2)
        nbrs.fit(df_zscore_pca)
        dists, ind = nbrs.kneighbors(df_zscore_pca) # n*k_init dist & index
        adj = nbrs.kneighbors_graph(df_zscore_pca, mode='distance') # n*n dist
        ret.append([sample_name, dists[:, 1:], ind[:, 1:], adj])
    return ret

##----------------------------------------------
##  4.2 KNN graph for each neighbour time points
##----------------------------------------------
def get_knn_link(Exp_list_HVG, Exp_list_comb, pca_n, k=100):
    ret = []
    for i in range(len(Exp_list_comb)-1):
        # Same variable gene for data t and t+1
        time_before_gene = Exp_list_comb[i].columns
        time_after_var_gene = Exp_list_HVG[i+1].columns
        var_gene = time_after_var_gene.intersection(time_before_gene)
        print(len(var_gene))
        data_before = Exp_list_comb[i].loc[:,var_gene]
        data_after = Exp_list_comb[i+1].loc[:,var_gene]
        border = data_before.shape[0] # sample count
        
        # zscore
        data_before = data_before.apply(zscore)
        data_after = data_after.apply(zscore)
        
        # PCA for data t+1, and projection data t to t+1(need to define PC number)
        if not pca_n:
            n_components = min(min(data_after.shape), 100)
        else:
            n_components = pca_n
        pca = PCA(n_components=n_components, svd_solver='full')
        data_after_pca = pca.fit_transform(data_after)
        data_before_pca = pca.transform(data_before)
        sample_name = np.array(['t%s_%s'%(i, j) for j in range(data_before_pca.shape[0])]
                               +['t%s_%s'%(i+1, j) for j in range(data_after_pca.shape[0])])
        data_pca = np.vstack([data_before_pca, data_after_pca])
        
        # KNN distance
        nbrs = NearestNeighbors(n_neighbors=k+1, metric='correlation', n_jobs=-2)
        nbrs.fit(data_pca)
        dists, ind = nbrs.kneighbors(data_pca) # n*k_init dist & index
        adj = nbrs.kneighbors_graph(data_pca, mode='distance') # n*n dist
        # filter inner links
        idx_bool_before = np.apply_along_axis(lambda x:x>=border, 1, ind[:border,:])
        idx_bool_after = np.apply_along_axis(lambda x:x<border, 1, ind[border:,:])
        idx_bool = np.vstack([idx_bool_before, idx_bool_after])
        ret.append([sample_name, dists[:,1:], ind[:,1:], idx_bool[:, 1:], border, adj])

    return(ret)

##--------------------------------
##  4.3 Filter sing cell KNN graph
##--------------------------------
def filter_knn_graph(knn_inner, knn_link, k_final=20, inner_max_local=3, 
                     inner_max_global=-1, link_max_local=3, link_max_global=0, mutual=True):
    '''
    '''
    ## combine inner and link edges
    node1s = [];node2s = []
    D_origs = [];D_locals = [];D_globals = []
    inner_bools = [];outgoing_bools = []

    for inner in knn_inner:
        node1 = np.repeat(inner[0], inner[1].shape[1])
        node2 = inner[0][inner[2]].flatten()
        D_orig = inner[1].flatten()
        D_local = (inner[1]/inner[1][:,0][:,None]).flatten()
        D_global = zscore(D_orig)
        inner_bool = np.repeat(1, len(node1))
        outgoing_bool = np.repeat(1, len(node1))
        # append
        node1s.append(node1);node2s.append(node2)
        D_origs.append(D_orig);D_locals.append(D_local);D_globals.append(D_global)
        inner_bools.append(inner_bool);outgoing_bools.append(outgoing_bool)

    for link in knn_link:
        node1 = np.repeat(link[0], np.apply_along_axis(sum, 1, link[3]))
        node2 = link[0][link[2][link[3]]]
        node1ID = np.repeat(np.arange(len(link[0])), np.apply_along_axis(sum, 1, link[3]))
        node2ID = link[2][link[3]]
        D_orig = link[1][link[3]]
        D_local = (link[1]/link[1][:,0][:,None])[link[3]]
        D_global = zscore(link[1].flatten())[link[3].flatten()]
        inner_bool = np.repeat(0, len(node1))
        # outgoing, t0-t1:0, t1-t0:1
        border = link[4]
        before_after_outgoing = np.repeat(0, sum((node1ID<border) & (node2ID>=border)))
        after_before_outgoing = np.repeat(1, sum((node1ID>=border) & (node2ID<border)))
        outgoing_bool = np.hstack([before_after_outgoing, after_before_outgoing])
        # append
        node1s.append(node1);node2s.append(node2)
        D_origs.append(D_orig);D_locals.append(D_local);D_globals.append(D_global)
        inner_bools.append(inner_bool);outgoing_bools.append(outgoing_bool)

    # all edge combine DataFrame
    all_node1 = np.hstack(node1s)
    all_node2 = np.hstack(node2s)
    all_D_orig = np.hstack(D_origs)
    all_D_local = np.hstack(D_locals)
    all_D_global = np.hstack(D_globals)
    all_inner_bool = np.hstack(inner_bools)
    all_outgoing_bool = np.hstack(outgoing_bools)
    all_edges = pd.DataFrame({'Node1':all_node1, 'Node2':all_node2, 
        'D_orig':all_D_orig, 'D_local':all_D_local, 'D_global':all_D_global, 
        'inner_bool':all_inner_bool, 'Outgoing_bool':all_outgoing_bool})
    
    ## filter edge
    # filter by threshold
    inner_keep_idx = (all_edges.loc[:, 'inner_bool']==1) & (all_edges.loc[:, 'D_local']<=inner_max_local) & (all_edges.loc[:, 'D_global']<=inner_max_global)
    link_keep_idx = (all_edges.loc[:, 'inner_bool']==0) & (all_edges.loc[:, 'D_local']<=link_max_local) & (all_edges.loc[:, 'D_global']<=link_max_global)
    keep_idx = inner_keep_idx | link_keep_idx
    all_edges_filter1 = all_edges.loc[keep_idx, :].reset_index(drop=True)
    print('Filter percentage by local/global threshold: %.2f%%'%((1-sum(keep_idx)/len(keep_idx))*100))

    # filter mutual knn and outgoing
    if mutual:
        mutual_idx = all_edges_filter1.loc[:,['Node1', 'Node2']].apply(lambda x:x[:2].sort_values().str.cat(), axis=1).duplicated()
    else:
        mutual_idx = ~all_edges_filter1.loc[:,['Node1', 'Node2']].apply(lambda x:x[:2].sort_values().str.cat(), axis=1).duplicated()
    out_idx = all_edges_filter1.loc[:, 'Outgoing_bool']==1
    keep_idx = out_idx & mutual_idx
    all_edges_filter2 = all_edges_filter1.loc[keep_idx, :].reset_index(drop=True)
    print('Filter percentage by mutual/outgoing threshold: %.2f%%'%((1-sum(keep_idx)/len(keep_idx))*100))

    # node max outgoing threshold
    tmp = all_edges_filter2.copy()
    tmp.loc[:,'Node1'] = all_edges_filter2.loc[:,'Node2']
    tmp.loc[:,'Node2'] = all_edges_filter2.loc[:,'Node1']
    change_node_df = pd.concat([all_edges_filter2,tmp]).reset_index(drop=True)
    topk_idx = change_node_df.groupby('Node1')['D_orig'].nsmallest(k_final).reset_index().loc[:,'level_1']
    all_edges_filter3 = change_node_df.loc[topk_idx,:].reset_index(drop=True)
    uniq_idx = ~all_edges_filter3.loc[:,['Node1', 'Node2']].apply(lambda x:x[:2].sort_values().str.cat(), axis=1).duplicated()
    all_edges_filter3 = all_edges_filter3.loc[uniq_idx, :].reset_index(drop=True)
    
    def nodes_clean(rank_edges, k_final):
        '''remove node over k_final edges base on ranked D_orig value
        '''
        node_dic = {};idx = []
        for k,nodes in enumerate(rank_edges):
            node1_not_in = nodes[0] not in node_dic
            node2_not_in = nodes[1] not in node_dic
            node1_cnt_bool = nodes[0] in node_dic and node_dic[nodes[0]]<k_final
            node2_cnt_bool = nodes[1] in node_dic and node_dic[nodes[1]]<k_final
            if node1_not_in and node2_not_in:
                node_dic[nodes[0]]=1
                node_dic[nodes[1]]=1
                idx.append(k)
            if node1_not_in and node2_cnt_bool:
                node_dic[nodes[0]]=1
                node_dic[nodes[1]]+=1
                idx.append(k)
            if node2_not_in and node1_cnt_bool:
                node_dic[nodes[0]]+=1
                node_dic[nodes[1]]=1
                idx.append(k)
            if node1_cnt_bool and node2_cnt_bool:
                node_dic[nodes[0]]+=1
                node_dic[nodes[1]]+=1
                idx.append(k)
        return idx
    all_edges_filter3 = all_edges_filter3.sort_values('D_orig').reset_index(drop=True)
    rank_edges = all_edges_filter3.loc[:,['Node1','Node2']].values
    idx = nodes_clean(rank_edges, k_final)
    all_edges_filter3 = all_edges_filter3.loc[idx,:].reset_index(drop=True)
    
    print('Filter percentage by max outgoing threshold: %.2f%%'%((1-all_edges_filter3.shape[0]/all_edges_filter2.shape[0])*100))

    return [all_edges, all_edges_filter3]

##------------------------------------------
##  4.4 Convert edgelist to affinity Matrix
##------------------------------------------
def get_allnodes(inner_list):
    allnodes = []
    for inn in inner_list:
        allnodes.append(inn[0])
    allnodes = np.hstack(allnodes)
    return allnodes

def convert_edgeDF_adjMTX(edgeDF, allnodes):
    # adjMatrix construction
    n = len(allnodes)
    adjMTX = [[0 for i in range(n)] for k in range(n)]

    def adjMTX_fill(ser):
        node1,node2,aff = ser['Node1'],ser['Node2'],1-ser['D_orig']
        x = np.argwhere(allnodes==node1)[0][0]
        y = np.argwhere(allnodes==node2)[0][0]
        adjMTX[x][y] = aff
    edgeDF.apply(adjMTX_fill, 1)
    sparse_adjMTX = scipy.sparse.csr_matrix(adjMTX)
    return sparse_adjMTX

##-----------------------------------
##  4.5. Plot single cell KNN graph
##-----------------------------------
# layout
def force_directed_layout(G, pdf_name='Layout.network.pdf', cell_names=None, verbose=True, iterations=2000):
    """" Function to compute force directed layout from the G
    :param G: networkx graph object converted from sparse matrix representing affinities between cells
    :param cell_names: pandas Series object with cell names
    :param verbose: Verbosity for force directed layout computation
    :param iterations: Number of iterations used by ForceAtlas 
    :return: Pandas data frame representing the force directed layout
    """

    forceatlas2 = ForceAtlas2(
        # Behavior alternatives
        outboundAttractionDistribution=False,  
        linLogMode=False,  
        adjustSizes=False,  
        edgeWeightInfluence=1.0,
        # Performance
        jitterTolerance=1.0,  
        barnesHutOptimize=True,
        barnesHutTheta=1.2,
        multiThreaded=False,  
        # Tuning
        scalingRatio=2.0,
        strongGravityMode=False,
        gravity=1.0,
        # Log
        verbose=verbose)

    ## use affinity construct KNN graph
    positions = forceatlas2.forceatlas2_networkx_layout(G, pos=None, iterations=iterations)
    positions = np.array([list(positions[i]) for i in range(len(positions))])

    ## plot KNN graph
    f = plt.figure(figsize=(15,15))
    nx.draw_networkx_nodes(G, positions, node_size=20, with_labels=False, node_color="blue", alpha=0.4)
    nx.draw_networkx_edges(G, positions, edge_color="green", alpha=0.5)
    plt.axis('off')
    plt.show()
    f.savefig(pdf_name, bbox_inches='tight')

    ## Convert to dataframe
    if cell_names is None:
        cell_names = np.arange(affinity_matrix.shape[0])

    positions = pd.DataFrame(positions, index=cell_names, columns=['x', 'y'])
    positions.loc[:,'tp'] = pd.Series(cell_names).apply(lambda x:x.split('_')[0]).tolist()
    return positions

# different timepoint cell distribution
def plot_layout_tp(fa_cord, pdf_name='Layout.network.split.pdf', tp_color=sns.color_palette()):
    # unique timepoint
    tps = sorted(fa_cord.loc[:,'tp'].unique())
    N = len(tps)
    row,col = N//3+1, 3

    # plot tp distribution
    f = plt.figure(figsize=(18,7*row))
    plt.subplot(row,3,1)
    ax = sns.scatterplot(x="x", y="y", hue="tp", data=fa_cord) # tp combine
    for i in range(N):
        plt.subplot(row,3,i+2)
        plt.scatter(fa_cord.loc[:,'x'], fa_cord.loc[:,'y'], s=3, color='lightgrey')
        plt.scatter(fa_cord.loc[fa_cord.loc[:,'tp']==tps[i],'x'], fa_cord.loc[fa_cord.loc[:,'tp']==tps[i],'y'], 
                   s=5, color=tp_color[i])
    plt.show()
    f.savefig(pdf_name, bbox_inches='tight')


def get_Ggaint(knn_inner, knn_filter_edges):
    allnodes = get_allnodes(knn_inner)
    adjMTX = convert_edgeDF_adjMTX(knn_filter_edges, allnodes)

    G_all = nx.from_numpy_matrix(adjMTX.toarray())

    connected_node_sets = [list(c) for c in sorted(nx.connected_components(G_all), key=len, reverse=True)]
    gaint_idx = connected_node_sets[0]
    adjMTX_gaint = adjMTX[gaint_idx,:][:,gaint_idx]
    nodes_gaint = allnodes[gaint_idx]
    G_gaint = nx.from_numpy_matrix(adjMTX_gaint.toarray())

    return nodes_gaint


####=====================================================================================================================
####  5. Cell clusters
####=====================================================================================================================
def cell_clusters(DF, hvgs, pca_n, knn_n):
    # convert data
    adata = sc.AnnData(DF)
    adata.obs_names_make_unique()
    adata.var_names_make_unique()
    # MT pct
    mito_genes = adata.var_names.str.startswith('MT-')
    adata.obs['percent_mito'] = np.sum(adata[:, mito_genes].X, axis=1) / np.sum(adata.X, axis=1)
    adata.obs['n_counts'] = adata.X.sum(axis=1)
    # normalize log-transform
    sc.pp.normalize_per_cell(adata, counts_per_cell_after=1e4)
    sc.pp.log1p(adata)
    # high variable genes
    sc.pp.highly_variable_genes(adata, min_mean=0.0125, max_mean=3, min_disp=0.5)
    # adata.var['highly_variable'] = False
    # adata.var['highly_variable'][hvgs] =True
    # adata_high = adata[:, adata.var['highly_variable']]
    # linear regression
    sc.pp.regress_out(adata_high, ['n_counts', 'percent_mito'])
    sc.pp.scale(adata_high, max_value=10)
    # pca
    sc.tl.pca(adata_high, n_comps=pca_n, svd_solver='arpack')
    # knn
    sc.pp.neighbors(adata_high, n_neighbors=knn_n, n_pcs=pca_n)
    sc.tl.louvain(adata_high, resolution=1)
    sc.tl.umap(adata_high)
    # sc.pl.umap(adata_high, color='louvain')
#     # sub cluster in main clusters
#     main_cluster = sorted(adata.obs.louvain.unique().tolist())
#     for i in main_cluster:
#         sc.tl.louvain(adata, resolution=1, restrict_to=('louvain', [i]))
#         adata.obs.louvain_R = adata.obs.louvain_R.astype('str')
#         adata.obs = adata.obs.rename(columns={'louvain_R':'louvain_%s'%i})
    return adata_high


####=====================================================================================================================
####  6. Single-cell KNN graph coarse grain
####=====================================================================================================================
def convert_scKNN_cgKNN(CTs, scknn_edge, CT_time):
    '''convert single cell KNN graph to celltype KNN graph base on cluster ID.
    '''
    newNode1 = []
    newNode2 = []
    edge_cnt = []
    nodes_cnt = {}
    for i in CTs:
        # specific celltype connected edges
        type_idx = scknn_edge.apply(lambda x:(x['Node1_celltype']==i) or (x['Node2_celltype']==i), axis=1)
        type_edge = scknn_edge.loc[type_idx,:]

        # specific celltype connected celltypes and each counts
        neighbour_nodes = type_edge.loc[:,['Node1_celltype','Node2_celltype']].values.flatten()
        neighbour_nodes = [j for j in neighbour_nodes if j!=i and j!='None']
        neighbour_nodes_cnt = pd.Series(neighbour_nodes).value_counts()
        
        # total counts of specific celltype connected 
        total_cnt = sum(neighbour_nodes_cnt)
        nodes_cnt[i] = total_cnt
        
        # construct specific celltype edges to other celltypes
        newNode1 += [i]*len(neighbour_nodes_cnt)
        newNode2 += neighbour_nodes_cnt.index.tolist()
        edge_cnt += neighbour_nodes_cnt.values.tolist()
    cgKNN_edges = pd.DataFrame({'Node1':newNode1, 'Node2':newNode2, 'Edge_cnt':edge_cnt})
    cgKNN_edges.loc[:,'Node_edge_cnt'] = cgKNN_edges.apply(lambda x: nodes_cnt[x['Node1']]+nodes_cnt[x['Node2']], axis=1)
    cgKNN_edges.loc[:, 'Weight'] = cgKNN_edges.loc[:,'Edge_cnt']/cgKNN_edges.loc[:,'Node_edge_cnt']
    cgKNN_edges.loc[:, 'Edge_times'] = cgKNN_edges.apply(lambda x:max(CT_time[x['Node1']], CT_time[x['Node2']]), axis=1)
    return cgKNN_edges


def convert_edgeDF_adjMTX_CGgaint(edgeDF, allnodes):
    '''Conver edgeDF to adjMatrix(for celltype edges)
    '''
    # adjMatrix construction
    n = len(allnodes)
    adjMTX = [[0 for i in range(n)] for k in range(n)]

    def adjMTX_fill(ser):
        node1,node2,aff = ser['Node1'],ser['Node2'],ser['Weight']
        x = np.argwhere(allnodes==node1)[0][0]
        y = np.argwhere(allnodes==node2)[0][0]
        adjMTX[x][y] = aff
    edgeDF.apply(adjMTX_fill, 1)
    sparse_adjMTX = scipy.sparse.csr_matrix(adjMTX)
    return sparse_adjMTX


def scaffold_prune(edge):
    '''Prune celltype edgeDF to scaffold. From last time to first time.
    Stop remove edge until connnected component increase or just one edge left.
    '''
    edge_time = sorted(edge.Edge_times.unique(), reverse=True)
    scaffold_list = []
    for t in edge_time:
        edge_t = edge.loc[edge.loc[:,'Edge_times']==t,:]
        edge_t = edge_t.sort_values('Weight')
        nodes_t = np.unique(edge_t.loc[:,['Node1','Node2']].values.flatten())
        adjMTX_t = convert_edgeDF_adjMTX_CGgaint(edge_t, nodes_t)
        G_t = nx.from_numpy_matrix(adjMTX_t.toarray())
        ncomponent_t = nx.number_connected_components(G_t)
        ret_i = edge_t.shape[0]-1 # 保留最后一行
        for i in range(1, edge_t.shape[0]):
            edge_t_del = edge_t.iloc[i:,:]
            nodes_t_del = np.unique(edge_t_del.loc[:,['Node1','Node2']].values.flatten())
            adjMTX_t_del = convert_edgeDF_adjMTX_CGgaint(edge_t_del, nodes_t_del)
            G_t_del = nx.from_numpy_matrix(adjMTX_t_del.toarray())
            ncomponent_t_del = nx.number_connected_components(G_t_del)
            if ncomponent_t_del>ncomponent_t:
                ret_i = i-1
                break
        scaffold_list.append(edge_t.iloc[ret_i:,:])
        print(t, ret_i)
    scaffoldDF = pd.concat(scaffold_list).sort_index()

    # # convert edgeDF to graph
    # scaffold_edge_idx = edge.index.isin(scaffoldDF.index)
    # nodes_scaffold = np.unique(scaffoldDF.loc[:,['Node1','Node2']].values.flatten())
    # adjMTX_scaffold = convert_edgeDF_adjMTX_CGgaint(scaffoldDF, nodes_scaffold)
    # G_scaffold = nx.from_numpy_matrix(adjMTX_scaffold.toarray())
    # mapping = {k:v for k,v in enumerate(nodes_scaffold)}
    # G_scaffold = nx.relabel_nodes(G_scaffold, mapping)
    # return scaffoldDF,scaffold_edge_idx,adjMTX_scaffold,nodes_scaffold,G_scaffold
    return scaffoldDF


def directed_edge(x):
    '''Convert undirected celltype edges to directed.
    Attention:
        celltype_times_adj is a global var.
    '''
    if celltype_times_adj[x['Node1']]<celltype_times_adj[x['Node2']]:
        fromnode = x['Node1']
        tonode = x['Node2']
    else:
        fromnode = x['Node2']
        tonode = x['Node1']
    ret = pd.Series({'From':fromnode, 'to':tonode, 'weight':x['Weight']})
    return ret


def plot_scaffold(fa_cord_all, conn_edges, pdfnm):
    fig = plt.figure(figsize=(5,5))
    ax = fig.add_subplot(111)
    cluster_cord = fa_cord_all.apply(lambda a:[a['x'], a['y']], axis=1).to_dict()
    for k,v in cluster_cord.items():
        plt.annotate(k, (v[0]-0.2, v[1]-0.2))
    N = len(set(conn_edges.loc[:,'From'].tolist()+conn_edges.loc[:,'to'].tolist()))
    sns.scatterplot(x='x', y='y', hue='Cluster', data=fa_cord_all, palette=sns.color_palette("husl", n_colors=N), s=600)
    for row in conn_edges.index:
        From = conn_edges.loc[row,'From']
        To = conn_edges.loc[row,'to']
        weight = conn_edges.loc[row,'weight']
        From_cord = cluster_cord[From]
        To_cord = cluster_cord[To]
#         plt.plot([From_cord[0], To_cord[0]], [From_cord[1], To_cord[1]], 'o-', color='grey', linewidth=weight*10)
#     plt.Arrow(x,y,w,h,width=0.05,zorder=i+1)
#     a.set_facecolor('0.7')
#     a.set_edgecolor('w')
        radius = math.sqrt(600)/2.
        arrow = mpl.patches.FancyArrowPatch(posA=(From_cord[0],From_cord[1]), posB=(To_cord[0],To_cord[1]), 
                                    arrowstyle='-|>', mutation_scale=20, color='grey', linewidth=weight*15, 
                                    shrinkA=radius, shrinkB=radius)
        ax.add_patch(arrow)


    plt.xlim(min(fa_cord_all.loc[:, 'x'])-5, max(fa_cord_all.loc[:, 'x'])+5)
    plt.ylim(min(fa_cord_all.loc[:, 'y'])-5, max(fa_cord_all.loc[:, 'y'])+5)
    plt.xlabel('Force directed layout x')
    plt.ylabel('Force directed layout y')
    plt.savefig(pdfnm)


def get_cell_trajectory(node_celltype, nodes_gaint, knn_filter_edges):

    gaint_idx = knn_filter_edges.apply(lambda x:((nodes_gaint==x[0]) | (nodes_gaint==x[1])).any(), axis=1)
    gaint_edge = knn_filter_edges.loc[gaint_idx,:]

    # Add node1/node2 celltypes 
    gaint_edge.loc[:,'Node1_celltype'] = gaint_edge.loc[:,'Node1'].map(node_celltype.to_dict()['celltype'])
    gaint_edge.loc[:,'Node2_celltype'] = gaint_edge.loc[:,'Node2'].map(node_celltype.to_dict()['celltype'])

    #--------- 5.2 Get celltype nodes attributes ---------
    # 每个细胞类型的平均时间(除None)
    node_celltype_gaint = node_celltype.loc[nodes_gaint,:]
    node_celltype_gaint.loc[:,'tp'] = node_celltype_gaint.index.to_series().apply(lambda x:int(x.split('_')[0][1:]))
    celltype_times_adj = node_celltype_gaint.loc[node_celltype_gaint.loc[:,'celltype']!='None',:].groupby('celltype')['tp'].mean()

    # 每个细胞类型的数量
    CG_node_sizes = node_celltype_gaint.loc[:,'celltype'].value_counts()

    # remove self-self connection(same celltype)
    nonself_idx = gaint_edge.apply(lambda x:x['Node1_celltype']!=x['Node2_celltype'], axis=1)
    gaint_edge_nonself = gaint_edge.loc[nonself_idx,:]

    # 所有细胞类型(除None)
    all_celltypes = node_celltype_gaint.loc[:,'celltype'].unique()
    all_celltypes = [i for i in all_celltypes if i!='None']

    #--------- 5.3 Convert sc-edgeDF to celltype-edgeDF ---------
    celltype_edges = convert_scKNN_cgKNN(all_celltypes, gaint_edge_nonself, celltype_times_adj)
    # weak_edge_idx = celltype_edges.loc[:,'Weight']<0.03
    # uniq_idx = ~celltype_edges.loc[:,'Edge_cnt':'Edge_times'].duplicated()

    #--------- 5.4 Find scaffold edges ---------
    scaffold_edge = scaffold_prune(celltype_edges)
    uniq_idx = ~scaffold_edge.loc[:,'Edge_cnt':'Edge_times'].duplicated()
    scaffold_edge = scaffold_edge.loc[uniq_idx, :]
    
    return scaffold_edge

def plot_cell_trajectory(cell_trajectoryG):
    conn_edges = cell_trajectoryG
    allnodes = np.array(sorted(list(set(conn_edges.loc[:, 'From'].tolist()+conn_edges.loc[:, 'to'].tolist()))))
    adjMTX = convert_edgeDF_adjMTX(conn_edges, allnodes)
    G_all = nx.from_numpy_matrix(adjMTX.toarray())

    # All nodes layout plot
    fa_cord_all = force_directed_layout(G=G_all, cell_names=allnodes)

    # plot force directed layout scaffold
    plot_scaffold(fa_cord_all, conn_edges, 'cell.trajectory.pdf')
