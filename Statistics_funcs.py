import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import pingouin as pg
import itertools as iter

# Calulate ANOVA & post-hoc results
def anova_spk(datatype="burst_data", rpath="./data/", spath="./results/"):
    if datatype == "burst_data":
        dataset = pd.read_csv(rpath+"Spk_char_burst.csv")
    elif datatype == "nonb_data":
        dataset = pd.read_csv(rpath+"Spk_char_nonb.csv")

    for spk_char in ["Kurtosis", "FF", "Burstiness", "BI", "LV", "FR"]:
        # Check equality of variance between groups
        eq_var = pg.homoscedasticity(dv = spk_char, group = "Session", data = dataset)
        pval   = eq_var["pval"].item() 

        # With the result of eq_Var, use welch-anova or anova
        if pval <= .05:
            anova = pg.welch_anova(
                dv      = spk_char, 
                between = "Session", 
                data    = dataset
            )
            posthoc = pg.pairwise_gameshowell(
                dv      = spk_char, 
                between = "Session", 
                data    = dataset
            )
        elif pval > .05:
            anova = pg.anova(
                dv      = spk_char, 
                between = "Session", 
                data    = dataset
            )
            posthoc = pg.pairwise_tukey(
                dv      = spk_char, 
                between = "Session", 
                data    = dataset
            )

        anova["Source"] = spk_char
        char_name    = pd.DataFrame(dict(parameter=spk_char), index=[0])
        posthoc      = pd.concat([char_name, posthoc], axis=1)
        path_anova   = spath+"anova_"+datatype+".csv"
        path_posthoc = spath+"posthoc_"+datatype+".csv"
        
        if spk_char == "kurt":
            anova.to_csv(path_anova, mode='w', index=False)
            posthoc.to_csv(path_posthoc, mode='w', index=False)
        else:
            anova.to_csv(path_anova, mode='a', index=False)
            posthoc.to_csv(path_posthoc, mode='a', index=False)

# Make burst-nonb violin figure with stim Sessione x-axis
def mkfig_spk(rpath="./data/", spath="./results/"):
    '''
    seaborn catplot by violin style
    It is violin plot part, if you want plot both dataset in same figure
    to compare each other.
    Also, split option will make burst/non-burst graph sticked each other.
    '''
    dataset = pd.read_csv(rpath+"Spk_char_whole.csv")
    colors = ["#EB7979", "#63CEFF"]
    for spk_char in ["Kurtosis", "FF", "Burstiness", "BI", "LV", "FR"]:
        # custom_params = {"axes.spines.right": False, "axes.spines.top": False}
        # sns.set_theme(style="ticks", rc=custom_params, font_scale=1.5)
        # sns.set_theme(rc=custom_params, font_scale=1.5)
        sns.set_theme(style="ticks", font='Dejavu Sans', palette=colors, font_scale=1.5)

        ax_c = sns.catplot(x="Session", y=spk_char,  
                        hue="type", data=dataset, 
                        kind="violin", density_norm="area", 
                        split=True, height=6, 
                        inner="quart", gap=0.1,
                        palette=colors , aspect=1.1)
        ax_c.set(xlabel = "Session", ylabel = spk_char)            
        plt.savefig(spath+"Spk_para_"+spk_char+"_violin_fig.svg")
        plt.clf()
        plt.close()

    '''
    seaborn boxplot by strip style
    For boxplot, we need to split the dataset by burst/non-burst
    1: burst, 2: non-burst
    '''
    '''
    dataset_burst = pd.read_csv(rpath+"Spk_char_burst.csv")
    dataset_nonb  = pd.read_csv(rpath+"Spk_char_nonb.csv")
    for idx in [1, 2]:
        if idx == 1:
            dataset = dataset_burst
        elif idx == 2:
            dataset = dataset_nonb
        
        for spk_char in ["Kurtosis", "FF", "Burstiness", "BI", "LV", "FR"]:
            colors = [
                "#DA4453", "#967ADC", "#E9573F", "#4A89DC",
                "#D770AD", "#8CC152", "#F6BB42"
            ]
            # sns.set_palette(sns.color_palette(colors))
            sns.set_theme(style="ticks", font='DejaVu Sans', palette=colors)

            ax = sns.boxplot(
                x="Session", y=spk_char, 
                showfliers=False, linewidth=1.5,
                data=dataset
            )
            ax = sns.stripplot(
                x="Session", y=spk_char, 
                jitter=True, size=6, zorder=0, 
                alpha=0.5, linewidth =1, 
                palette=colors, data=dataset
            )
            
            lower=(math.floor(ax.get_ylim()[0]))
            upper=(math.ceil(ax.get_ylim()[1]))+3

            plt.xticks(fontsize=14)
            plt.yticks(np.arange(lower, upper, step=2), fontsize=14)
            ax.tick_params(width=1)
            ax.set_xlabel(xlabel = "MER session", fontsize=14)
            ax.set_ylabel(ylabel = spk_char, fontsize=14)
            ax.figure.tight_layout()

            sns.despine(top = True, right = True)
            plt.show()
            
            if idx == 1:
                plt.savefig(spath+"Spk_para_"+"burst_"+spk_char+"_fig.png")
            elif idx == 2:
                plt.savefig(spath+"Spk_para_"+"nonb_"+spk_char+"_fig.png")
            plt.clf()
    '''


# Calculate mulicollinearity between variables
def multcoll(datatype="burst_data", rpath = './data/', spath="./results/"):
    if datatype   == "burst_data":
        dataset = pd.read_csv(rpath+"Spk_char_burst_before.csv")
        sigpara =  ['Session', 'Burstiness', 'Kurtosis', 'FF', 'BI', 'LV', 'FR']
    elif datatype == "nonb_data":
        dataset = pd.read_csv(rpath+"Spk_char_nonb_before.csv")

    # colors = [
    #     "#DA4453", "#967ADC", "#E9573F", "#4A89DC",
    #     "#D770AD", "#8CC152", "#F6BB42"
    # ]
    colors = [
        "#DA4453"
    ]
    sns.set_theme(
        rc={
            "font.family": "sans-serif",
            "font.sans-serif": ["Helvetica", "Arial"],
            "font.size": 7,
            "axes.labelsize": 7,
            "xtick.labelsize": 6.5,
            "ytick.labelsize": 6.5,
            "legend.fontsize": 6.5,
        }
    )

    ax     = sns.pairplot(dataset[sigpara], hue = "Session", palette=colors)
    # sns.set(font_scale=2)

    row_idx = 0
    col_idx = 0
    row = [0, 1, 2, 3, 4]
    col = [1, 2, 3, 4, 5]
    init_para = 'Burstiness'
    props = dict(boxstyle='round', facecolor='white', alpha=0.5)

    for pair in iter.combinations(sigpara[1:], 2):
        corr_coef = np.corrcoef(dataset[pair[0]], dataset[pair[1]])
        if pair[0] == 'Burstiness':
            print("Correlation coefficient between %s and %s: %.3f" % (pair[0], pair[1], corr_coef[0,1]))
        textstr = "\n".join(["r = %.3f" % corr_coef[0,1]])

        if pair[0] != init_para:
            init_para = pair[0]
            row_idx += 1
            col_idx = row_idx

        # if row_idx == 0 and col_idx==0:
        #     ax.axes[row[row_idx],col[col_idx]].text(
        #         -5, 3, textstr, 
        #         ha='left', va='center',
        #         fontsize=14, bbox=props
        #     )
        # elif row_idx == 0 and col_idx==1:
        #     ax.axes[row[row_idx],col[col_idx]].text(
        #         -5, 3, textstr, 
        #         ha='left', va='center',
        #         fontsize=14, bbox=props
        #     )
        # elif row_idx == 0 and col_idx==2:
        #     ax.axes[row[row_idx],col[col_idx]].text(
        #         -5, 3, textstr, 
        #         ha='left', va='center',
        #         fontsize=14, bbox=props
        #     )
        # elif row_idx == 0 and col_idx==3:
        #     ax.axes[row[row_idx],col[col_idx]].text(
        #         -5, 3, textstr, 
        #         ha='left', va='center',
        #         fontsize=14, bbox=props
        #     )
        # elif row_idx == 0 and col_idx==4:
        #     ax.axes[row[row_idx],col[col_idx]].text(
        #         -5, 3, textstr, 
        #         ha='left', va='center',
        #         fontsize=14, bbox=props
        #     )
        # col_idx += 1
    plt.savefig(spath+"Spk_para_multcoll.png", dpi=300)
    plt.close()
    # plt.show()

    # df = dataset[sigpara].dropna()
    # features = "+".join(df.columns[1:])
    # features = "+".join(np.concatenate((df.columns[1:2],df.columns[3:]), axis=None))
    # y, X = dmatrices('Session ~' + features, df, return_type='dataframe')

    # vif = pd.DataFrame()
    # vif["VIF Factor"] = [variance_inflation_factor(X.values, i) for i in range(X.shape[1])]
    # vif["features"] = X.columns

    return

def paired_updrs(spath="./results/", rpath="./data/"):
    # updrs_data = pd.read_csv(rpath+"UPDRS_score.csv")
    updrs_data = pd.read_csv(rpath+"LEDD.csv")
    group1 = 'PS'
    group2 = 'Both-on'
        
    # colors = ["#F26666", "#F28705", "#7ED1F2", "#0FF207"]
    colors = ["#F26666", "#0FF207"]
    sns.set_theme(
        rc={
            "font.family": "sans-serif",
            "font.sans-serif": ["Helvetica", "Arial"],
            "font.size": 7,
            "axes.labelsize": 7,
            "xtick.labelsize": 6.5,
            "ytick.labelsize": 6.5,
            "legend.fontsize": 6.5,
        }
    )

    fig, ax = plt.subplots(figsize=(4,4), dpi=300)
    # fig, ax = plt.plot()
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    sns.set_palette(sns.color_palette(colors))
    sns.set_context("notebook", font_scale= 3)
    sns.boxplot(
        data       = updrs_data[[group1, group2]],
        zorder     = 0,
        width      = 0.3,
        linewidth  = 0.8,
        showfliers = False,
        ax         = ax
    )
    sns.stripplot(
        data      = updrs_data[[group1, group2]],
        size      = 8,
        zorder    = 2,
        alpha     = 0.8,
        linewidth = 1,
        jitter    = False,
        ax        = ax
    )
    for idx in updrs_data.index:
        ax.plot(
            updrs_data.loc[idx,[group1, group2]],
            color     = 'grey',
            zorder    = 1,
            linewidth = 1, 
            linestyle = '--'
        )

    # ax.set_xticks([-0.5, 0, 1 ,1.5])
    plt.ylim([0, 80])
    plt.yticks([0, 1500, 3000])
    # plt.ylabel("UPDRS-Ⅲ")
    plt.ylabel("LEDD")
    plt.tight_layout()
    plt.savefig(spath+"paired_updrs.svg", dpi=300)
    plt.close()
    return

def anova_updrs(rpath="./data/", spath="./results/"):
    updrs_data = pd.read_csv(rpath+"UPDRS_for_anova.csv")
    updrs_data = updrs_data.dropna()
    
    # Check equality of variance between groups
    updrs = "UPDRS"
    eq_var = pg.homoscedasticity(
        dv = updrs, 
        group = "condition",
        data = updrs_data
    )
    pval   = eq_var["pval"].item() 

    # With the result of eq_Var, use welch-anova or anova
    if pval <= .05:
        anova = pg.welch_anova(
            dv      = updrs, 
            between = "condition", 
            data    = updrs_data
        )
        posthoc = pg.pairwise_gameshowell(
            dv      = updrs, 
            between = "condition", 
            data    = updrs_data
        )
    elif pval > .05:
        anova = pg.anova(
            dv      = updrs, 
            between = "condition", 
            data    = updrs_data
        )
        posthoc = pg.pairwise_tukey(
            dv      = updrs, 
            between = "condition", 
            data    = updrs_data
        )

    anova["Source"] = updrs
    char_name    = pd.DataFrame(dict(parameter=updrs), index=[0])
    posthoc      = pd.concat([char_name, posthoc], axis=1)
    path_anova   = spath+"anova_updrs.csv"
    path_posthoc = spath+"posthoc_updrs.csv"
    
    anova.to_csv(path_anova, mode='w', index=False)
    posthoc.to_csv(path_posthoc, mode='w', index=False)
    
    return