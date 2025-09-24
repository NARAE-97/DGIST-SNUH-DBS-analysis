import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import pickle

from sklearn import svm
from sklearn.metrics import confusion_matrix, RocCurveDisplay, auc
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV, \
                                    StratifiedKFold
from sklearn.metrics import roc_curve

def load_and_preprocess(csv_path):
    df = pd.read_csv(csv_path)
    biomarkers = ['Kurtosis', 'FF', 'Burstiness', 'BI']

    # aggregate by taking the maximum value for each biomarker
    df = df.groupby(['name', 'hemi', 'Session', 'ch', 'type'])[biomarkers].max().reset_index()
    return df

def prepare_model1_data(df):
    """
    Model 1: Predict type (Burst or Non-burst) using biomarkers from the Before session
    """
    df_before = df[df['Session'] == 'Before'].copy()
    X = df_before[['Kurtosis', 'FF', 'Burstiness', 'BI']].values
    y = df_before['type'].map({'Burst': 1, 'Non-burst': 0}).values

    # Find indices where X has no NaN values and remove corresponding y values
    X_cleaned = pd.DataFrame(X).dropna()
    indices_to_keep = X_cleaned.index
    y_cleaned = y[indices_to_keep]

    # Use NaN-removed data for X and y
    X = X_cleaned.values
    y = y_cleaned
    
    if X.shape[0] != y.shape[0]:
        raise ValueError("The number of samples in X and y do not match.")
    
    return X, y

def prepare_model2_data(df, label_csv_path="./data/Spk_char_label.csv"):
    """
    Model 2: Compute the biomarker change (diff) for each channel under stimulation current conditions 
    (D-1 mA, D-2 mA, D-3 mA) relative to the “Before” state, then label them according to the optv value 
    recorded in the CSV file.
    
    Labeling rules:
      - If optv == 3: diff_d3 → 1, diff_d1 & diff_d2 → 0
      - If optv == 2: diff_d3 & diff_d2 → 1, diff_d1 → 0
      - If optv == 1: diff_d3, diff_d2, diff_d1 all → 1
      - For “Non-burst” type: always label 0 (False)
      
    Returns:
      X: a numpy array containing biomarker diff values for each example (channel × stimulation condition)
      y: the label for each example (1: True, 0: False)
    """    
    # Lead optimal voltage values from CSV file
    label_df = pd.read_csv(label_csv_path)
    label_dict = {f"{row['name']}_{row['hemi']}": row['optv'] for _, row in label_df.iterrows()}

    # Get the burst and non-burst channel information in the 'Before' session
    df_before = df[df['Session'] == 'Before'].copy()
    
    # Include both Burst and Non-burst channels
    df_burst_nonb_channels = df_before[['name', 'hemi', 'ch', 'type']]
    df_burst_nonb = pd.merge(df, df_burst_nonb_channels, on=['name', 'hemi', 'ch', 'type'])
    
    biomarkers = ['Kurtosis', 'FF', 'Burstiness', 'BI']

    # Generate pivot table for easier diff calculation
    df_pivot = df_burst_nonb.pivot_table(
        index=['name', 'hemi', 'ch', 'type'],
        columns=['Session'],
        values=biomarkers
    ).reorder_levels([1, 0], axis=1).sort_index(axis=1)
    
    # Calculate differences between 'Before' and each stimulation session
    diff_d1v = df_pivot['Before'] - df_pivot['D-1V']
    diff_d2v = df_pivot['Before'] - df_pivot['D-2V']
    diff_d3v = df_pivot['Before'] - df_pivot['D-3V']

    X_list = []
    y_list = []
    
    # For each channel, set labels based on optv and type
    for idx in df_pivot['Before'].index:
        # idx: (name, hemi, ch)
        key = f"{idx[0]}_{idx[1]}"
        opt = label_dict.get(key, None)
        
        # If "Non-burst", always label 0
        if idx[-1] == 'Non-burst':
            X_list.append(diff_d1v.loc[idx])
            y_list.append(0)
            X_list.append(diff_d2v.loc[idx])
            y_list.append(0)
            X_list.append(diff_d3v.loc[idx])
            y_list.append(0)
            continue
        
        elif idx[-1] == 'Burst':
        # If "Burst", set labels based on optv
            if opt is None:
                # Without optv info, skip this channel
                continue
            
            # Set labels based on optv value
            label_d1 = 1 if opt <= 1 else 0
            label_d2 = 1 if opt <= 2 else 0
            label_d3 = 1 if opt <= 3 else 0

            # Append biomarker diffs and corresponding labels to lists
            X_list.append(diff_d1v.loc[idx])
            y_list.append(label_d1)
            X_list.append(diff_d2v.loc[idx])
            y_list.append(label_d2)
            X_list.append(diff_d3v.loc[idx])
            y_list.append(label_d3)
    
    X = np.vstack(X_list)
    y = np.array(y_list)

    # Find indices where X has no NaN values and remove corresponding y values
    X_cleaned = pd.DataFrame(X).dropna()
    indices_to_keep = X_cleaned.index
    y_cleaned = y[indices_to_keep]

    # Use NaN-removed data for X and y
    X = X_cleaned.values
    y = y_cleaned
    
    if X.shape[0] != y.shape[0]:
        raise ValueError("The number of samples in X and y do not match.")
    
    return X, y

def plot_conf_mat(true_y, pred_y):
    # Get confusion matrix
    cf_matrix = confusion_matrix(true_y, pred_y, normalize='all')
    ax = sns.heatmap(
        cf_matrix, 
        annot=True, 
        fmt='.2%', 
        cmap='Blues',
        annot_kws={"fontsize":12}
    )
    ax.set_xlabel('Predicted Values', fontsize=12)
    ax.set_ylabel('Actual Values', fontsize=12)
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)

    ## Ticket labels - List must be in alphabetical order
    ax.xaxis.set_ticklabels(['Non-effective','Effective'])
    ax.yaxis.set_ticklabels(['Non-effective','Effective'])
    
    ## Display the visualization of the Confusion Matrix.
    plt.show()

    return

def plot_roc(true_y, pred_y):
    fprs, tprs, thresholds = roc_curve(true_y, pred_y)

    # Plot chance level line
    plt.figure(figsize=(8,6))
    plt.plot([0,1],[0,1],label='Chance')

    # Plot ROC
    plt.plot(fprs,tprs,label='ROC')
    plt.xlabel('False Positive Rate', fontsize=12)
    plt.xticks(fontsize=12)
    plt.ylabel('True Positive Rate', fontsize=12)
    plt.yticks(fontsize=12)
    plt.legend(fontsize=12)
    plt.grid()
    plt.show()

    return


def SVM_train(X_train, y_train, nested = False):
    clf = svm.SVC()
    pipe = Pipeline([('scaler',StandardScaler()), ('svc',clf)])
    params = {
        'svc__max_iter' : [-1],
        'svc__probability' : [True],
        'svc__kernel': ['rbf'],
        'svc__C'     : np.linspace(1e-1,2,30),
        'svc__gamma' : np.linspace(1e-1,2,30),
        'svc__class_weight' : ['balanced'],
        'svc__random_state' : [24],
    }
    mdl = GridSearchCV(
        pipe,
        params,
        cv = StratifiedKFold(n_splits=10, shuffle=True, random_state=24),
    )

    # Training model
    if nested == False:
        mdl.fit(X_train, y_train)
        print('Gridsearch best parameters : ', mdl.best_params_)
        print("LOO best score: " + str(mdl.best_score_))
        return mdl
    
    if nested == True:
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=24)

        tprs = []
        aucs = []
        mean_fpr = np.linspace(0, 1, 100)
        conf_mat = np.array([[0, 0], [0, 0]], dtype='int64')
        
        sensitivity_list = []
        specificity_list = []

        # Create a figure for ROC curves
        fig, ax = plt.subplots(figsize=(12, 6))

        for idx, (train, test) in enumerate(cv.split(X_train, y_train)):
            mdl.fit(X_train[train], y_train[train])
            
            viz = RocCurveDisplay.from_estimator(
                mdl,
                X_train[test],
                y_train[test],
                name ='ROC fold {}'.format(idx),
                ax=ax,
                curve_kwargs={'alpha':0.3, 'lw':1}
            )
            interp_tpr = np.interp(mean_fpr, viz.fpr, viz.tpr)
            interp_tpr[0] = 0.0
            tprs.append(interp_tpr)
            aucs.append(viz.roc_auc)
            
            conf_tmp = confusion_matrix(y_train[test], mdl.predict(X_train[test]))
            conf_mat = conf_mat + conf_tmp

            # Calculate sensitivity and specificity
            tn, fp, fn, tp = conf_tmp.ravel()
            sensitivity = tp / (tp + fn) if (tp + fn) != 0 else 0
            specificity = tn / (tn + fp) if (tn + fp) != 0 else 0

            sensitivity_list.append(sensitivity)
            specificity_list.append(specificity)

        # ROC curve plot
        ax.plot([0, 1], [0, 1], linestyle="--", lw=2, color="r", label="Chance", alpha=0.8)

        mean_tpr = np.mean(tprs, axis=0)
        mean_tpr[-1] = 1.0
        mean_auc = auc(mean_fpr, mean_tpr)
        std_auc = np.std(aucs)
        ax.plot(
            mean_fpr,
            mean_tpr,
            color="b",
            label=r"Mean ROC (AUC = %0.2f $\pm$ %0.2f)" % (mean_auc, std_auc),
            lw=2,
            alpha=0.8
        )

        std_tpr = np.std(tprs, axis=0)
        tprs_upper = np.minimum(mean_tpr + std_tpr, 1)
        tprs_lower = np.maximum(mean_tpr - std_tpr, 0)
        ax.fill_between(
            mean_fpr,
            tprs_lower,
            tprs_upper,
            color="grey",
            alpha=0.2,
            label=r"$\pm$ 1 std. dev.",
        )

        ax.set(
            xlim=[-0.05, 1.05],
            ylim=[-0.05, 1.05],
        )
        ax.legend(loc="lower right", fontsize=12)
        ax.set_xlabel('False Positive Rate', fontsize=12)
        ax.set_ylabel('True Positive Rate', fontsize=12)
        plt.xticks(fontsize=12)
        plt.yticks(fontsize=12)
        plt.show()
        
        # Confusion matrix with percentage plot
        conf_1 = (conf_mat[0, 0]/(conf_mat[0, 0]+conf_mat[0, 1]))
        conf_2 = (conf_mat[0, 1]/(conf_mat[0, 0]+conf_mat[0, 1]))
        conf_3 = (conf_mat[1, 0]/(conf_mat[1, 0]+conf_mat[1, 1]))
        conf_4 = (conf_mat[1, 1]/(conf_mat[1, 0]+conf_mat[1, 1]))
        conf_mat_per = np.array([[conf_1, conf_2], [conf_3, conf_4]])

        # Create a heatmap for the confusion matrix with percentage values
        ax = sns.heatmap(
            conf_mat_per, 
            annot=True, 
            fmt='.1%',
            cmap='Blues',
            annot_kws={"fontsize":12},
        )
        cbar = ax.collections[0].colorbar
        cbar.ax.tick_params(labelsize=12)      # tick label font size
        ax.set_xlabel('Predicted', fontsize=12)
        ax.set_ylabel('Actual', fontsize=12)
        plt.xticks(fontsize=12)
        plt.yticks(fontsize=12)
        ax.xaxis.set_ticklabels(['Non-effective','Effective'])
        ax.yaxis.set_ticklabels(['Non-effective','Effective'])
        plt.show()

        # Calculate mean and std of specificity and sensitivity
        mean_sensitivity = np.mean(sensitivity_list)
        std_sensitivity = np.std(sensitivity_list)
        mean_specificity = np.mean(specificity_list)
        std_specificity = np.std(specificity_list)

        scores = {
            'mean_auc': mean_auc,
            'std_auc': std_auc,
            'mean_sensitivity': mean_sensitivity,
            'std_sensitivity': std_sensitivity,
            'mean_specificity': mean_specificity,
            'std_specificity': std_specificity,
        }
        return scores

def SVM_test(mdl, X_test, y_test):
    # Plot confusion matrix
    pred_y      = mdl.predict(X_test)
    pred_prob_y = mdl.predict_proba(X_test)[:,1]

    plot_conf_mat(y_test, pred_y)
    plot_roc(y_test, pred_prob_y)
    print("Test score : " + str(mdl.score(X_test, y_test)))
    return

# Save & load the model parameter
def Save_mdl(mdl, path='./data'):
    # save
    with open(path+'/svc_model.pkl','wb') as f:
        pickle.dump(mdl,f)
    return

def Load_mdl(path='./data/', filename = 'svc_model.pkl'):
    # load
    with open(path+filename, 'rb') as f:
        mdl = pickle.load(f)
    return mdl