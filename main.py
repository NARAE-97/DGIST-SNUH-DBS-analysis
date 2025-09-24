import os
import Statistics_funcs as sf
import Classifier_funcs_with_label as cf

if __name__=="__main__":
    '''
    Stochastic analysis part.
    We will calculate p-value or  multucollinearity in this part
    '''
    if not os.path.exists("./results"):
        os.makedirs("./results")
    sf.anova_spk("burst_data")
    sf.anova_spk("nonb_data")
    sf.mkfig_spk()
    sf.multcoll()
    sf.paired_updrs()
    sf.anova_updrs()

    # '''
    # Classifier part.
    # We will train SVM model with the data and evaluate the model.
    # '''
    csv_path = './data/Spk_char_whole.csv'
    df = cf.load_and_preprocess(csv_path)
    X1, y1 = cf.prepare_model1_data(df)
    X2, y2 = cf.prepare_model2_data(df)

    scores1 = cf.SVM_train(X1, y1, nested=True)
    scores2 = cf.SVM_train(X2, y2, nested=True)

    print("Model1: ", scores1)
    print("Model2: ", scores2)
    pass