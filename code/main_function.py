import numpy as np 
from utils.process_para import optPara
from travaltes import trainTestMain
from utils.get_data import getTraValTesData
import pandas as pd
import os
import datetime

def cross5CV(opt):
    opt.cross_indent = 'cross'
    nfold = opt.nfold
    dataName = opt.dataName
    crossKey = opt.crossKey
    seed_cross = opt.seed_cross
    seed_indent = opt.seed_indent
    exp_name = opt.exp_name
    dataPath = './Datasets/' + dataName + '/'
    opt.usedDataPath = usedDataPath = './Datasets/' + dataName + '/used_data/'
    opt.splitPath = splitPath = '_'.join([dataPath + 'splitData_TraValTes', str(nfold) + 'nfold', 'seedIndent' + str(seed_indent), 'seedCross' + str(seed_cross) + '/'])
    lossType = opt.lossType
    result_key = opt.result_key
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    opt.resultTxt = resultTxt = dataName + '/' + '_'.join([exp_name, dataName, 'resultTxt', lossType, crossKey, result_key, timestamp]) + '.csv'

    result = np.zeros((nfold + 2, 8))
    fold_results = []
    for kfold in range(nfold):
        opt.kfold = kfold
        tra_name = splitPath + crossKey + '_tra_kfold' + str(kfold) + '_seed' + str(opt.seed_indent) + '.txt'
        tes_name = splitPath + crossKey + '_tes_kfold' + str(kfold) + '_seed' + str(opt.seed_indent) + '.txt'
        val_name = tes_name
        sim_A, sim_b, tra_array, val_array, tes_array = getTraValTesData(dataName, usedDataPath, tra_name, val_name, tes_name)
        test_label, test_score, criteria_result, model_max, F_u, F_i = trainTestMain(opt, sim_A, sim_b, tra_array, val_array, tes_array)
        result[kfold, 0] = kfold
        result[kfold, 1:] = np.array(criteria_result)
        fold_results.append(f"Fold {kfold + 1}: AUC={criteria_result[0]:.4f}, AUPR={criteria_result[1]:.4f}, F1={criteria_result[2]:.4f}, ACC={criteria_result[3]:.4f}, REC={criteria_result[4]:.4f}, PRE={criteria_result[5]:.4f}")
    result[-2] = np.mean(result[:-2], axis=0)
    result[-1] = np.std(result[:-2], axis=0)
    columns = ['Fold', 'AUPR', 'AUC', 'F1', 'Accuracy', 'Recall', 'Specificity', 'Precision']
    df = pd.DataFrame(result, columns=columns[:result.shape[1]])
    df.to_csv(resultTxt, index=False, float_format='%.4f')
    return test_label, test_score, criteria_result, model_max, F_u, F_i


def indentTraTes(opt):
    opt.cross_indent = 'indent'
    nfold = opt.nfold
    dataName = opt.dataName
    indentKey = opt.indentKey
    seed_cross = opt.seed_cross
    seed_indent = opt.seed_indent
    exp_name = opt.exp_name
    dataPath = './Datasets/' + dataName + '/'
    opt.usedDataPath = usedDataPath = dataPath + 'used_data/'
    opt.splitPath = splitPath = '_'.join(
        [dataPath + 'splitData_TraValTes', str(nfold) + 'nfold', 'seedIndent' + str(seed_indent),
         'seedCross' + str(seed_cross) + '/'])
    lossType = opt.lossType
    result_key = opt.result_key
    epoch = opt.epochs
    layers = opt.num_layer
    alp = opt.alp

    # Build the output filename from key parameters.
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    file_name = './output/epoch_' + str(epoch) + '_layer_' + str(layers) + '_alp_' + str(alp) + '_' + timestamp + '.txt'
    opt.resultTxt = resultTxt = file_name
    opt.kfold = kfold = -1

    tra_name = splitPath + indentKey + '_tra_kfold' + str(kfold) + '_seed' + str(opt.seed_indent) + '.txt'
    tes_name = splitPath + indentKey + '_tes_kfold' + str(kfold) + '_seed' + str(opt.seed_indent) + '.txt'
    val_name = tes_name

    sim_A, sim_b, tra_array, val_array, tes_array = getTraValTesData(dataName, usedDataPath, tra_name, val_name,
                                                                     tes_name)
    test_label, test_score, criteria_result, model_max, F_u, F_i = trainTestMain(opt, sim_A, sim_b, tra_array,
                                                                                 val_array, tes_array)

    # Save the results as one line of text.
    # Ensure the output directory exists.
    output_dir = os.path.dirname(resultTxt)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    with open(resultTxt, 'w') as f:
        f.write(f"AUPR={criteria_result[0]:.4f}, AUC={criteria_result[1]:.4f}, "
                f"F1={criteria_result[2]:.4f}, ACC={criteria_result[3]:.4f}, "
                f"REC={criteria_result[4]:.4f}, PRE={criteria_result[6]:.4f}\n")
    print(f"Independent test results saved to {resultTxt}")

    # Save prediction scores for later ROC and PR curves.
    score_file = './output/indent_' + dataName + '_' + timestamp + '_scores.csv'
    score_data = np.vstack((np.array(test_label), np.array(test_score))).T
    np.savetxt(score_file, score_data, fmt='%.6f', delimiter=',',
               header='label,score', comments='')
    print(f"Prediction scores saved to {score_file}")

    return test_label, test_score, criteria_result, model_max, F_u, F_i

if __name__ == '__main__':
    opt = optPara()    
    cross5CV(opt)
    test_label, test_score, criteria_result, model_max, F_u, F_i = indentTraTes(opt)
