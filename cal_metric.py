import numpy as np
import pandas as pd
from scipy.stats import pearsonr
from sklearn.metrics import mean_squared_error
from scipy.spatial.distance import jensenshannon
from math import sqrt
import argparse
import os
parser = argparse.ArgumentParser(description='metrics')
parser.add_argument("--document", type=str, default='dataset6_MHPR')
args = parser.parse_args()


Data =args.document #
outdir = 'result/' + Data + '/'
if not os.path.exists(outdir):
    os.mkdir(outdir)

def read_csv(file_path):
    return pd.read_csv(file_path)

def normalize_data(df):
    return (df - df.min()) / (df.max() - df.min())

def ssim(im1, im2):
    assert len(im1.shape) == 2 and len(im2.shape) == 2, "Both matrices must be 2-dimensional."
    assert im1.shape == im2.shape, "The shapes of the two matrices must match."

    # 确定M值
    M = max(im1.max(), im2.max())

    mu1 = im1.mean()
    mu2 = im2.mean()
    sigma1 = np.sqrt(((im1 - mu1) ** 2).mean())
    sigma2 = np.sqrt(((im2 - mu2) ** 2).mean())
    sigma12 = ((im1 - mu1) * (im2 - mu2)).mean()

    k1, k2, L = 0.01, 0.03, M
    C1 = (k1 * L) ** 2
    C2 = (k2 * L) ** 2
    C3 = C2 / 2

    l12 = (2 * mu1 * mu2 + C1) / (mu1 ** 2 + mu2 ** 2 + C1)
    c12 = (2 * sigma1 * sigma2 + C2) / (sigma1 ** 2 + sigma2 ** 2 + C2)
    s12 = (sigma12 + C3) / (sigma1 * sigma2 + C3)

    ssim = l12 * c12 * s12

    return ssim

def calculate_ssim_for_genes(true_labels_df, predicted_labels_df):
    ssim_results = {}
    for gene in true_labels_df.columns:
        true_label = true_labels_df[gene].values.reshape(-1, 1)  # 将其转换为二维数组
        predicted_label = predicted_labels_df[gene].values.reshape(-1, 1)  # 同上
        ssim_score = ssim(true_label, predicted_label)  # 使用修改后的二维数组调用SSIM计算函数
        ssim_results[gene] = ssim_score

    ssim_score = np.mean(list(ssim_results.values()))
    return ssim_score
def calculate_pcc(true_labels, predicted_values):
    pcc_results = [pearsonr(true_labels[gene], predicted_values[gene])[0] for gene in true_labels]
    return np.mean(pcc_results)

def calculate_rmse(true_labels, predicted_values):
    rmse_results = [sqrt(mean_squared_error(true_labels[gene], predicted_values[gene])) for gene in true_labels]
    return np.mean(rmse_results)

def calculate_js(true_labels, predicted_values):
    js_results = [jensenshannon(true_labels[gene], predicted_values[gene]) for gene in true_labels]
    return np.mean(js_results)

def main(true_labels_csv, predicted_values_csv):

    true_labels = read_csv(true_labels_csv)
    predicted_values = read_csv(predicted_values_csv)

    true_labels_normalized = normalize_data(true_labels)
    predicted_values_normalized = normalize_data(predicted_values)

    pcc = calculate_pcc(true_labels_normalized, predicted_values_normalized)
    rmse = calculate_rmse(true_labels, predicted_values)
    js = calculate_js(true_labels_normalized, predicted_values_normalized)
    ssim = calculate_ssim_for_genes(true_labels_normalized, predicted_values_normalized).mean()

    results_df = pd.DataFrame({
        "PCC": [pcc],
        "SSIM": [ssim],
        "RMSE": [rmse],
        "JS": [js]
    }, index=["Value"])

    return results_df

true_labels_csv = outdir + 'original.csv'
predicted_values_csv = outdir + 'prediction.csv'

results_df = main(true_labels_csv, predicted_values_csv)
results_df.to_csv(outdir + '/' + Data + '_final_result.csv', header=1, index=1)
