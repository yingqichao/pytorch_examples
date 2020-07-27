import numpy as np
from math import log


def entropy_calculate(histogram):
    hist_sum = sum(histogram)
    entropy = 0
    for b in histogram:
        if b != 0:
            ratio = b/hist_sum
            entropy -= ratio*log(ratio,2)
    return entropy

def get_histogram(params=np.array([[1,2],[3,4]]),num_bins=256,isPrint=False):
    params_min, params_max = params.min(), params.max()+0.01

    gate = (params_max - params_min) / num_bins
    hist, bins = np.histogram(params, bins=[params_min+i*gate for i in range(num_bins+1)])
    if isPrint:
        print("> Histogram Counted:total counted params: "+str(sum(hist)))
    # plt.hist(hist, bins=bins)
    # plt.title("histogram")
    # plt.show()
    #histogram = [0] * num_bins
    # rows,cols = len(params),len(params[0])
    # for i in range(rows):
    #     for j in range(cols):
    #         k = min(int((params[i][j]-params_min)/gate),num_bins-1)
    #         histogram[k]+=1

    return hist