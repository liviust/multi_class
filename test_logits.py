import json
from voc_eval import get_average_precision
import numpy as np


def main():
    file_path = 'result.json'
    with open(file_path, 'r') as json_file:
        data = json.load(json_file)
    pre = np.array(data['prediction'])
    gt = np.array(data['ground_true'])
    mAP = 0
    for i in xrange(20):
        p = get_average_precision(pre[:, i], gt[:, i])
        print ('mAP: %.3f' %p)
        mAP += p
    print ('mAP: %.3f' % (mAP/20))


if __name__ == '__main__':
    main()