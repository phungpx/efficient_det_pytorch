from collections import Counter
from typing import List, Tuple, Dict

import numpy as np
from torch import nn
from shapely import geometry


class MeanAveragePrecision(nn.Module):
    def __init__(
        self,
        classes: Dict[str, int],
        iou_threshold: float = 0.5,
        method: str = 'every_point_interpolation'  # or '11_point_interpolation'
    ):
        super(MeanAveragePrecision, self).__init__()
        self.classes = {class_id: class_name for class_name, class_id in classes.items()}
        self.iou_threshold = iou_threshold
        self.method = method

    def forward(self, detections: list, ground_truths: list) -> dict:
        r'''
        Args
            detections: list with all detections ([image_id, class_id, confidence, [x1, y1, x2, y2]])
            ground_truths: list with all ground_truths ([image_id, class_id, 1., [x1, y1, x2, y2]])
        Outputs:
            A list of dictionaries. Each dictionary contains information and metrics of each class.
            The keys of each dictionary are:
            dict['class']: class representing the current dictionary;
            dict['precision']: array with the precision values;
            dict['recall']: array with the recall values;
            dict['AP']: average precision;
            dict['interpolated precision']: interpolated precision values;
            dict['interpolated recall']: interpolated recall values;
            dict['total ground truths']: total number of ground truth positives;
            dict['total detections']: total number of detections;
            dict['total TP']: total number of True Positive detections;
            dict['total FP']: total number of False Negative detections;
        '''
        results = []

        class_indices = sorted(map(int, self.classes.keys()))
        for class_id in class_indices:
            # get only detection of class_id -> c_dets
            c_dets = [det for det in detections if det[1] == class_id]
            num_detections = len(c_dets)

            # get only ground truth of class_id -> c_gts
            c_gts = [gt for gt in ground_truths if gt[1] == class_id]
            num_ground_truths = len(c_gts)

            # initialize TP, FP with all 0 values
            TP = [0] * num_detections
            FP = [0] * num_detections

            # create dictionary with amount of gts for each image
            # Ex: amount_c_gts = {0: [0,0,0], 1: [0,0,0,0,0]}
            amount_c_gts = Counter([gt[0] for gt in c_gts])
            amount_c_gts = {
                image_id: [0] * num_gts for image_id, num_gts in amount_c_gts.items()
            }

            # sort c_dets by decreasing confidence
            c_dets = sorted(c_dets, key=lambda det: det[2], reverse=True)

            for i, det in enumerate(c_dets):
                # collect grounth truths which have same image_id with det-> gts
                gts = [gt for gt in c_gts if gt[0] == det[0]]

                j_max, iou_max = 0, 0.
                for j, gt in enumerate(gts):
                    iou = self._iou(boxA=det[3], boxB=gt[3])
                    if iou > iou_max:
                        iou_max = iou
                        j_max = j

                # assign c_det (detection) as true positive/don't care/false positive
                if iou_max >= self.iou_threshold:
                    if amount_c_gts[det[0]][j_max] == 0:
                        TP[i] = 1  # count as true positive
                        amount_c_gts[det[0]][j_max] == 1  # flag as already 'seen'
                    else:
                        FP[i] = 1  # count as false positive
                else:
                    FP[i] = 1  # count as false positive

            # compute precision, recall and average precision
            acc_TP = np.cumsum(TP)
            acc_FP = np.cumsum(FP)

            if num_ground_truths > 0:
                prec = np.divide(acc_TP, (acc_FP + acc_TP)).tolist()
                rec = (acc_TP / num_ground_truths).tolist()
            else:
                prec, rec = [0] * num_detections, [0] * num_detections

            if self.method == 'every_point_interpolation':
                ap, mrec, mprec = self.every_points_interpolated_AP(rec=rec, prec=prec)
            elif self.method == 'elevent_point_interpolation':
                ap, mrec, mprec = self.eleven_points_interpolated_AP(rec=rec, prec=prec)
            else:
                raise RuntimeError('Interpolation Method is Wrong.')

            result = {
                'AP': ap,
                'class': self.classes[class_id],
                'recall': rec,
                'precision': prec,
                'interpolated recall': mrec,
                'interpolated precision': mprec,
                'total detections': num_detections,
                'total ground truths': num_ground_truths,
                'total TP': sum(TP),
                'total FP': sum(FP)
            }

            results.append(result)

        APs = [result['AP'] for result in results]
        print(f'MeanAveragePrecision: {sum(APs) / len(APs) if len(APs) else 0.}')

        return results

    def every_points_interpolated_AP(
        self,
        rec: List[float],
        prec: List[float]
    ) -> Tuple[float, List[float], List[float]]:

        mrec = [0.] + rec + [1.]
        mprec = [0.] + prec + [0.]

        # range(start, end=0, step=-1)
        for i in range(len(mprec) - 1, 0, -1):
            mprec[i - 1] = max(mprec[i - 1], mprec[i])

        ap = 0.
        # range(start, end, step=1)
        for i in range(1, len(mrec)):
            if mrec[i] != mrec[i - 1]:
                ap += (mrec[i] - mrec[i - 1]) * mprec[i]

        return ap, mrec[0: len(mrec) - 1], mprec[0: len(mprec) - 1]

    # 11-point interpolated average precision
    def eleven_points_interpolated_AP(
        self,
        rec: List[float],
        prec: List[float]
    ) -> Tuple[float, List[float], List[float]]:

        rho_interp = []
        recall_valid = []

        recall_values = np.linspace(start=0, stop=1, num=11).tolist()
        # for each recall_values (0, 0.1, 0.2, ... , 1)
        for r in recall_values[::-1]:
            # obtain all recall values higher or equal than r
            arg_greater_recalls = np.argwhere(np.array(rec) >= r)

            prec_max = 0.
            # ff there are recalls above r
            if arg_greater_recalls.size != 0:
                prec_max = max(prec[arg_greater_recalls.min():])

            recall_valid.append(r)
            rho_interp.append(prec_max)

        # by definition AP = sum(max(precision whose recall is above r)) / 11
        ap = sum(rho_interp) / 11

        # generating values for the plot
        rvals = [recall_valid[0]] + recall_valid + [0.]
        pvals = [0.] + rho_interp + [0.]

        cc = []
        for i in range(len(rvals)):
            p = (rvals[i], pvals[i - 1])
            if p not in cc:
                cc.append(p)
            p = (rvals[i], pvals[i])
            if p not in cc:
                cc.append(p)

        recall_values = [i[0] for i in cc]
        rho_interp = [i[1] for i in cc]

        return ap, recall_values, rho_interp

    def _iou(self, boxA: List[float], boxB: List[float]) -> float:
        r'''Calculates intersection over union
        Args:
            boxA: List[box[x_min, y_min, x_max, y_max]]
            boxB: List[box[x_min, y_min, x_max, y_max]]
        Return:
            iou: float, intersection over union of boxA, boxB
        '''
        iou = 0.
        boxA = geometry.box(*[boxA[0], boxA[1], boxA[2] + 1, boxA[3] + 1])
        boxB = geometry.box(*[boxB[0], boxB[1], boxB[2] + 1, boxB[3] + 1])
        if boxA.intersects(boxB):
            iou = boxA.intersection(boxB).area / boxA.union(boxB).area

        return iou
