from .builder import DATASETS
from .coco import CocoDataset


from collections import OrderedDict
import itertools
from mmcv.utils import logging
from mmcv.utils.logging import print_log
import numpy as np
import pandas as pd
from pycocotools.cocoeval import COCOeval
from terminaltables.ascii_table import AsciiTable

@DATASETS.register_module()
class WheatDataset(CocoDataset):

    CLASSES = ('wheat_head', )

    def evaluate(self,
                 results,
                 metric='bbox',
                 logger=None,
                 jsonfile_prefix=None,
                 classwise=False,
                 proposal_nums=(100, 300, 1000),
                 iou_thrs=None,
                 metric_items=None):
        """Evaluation in COCO protocol.

        Args:
            results (list[list | tuple]): Testing results of the dataset.
            metric (str | list[str]): Metrics to be evaluated. Options are
                'bbox', 'segm', 'proposal', 'proposal_fast'.
            logger (logging.Logger | str | None): Logger used for printing
                related information during evaluation. Default: None.
            jsonfile_prefix (str | None): The prefix of json files. It includes
                the file path and the prefix of filename, e.g., "a/b/prefix".
                If not specified, a temp file will be created. Default: None.
            classwise (bool): Whether to evaluating the AP for each class.
            proposal_nums (Sequence[int]): Proposal number used for evaluating
                recalls, such as recall@100, recall@1000.
                Default: (100, 300, 1000).
            iou_thrs (Sequence[float], optional): IoU threshold used for
                evaluating recalls/mAPs. If set to a list, the average of all
                IoUs will also be computed. If not specified, [0.50, 0.55,
                0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90, 0.95] will be used.
                Default: None.
            metric_items (list[str] | str, optional): Metric items that will
                be returned. If not specified, ``['AR@100', 'AR@300',
                'AR@1000', 'AR_s@1000', 'AR_m@1000', 'AR_l@1000' ]`` will be
                used when ``metric=='proposal'``, ``['mAP', 'mAP_50', 'mAP_75',
                'mAP_s', 'mAP_m', 'mAP_l']`` will be used when
                ``metric=='bbox' or metric=='segm'``.

        Returns:
            dict[str, float]: COCO style evaluation metric.
        """

        metrics = metric if isinstance(metric, list) else [metric]
        allowed_metrics = ['bbox', 'segm', 'proposal', 'proposal_fast']
        for metric in metrics:
            if metric not in allowed_metrics:
                raise KeyError(f'metric {metric} is not supported')
        if iou_thrs is None:
            iou_thrs = np.linspace(
                .5, 0.95, int(np.round((0.95 - .5) / .05)) + 1, endpoint=True)
        if metric_items is not None:
            if not isinstance(metric_items, list):
                metric_items = [metric_items]

        result_files, tmp_dir = self.format_results(results, jsonfile_prefix)

        eval_results = OrderedDict()
        cocoGt = self.coco
        for metric in metrics:
            msg = f'Evaluating {metric}...'
            if logger is None:
                msg = '\n' + msg
            print_log(msg, logger=logger)

            if metric == 'proposal_fast':
                ar = self.fast_eval_recall(
                    results, proposal_nums, iou_thrs, logger='silent')
                log_msg = []
                for i, num in enumerate(proposal_nums):
                    eval_results[f'AR@{num}'] = ar[i]
                    log_msg.append(f'\nAR@{num}\t{ar[i]:.4f}')
                log_msg = ''.join(log_msg)
                print_log(log_msg, logger=logger)
                continue

            if metric not in result_files:
                raise KeyError(f'{metric} is not in results')
            try:
                cocoDt = cocoGt.loadRes(result_files[metric])
            except IndexError:
                print_log(
                    'The testing results of the whole dataset is empty.',
                    logger=logger,
                    level=logging.ERROR)
                break

            iou_type = 'bbox' if metric == 'proposal' else metric
            cocoEval = COCOeval(cocoGt, cocoDt, iou_type)
            cocoEval.params.catIds = self.cat_ids
            cocoEval.params.imgIds = self.img_ids
            cocoEval.params.maxDets = list(proposal_nums)
            cocoEval.params.iouThrs = iou_thrs
            # mapping of cocoEval.stats
            coco_metric_names = {
                'mAP': 0,
                'mAP_50': 1,
                'mAP_75': 2,
                'mAP_s': 3,
                'mAP_m': 4,
                'mAP_l': 5,
                'AR@100': 6,
                'AR@300': 7,
                'AR@1000': 8,
                'AR_s@1000': 9,
                'AR_m@1000': 10,
                'AR_l@1000': 11
            }
            if metric_items is not None:
                for metric_item in metric_items:
                    if metric_item not in coco_metric_names:
                        raise KeyError(
                            f'metric item {metric_item} is not supported')

            if metric == 'proposal':
                cocoEval.params.useCats = 0
                cocoEval.evaluate()
                cocoEval.accumulate()
                cocoEval.summarize()
                if metric_items is None:
                    metric_items = [
                        'AR@100', 'AR@300', 'AR@1000', 'AR_s@1000',
                        'AR_m@1000', 'AR_l@1000'
                    ]

                for item in metric_items:
                    val = float(
                        f'{cocoEval.stats[coco_metric_names[item]]:.3f}')
                    eval_results[item] = val
            else:
                cocoEval.evaluate()
                cocoEval.accumulate()
                cocoEval.summarize()
                if classwise:  # Compute per-category AP
                    # Compute per-category AP
                    # from https://github.com/facebookresearch/detectron2/
                    precisions = cocoEval.eval['precision']
                    # precision: (iou, recall, cls, area range, max dets)
                    assert len(self.cat_ids) == precisions.shape[2]

                    results_per_category = []
                    for idx, catId in enumerate(self.cat_ids):
                        # area range index 0: all area ranges
                        # max dets index -1: typically 100 per image
                        nm = self.coco.loadCats(catId)[0]
                        precision = precisions[:, :, idx, 0, -1]
                        precision = precision[precision > -1]
                        if precision.size:
                            ap = np.mean(precision)
                        else:
                            ap = float('nan')
                        results_per_category.append(
                            (f'{nm["name"]}', f'{float(ap):0.3f}'))

                    num_columns = min(6, len(results_per_category) * 2)
                    results_flatten = list(
                        itertools.chain(*results_per_category))
                    headers = ['category', 'AP'] * (num_columns // 2)
                    results_2d = itertools.zip_longest(*[
                        results_flatten[i::num_columns]
                        for i in range(num_columns)
                    ])
                    table_data = [headers]
                    table_data += [result for result in results_2d]
                    table = AsciiTable(table_data)
                    print_log('\n' + table.table, logger=logger)

                if metric_items is None:
                    metric_items = [
                        'mAP', 'mAP_50', 'mAP_75', 'mAP_s', 'mAP_m', 'mAP_l'
                    ]

                for metric_item in metric_items:
                    key = f'{metric}_{metric_item}'
                    val = float(
                        f'{cocoEval.stats[coco_metric_names[metric_item]]:.3f}'
                    )
                    eval_results[key] = val
                ap = cocoEval.stats[:6]
                eval_results[f'{metric}_mAP_copypaste'] = (
                    f'{ap[0]:.3f} {ap[1]:.3f} {ap[2]:.3f} {ap[3]:.3f} '
                    f'{ap[4]:.3f} {ap[5]:.3f}')

                # compute Average Domain accuracy
                print("Computing ADA score...")
                df = pd.DataFrame(columns=["image_name", "domain", "gtBoxes", "predBoxes", "acc"])
                for idx, img_id in enumerate(cocoGt.getImgIds()):
                    img_info = cocoGt.loadImgs([img_id])[0]

                    # get groundtruth annotations
                    gt_ann_ids = cocoGt.getAnnIds([img_id])
                    gt_anns = cocoGt.loadAnns(gt_ann_ids)
                    targets = [ann["bbox"] for ann in gt_anns] or [[0, 0, 0, 0]]

                    # get prediction annotations
                    dt_ann_ids = cocoDt.getAnnIds([img_id])
                    dt_anns = cocoDt.loadAnns(dt_ann_ids)
                    predictions = [ann["bbox"] for ann in dt_anns] or [[0, 0, 0, 0]]

                    # compute accuracy
                    dt_arr = self.xywh2xyxy(np.array(predictions))
                    gt_arr = self.xywh2xyxy(np.array(targets))
                    acc = self._accuracy(dt_arr, gt_arr)

                    # add row to dataframe
                    df.loc[idx] = [img_info["filename"], img_info["domain"],
                                    targets, predictions, acc]

                # compute ADA score
                ada_score = df[["acc", "domain"]].groupby("domain").mean().mean().values[0] 
                print(f"Average Domain Accuracy (ADA) @[ 1oU=0.5 ] = {ada_score:0.3f}")
                eval_results["ADA"] = float(f'{ada_score:0.3f}')

        if tmp_dir is not None:
            tmp_dir.cleanup()
        return eval_results
    
    @staticmethod
    def xywh2xyxy(bboxes):
        y = bboxes.copy()
        y[:, 2] = bboxes[:, 2] + bboxes[:, 0]  # xmax
        y[:, 3] = bboxes[:, 3] + bboxes[:, 1]  # ymax
        return y

    @staticmethod
    def _accuracy(dts: np.array, gts: np.array, iou_thr: int = 0.5) -> float:
        """
        Compute accuracy between two lists
        Expected format is (x_min,y_min,x_max,y_max)

        """
        if len(dts) > 0 and len(gts) > 0:
            pick = WheatDataset._get_matches(dts, gts, overlapThresh=iou_thr)
            tp = len(pick)
            fn = len(gts) - len(pick)
            fp = len(dts) - len(pick)
            acc = float(tp) / (float(tp) + float(fn) + float(fp))

        elif len(dts) == 0 and len(gts) > 0:
            acc = 0.
        elif len(dts) > 0 and len(gts) == 0:
            acc = 0.
        elif len(dts) == 0 and len(gts) == 0:
            acc = 1.

        return acc

    @staticmethod
    def _get_matches(gts, dts, overlapThresh=0.5):

        gts = np.array([np.array(bbox) for bbox in gts])
        boxes = np.array([np.array(bbox) for bbox in dts])

        # initialize the list of picked indexes
        pick = []

        # grab the coordinates of the bounding boxes
        x1 = boxes[:, 0]
        y1 = boxes[:, 1]
        x2 = boxes[:, 2]
        y2 = boxes[:, 3]

        # compute the area of the bounding boxes and sort the bounding
        # boxes by the bottom-right y-coordinate of the bounding box
        area = (x2 - x1 + 1) * (y2 - y1 + 1)

        # keep looping while some indexes still remain in the indexes
        # list
        area_gt = (gts[:, 2] - gts[:, 0]) * (gts[:, 3] - gts[:, 1])
        gts = gts[np.argsort(area_gt)]
        idxs = list(range(len(area)))
        for (x, y, xx, yy) in gts:
            # grab the last index in the indexes list and add the
            # index value to the list of picked indexes
            area_ = (xx - x) * (yy - y)

            # find the largest (x, y) coordinates for the start of
            # the bounding box and the smallest (x, y) coordinates
            # for the end of the bounding box
            xx1 = np.maximum(x, x1[idxs])
            yy1 = np.maximum(y, y1[idxs])
            xx2 = np.minimum(xx, x2[idxs])
            yy2 = np.minimum(yy, y2[idxs])

            # compute the width and height of the bounding box
            ww = np.maximum(0, xx2 - xx1 + 1)
            hh = np.maximum(0, yy2 - yy1 + 1)

            # compute intersection over union (union is area 1 +area 2-intersection)
            overlap = (ww * hh) / (area[idxs] + area_ - (ww*hh))


            #true_matches = np.where(overlap > overlapThresh)
            if len(overlap) > 0:
                potential_match = np.argmax(overlap) # we select the best match

                if overlap[potential_match] > overlapThresh: # we check if it scores above the threshold
                    pick.append(idxs[potential_match])
                    # delete all indexes from the index list that have
                    idxs = np.delete(idxs, [potential_match])

        # return only the bounding boxes that were picked using the
        # integer data type
        return pick
