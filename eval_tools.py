import csv
import json
import time
from collections import OrderedDict, defaultdict

import torch
from mivolo.data.misc import cumulative_error, cumulative_score
from timm.utils import AverageMeter, accuracy


def time_sync():
    # pytorch-accurate time
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    return time.time()


def write_results(results_file, results, format="csv"):
    with open(results_file, mode="w") as cf:
        if format == "json":
            json.dump(results, cf, indent=4)
        else:
            if not isinstance(results, (list, tuple)):
                results = [results]
            if not results:
                return
            dw = csv.DictWriter(cf, fieldnames=results[0].keys())
            dw.writeheader()
            for r in results:
                dw.writerow(r)
            cf.flush()


class Metrics:
    def __init__(self, l_for_cs, draw_hist, age_classes=None):
        self.batch_time = AverageMeter()
        self.preproc_batch_time = AverageMeter()
        self.seen = 0

        self.losses = AverageMeter()
        self.top1_m_gender = AverageMeter()
        self.top1_m_age = AverageMeter()

        if age_classes is None:
            self.is_regression = True
            self.av_csl_age = AverageMeter()
            self.max_error = AverageMeter()
            self.per_age_error = defaultdict(list)
            self.l_for_cs = l_for_cs
        else:
            self.is_regression = False

        self.draw_hist = draw_hist

    def update_regression_age_metrics(self, age_out, age_target):
        batch_size = age_out.size(0)

        age_abs_err = torch.abs(age_out - age_target)
        age_acc1 = torch.sum(age_abs_err) / age_out.shape[0]
        age_csl = cumulative_score(age_out, age_target, self.l_for_cs)
        me = cumulative_error(age_out, age_target, 20)

        self.top1_m_age.update(age_acc1.item(), batch_size)
        self.av_csl_age.update(age_csl.item(), batch_size)
        self.max_error.update(me.item(), batch_size)

        if self.draw_hist:
            for i in range(age_out.shape[0]):
                self.per_age_error[int(age_target[i].item())].append(age_abs_err[i].item())

    def update_age_accuracy(self, age_out, age_target):
        batch_size = age_out.size(0)
        if batch_size == 0:
            return
        correct = torch.sum(age_out == age_target)
        age_acc1 = correct * 100.0 / batch_size
        self.top1_m_age.update(age_acc1.item(), batch_size)

    def update_gender_accuracy(self, gender_out, gender_target):
        if gender_out is None or gender_out.size(0) == 0:
            return
        batch_size = gender_out.size(0)
        gender_acc1 = accuracy(gender_out, gender_target, topk=(1,))[0]
        if gender_acc1 is not None:
            self.top1_m_gender.update(gender_acc1.item(), batch_size)

    def update_loss(self, loss, batch_size):
        self.losses.update(loss.item(), batch_size)

    def update_time(self, process_time, preprocess_time, batch_size):
        self.seen += batch_size
        self.batch_time.update(process_time)
        self.preproc_batch_time.update(preprocess_time)

    def get_info_str(self, batch_size):
        avg_time = (self.preproc_batch_time.sum + self.batch_time.sum) / self.batch_time.count
        cur_time = self.batch_time.val + self.preproc_batch_time.val
        middle_info = (
            "Time: {cur_time:.3f}s ({avg_time:.3f}s, {rate_avg:>7.2f}/s)  "
            "Loss: {loss.val:>7.4f} ({loss.avg:>6.4f})  "
            "Gender Acc: {top1gender.val:>7.2f} ({top1gender.avg:>7.2f}) ".format(
                cur_time=cur_time,
                avg_time=avg_time,
                rate_avg=batch_size / avg_time,
                loss=self.losses,
                top1gender=self.top1_m_gender,
            )
        )

        if self.is_regression:
            age_info = (
                "Age CS@{l_for_cs}: {csl.val:>7.4f} ({csl.avg:>7.4f})  "
                "Age CE@20: {max_error.val:>7.4f} ({max_error.avg:>7.4f})  "
                "Age ME: {top1age.val:>7.2f} ({top1age.avg:>7.2f})".format(
                    top1age=self.top1_m_age, csl=self.av_csl_age, max_error=self.max_error, l_for_cs=self.l_for_cs
                )
            )
        else:
            age_info = "Age Acc: {top1age.val:>7.2f} ({top1age.avg:>7.2f})".format(top1age=self.top1_m_age)

        return middle_info + age_info

    def get_result(self):
        age_top1a = self.top1_m_age.avg
        gender_top1 = self.top1_m_gender.avg if self.top1_m_gender.count > 0 else None

        mean_per_image_time = self.batch_time.sum / self.seen
        mean_preprocessing_time = self.preproc_batch_time.sum / self.seen

        results = OrderedDict(
            mean_inference_time=mean_per_image_time * 1e3,
            mean_preprocessing_time=mean_preprocessing_time * 1e3,
            agetop1=round(age_top1a, 4),
            agetop1_err=round(100 - age_top1a, 4),
        )

        if self.is_regression:
            results.update(
                dict(
                    max_error=self.max_error.avg,
                    csl=self.av_csl_age.avg,
                    per_age_error=self.per_age_error,
                )
            )

        if gender_top1 is not None:
            results.update(dict(gendertop1=round(gender_top1, 4), gendertop1_err=round(100 - gender_top1, 4)))

        return results
