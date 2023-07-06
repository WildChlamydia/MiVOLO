import argparse
import json
import logging
from typing import Tuple

import matplotlib.pyplot as plt
import seaborn as sns
import torch
from eval_tools import Metrics, time_sync, write_results
from mivolo.data.dataset import build as build_data
from mivolo.modeling.mi_volo import MiVOLO
from timm.utils import setup_default_logging

_logger = logging.getLogger("inference")
LOG_FREQUENCY = 10


def get_parser():
    parser = argparse.ArgumentParser(description="PyTorch MiVOLO Validation")
    parser.add_argument("--dataset_images", default="", type=str, required=True, help="path to images")
    parser.add_argument("--dataset_annotations", default="", type=str, required=True, help="path to annotations")
    parser.add_argument(
        "--dataset_name",
        default=None,
        type=str,
        required=True,
        choices=["utk", "imdb", "lagenda", "fairface", "adience"],
        help="dataset name",
    )
    parser.add_argument("--split", default="validation", help="dataset split (default: validation)")
    parser.add_argument("--checkpoint", default="", type=str, required=True, help="path to mivolo checkpoint")

    parser.add_argument("--batch-size", default=64, type=int, help="batch size")
    parser.add_argument(
        "--workers", default=4, type=int, metavar="N", help="number of data loading workers (default: 4)"
    )
    parser.add_argument("--device", default="cuda", type=str, help="Device (accelerator) to use.")
    parser.add_argument("--l-for-cs", type=int, default=5, help="L for CS (cumulative score)")

    parser.add_argument("--half", action="store_true", default=False, help="use half-precision model")
    parser.add_argument(
        "--with-persons", action="store_true", default=False, help="If the model will run with persons, if available"
    )
    parser.add_argument(
        "--disable-faces", action="store_true", default=False, help="If the model will use only persons if available"
    )

    parser.add_argument("--draw-hist", action="store_true", help="Draws the hist of error by age")
    parser.add_argument(
        "--results-file",
        default="",
        type=str,
        metavar="FILENAME",
        help="Output csv file for validation results (summary)",
    )
    parser.add_argument(
        "--results-format", default="csv", type=str, help="Format for results file one of (csv, json) (default: csv)."
    )

    return parser


def process_batch(
    mivolo_model: MiVOLO,
    input: torch.tensor,
    target: torch.tensor,
    num_classes_gender: int = 2,
):

    start = time_sync()
    output = mivolo_model.inference(input)
    # target with age == -1 and gender == -1 marks that sample is not valid
    assert not (all(target[:, 0] == -1) and all(target[:, 1] == -1))

    if not mivolo_model.meta.only_age:
        gender_out = output[:, :num_classes_gender]
        gender_target = target[:, 1]
        age_out = output[:, num_classes_gender:]
    else:
        age_out = output
        gender_out, gender_target = None, None

    # measure elapsed time
    process_time = time_sync() - start

    age_target = target[:, 0].unsqueeze(1)

    return age_out, age_target, gender_out, gender_target, process_time


def _filter_invalid_target(out: torch.tensor, target: torch.tensor):
    # exclude samples where target gt == -1, that marks sample is not valid
    mask = target != -1
    return out[mask], target[mask]


def postprocess_gender(gender_out: torch.tensor, gender_target: torch.tensor) -> Tuple[torch.tensor, torch.tensor]:
    if gender_target is None:
        return gender_out, gender_target
    return _filter_invalid_target(gender_out, gender_target)


def postprocess_age(age_out: torch.tensor, age_target: torch.tensor, dataset) -> Tuple[torch.tensor, torch.tensor]:
    # Revert _norm_age() operation. Output is 2 float tensors

    age_out, age_target = _filter_invalid_target(age_out, age_target)

    age_out = age_out * (dataset.max_age - dataset.min_age) + dataset.avg_age
    # clamp to 0 because age can be below zero
    age_out = torch.clamp(age_out, min=0)

    if dataset.age_classes is not None:
        # classification case
        age_out = torch.round(age_out)
        if dataset._intervals.device != age_out.device:
            dataset._intervals = dataset._intervals.to(age_out.device)
        age_inds = torch.searchsorted(dataset._intervals, age_out, side="right") - 1
        age_out = age_inds
    else:
        age_target = age_target * (dataset.max_age - dataset.min_age) + dataset.avg_age
    return age_out, age_target


def validate(args):

    if torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.benchmark = True

    mivolo_model = MiVOLO(
        args.checkpoint,
        args.device,
        half=args.half,
        use_persons=args.with_persons,
        disable_faces=args.disable_faces,
        verbose=True,
    )

    dataset, loader = build_data(
        name=args.dataset_name,
        images_path=args.dataset_images,
        annotations_path=args.dataset_annotations,
        split=args.split,
        mivolo_model=mivolo_model,  # to get meta information from model
        workers=args.workers,
        batch_size=args.batch_size,
    )

    d_stat = Metrics(args.l_for_cs, args.draw_hist, dataset.age_classes)

    # warmup, reduce variability of first batch time, especially for comparing torchscript vs non
    mivolo_model.warmup(args.batch_size)

    preproc_end = time_sync()
    for batch_idx, (input, target) in enumerate(loader):

        preprocess_time = time_sync() - preproc_end
        # get output and calculate loss
        age_out, age_target, gender_out, gender_target, process_time = process_batch(
            mivolo_model, input, target, dataset.num_classes_gender
        )

        gender_out, gender_target = postprocess_gender(gender_out, gender_target)
        age_out, age_target = postprocess_age(age_out, age_target, dataset)

        d_stat.update_gender_accuracy(gender_out, gender_target)
        if d_stat.is_regression:
            d_stat.update_regression_age_metrics(age_out, age_target)
        else:
            d_stat.update_age_accuracy(age_out, age_target)
        d_stat.update_time(process_time, preprocess_time, input.shape[0])

        if batch_idx % LOG_FREQUENCY == 0:
            _logger.info(
                "Test: [{0:>4d}/{1}]  " "{2}".format(batch_idx, len(loader), d_stat.get_info_str(input.size(0)))
            )

        preproc_end = time_sync()

    # model info
    results = dict(
        model=args.checkpoint,
        dataset_name=args.dataset_name,
        param_count=round(mivolo_model.param_count / 1e6, 2),
        img_size=mivolo_model.input_size,
        use_faces=mivolo_model.meta.use_face_crops,
        use_persons=mivolo_model.meta.use_persons,
        in_chans=mivolo_model.meta.in_chans,
        batch=args.batch_size,
    )
    # metrics info
    results.update(d_stat.get_result())
    return results


def main():
    parser = get_parser()
    setup_default_logging()
    args = parser.parse_args()

    if torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.benchmark = True

    results = validate(args)

    result_str = " * Age Acc@1 {:.3f} ({:.3f})".format(results["agetop1"], results["agetop1_err"])
    if "gendertop1" in results:
        result_str += " Gender Acc@1 1 {:.3f} ({:.3f})".format(results["gendertop1"], results["gendertop1_err"])
    result_str += " Mean inference time {:.3f} ms Mean preprocessing time {:.3f}".format(
        results["mean_inference_time"], results["mean_preprocessing_time"]
    )
    _logger.info(result_str)

    if args.draw_hist and "per_age_error" in results:
        err = [sum(v) / len(v) for k, v in results["per_age_error"].items()]
        ages = list(results["per_age_error"].keys())
        sns.scatterplot(x=ages, y=err, hue=err)
        plt.legend([], [], frameon=False)
        plt.xlabel("Age")
        plt.ylabel("MAE")
        plt.savefig("age_error.png", dpi=300)

    if args.results_file:
        write_results(args.results_file, results, format=args.results_format)

    # output results in JSON to stdout w/ delimiter for runner script
    print(f"--result\n{json.dumps(results, indent=4)}")


if __name__ == "__main__":
    main()
