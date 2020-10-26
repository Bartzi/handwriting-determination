import argparse
import sys

import matplotlib

from tqdm import tqdm

from evaluation.determination_evaluator import DeterminationEvaluator

matplotlib.use('Agg')


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Tool that evaluates handwriting model", formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("log_dir", help='path to log dir of training')
    parser.add_argument("snapshot_prefix", help="prefix of snapshots to evaluate")
    parser.add_argument("eval_gt", help="path to evaluation gt")
    parser.add_argument("--log-name", default="log", help="name of the log file")
    parser.add_argument("-g", "--gpu", default="@numpy", help="GPU to use")
    parser.add_argument("-b", "--batch-size", dest='eval_batch_size', default=1, type=int, help="batch size for evaluation")
    parser.add_argument("-e", "--evaluation-name", default="evaluation_result", help="base name of the resulting eval result file")
    parser.add_argument("--force-reset", action='store_true', default=False, help="reset old eval results and start fresh")
    parser.add_argument("-r", "--render-regions", choices=['tp', 'fp', 'fn', 'tn', 'all', 'none'], help="render regions, choose which to render")
    parser.add_argument("--render-negatives", action='store_true', default=False, help="render negative decisions instead of positive")

    args = parser.parse_args()
    if args.render_regions == 'none':
        args.render_regions = None

    evaluator = DeterminationEvaluator(args)
    snapshots_to_evaluate = evaluator.filter_snapshots(args.snapshot_prefix)

    for snapshot in tqdm(snapshots_to_evaluate):
        evaluator.load_weights(snapshot)
        evaluator.evaluate()

    evaluator.save_eval_result({}, plot_and_print_only=True)
    print("done, cleaning up!")
    sys.exit(1)
