#!/usr/bin/env python3

import itertools
import argparse
import re

parser = argparse.ArgumentParser()
parser.add_argument('log_file')
parser.add_argument('--dev-prefix')
parser.add_argument('--score', default='bleu', choices=('ter', 'bleu', 'wer'))
parser.add_argument('--task-name')

if __name__ == '__main__':
    args = parser.parse_args()

    with open(args.log_file) as log_file:
        scores = {}
        current_step = 0
        max_step = 0

        for line in log_file:
            m = re.search('step (\d+)', line)
            if m:
                current_step = int(m.group(1))
                max_step = max(max_step, current_step)
                continue

            if args.task_name is not None:
                if not re.search(args.task_name, line):
                    continue
            if args.dev_prefix is not None:
                if not re.search(args.task_name, line):
                    continue

            m = re.findall('(bleu|score|ter|wer|penalty|ratio)=(\d+.\d+)', line)
            if m:
                scores_ = {k: float(v) for k, v in m}
                scores.setdefault(current_step, scores_)

        def key(d):
            score = d.get(args.score.lower())
            if score is None:
                score = d.get('score')

            if args.score in ('ter', 'wer'):
                score = -score
            return score

        step, best = max(scores.items(), key=lambda p: key(p[1]))

        if 'score' in best:
            missing_key = next(k for k in ['ter', 'bleu', 'wer'] if k not in best)
            best[missing_key] = best.pop('score')

        keys = [args.score, 'ter', 'bleu', 'wer', 'penalty', 'ratio']
        best = sorted(best.items(), key=lambda p: keys.index(p[0]))

        print(' '.join(itertools.starmap('{}={:.2f}'.format, best)) + ' step={}/{}'.format(step, max_step))
