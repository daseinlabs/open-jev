"""Step 4: let a trained model run fresh conversations, with and without deferral.

Below the threshold the model defers and the parent makes the call (an oracle here; your
big model in production). The trade-off to read: success vs. share of decisions handled."""
import json
import random
import sys

from openjev.tod.envs.shopdesk import ShopDesk, teacher
from openjev.tod.rl import HeadPolicy

from adapter import SPEC

artifact = sys.argv[1] if len(sys.argv) > 1 else 'runs/shopdesk/sft'
seeds = range(10_000_000, 10_000_200)  # never used for traces or RL


def teacher_success(seed):
    desk = ShopDesk().reset(seed)
    for tool, order_id in teacher(desk, random.Random(seed + 1)):
        desk.call(tool, order_id)
    return desk.success


policy = HeadPolicy(artifact, SPEC)
report = {'teacher': sum(map(teacher_success, seeds)) / len(seeds)}
for threshold in (None, 0.7, 0.9):
    report[f'model@{threshold}'] = policy.evaluate(ShopDesk, seeds, threshold=threshold)
print(json.dumps(report, indent=2))
