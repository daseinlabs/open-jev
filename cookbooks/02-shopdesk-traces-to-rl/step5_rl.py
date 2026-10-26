"""Step 5: RL. Start from the SFT model and learn from the desk's reward.

`improve` plays conversations; for each one that failed, it replays it up to every
decision, tries the other candidates, and lets the model finish. Choices that turn a
failure into a success become new training labels. The SFT training features act as an
anchor (on by default), so behaviour the traces already taught well is kept. Only the head learns."""
from openjev.tod.envs.shopdesk import ShopDesk
from openjev.tod.rl import HeadPolicy, improve

from adapter import SPEC

policy = HeadPolicy('runs/shopdesk/sft', SPEC)
improve(policy, ShopDesk, 'runs/shopdesk/rl', max_reward=1.0)  # 3 iterations x 128 conversations
