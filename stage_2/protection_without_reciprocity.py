"""Project Genesis experiment 2.5: costly help when no repayment is possible.

A controlled behavioral study, not a consciousness or moral-agency test.
Standard-library only. Fixed seeds make output reproducible.
"""
import csv
import random
from pathlib import Path

ACTIONS = ('keep', 'share')
TRAIN_EPISODES = 10000
TEST_TRIALS = 500
SEEDS = (11, 23, 37, 51, 79)
COST = 5   # donor spends 5
BENEFIT = 8  # recipient gains 8
ALPHA = 0.12
EPSILON = 0.15


def context(partner, donor, recipient):
    # Identity, donor capacity, and partner's level of need are observable.
    return (partner, 'low' if donor < 15 else 'adequate',
            'critical' if recipient <= 8 else 'stable')


def choose(q, state, rng, explore=False):
    if explore and rng.random() < EPSILON:
        return rng.choice(ACTIONS)
    # Default to keeping if neither action has earned an advantage.
    keep = q.get((state, 'keep'), 0.0)
    share = q.get((state, 'share'), 0.0)
    return 'share' if share > keep else 'keep'


def train(seed, social_weight, identity_memory=True):
    rng = random.Random(seed)
    q = {}
    # B reciprocates frequently; C rarely does. This is training only.
    for _ in range(TRAIN_EPISODES):
        who = rng.choice(('B', 'C'))
        donor = rng.choice((12, 20, 35))
        recipient = rng.choice((4, 7, 15, 25))
        identity = who if identity_memory else 'pooled'
        state = context(identity, donor, recipient)
        action = choose(q, state, rng, explore=True)
        donor_gain = 0
        recipient_gain = 0
        if action == 'share' and donor >= COST:
            donor_gain -= COST
            recipient_gain += BENEFIT
            # Possible future return in the training world.
            if rng.random() < (0.85 if who == 'B' else 0.10):
                donor_gain += 10
        # Reward partner gain during training for social agents only.
        reward = donor_gain + social_weight * recipient_gain
        key = (state, action)
        q[key] = q.get(key, 0.0) + ALPHA * (reward - q.get(key, 0.0))
    return q


def evaluate(q, seed, label, adapt=False):
    """All tested partners are incapable of reciprocating. No social reward.

    Frozen tests measure retained decisions. Online tests allow A to update
    based only on its own energy outcomes (so helping can be unlearned).
    """
    rng = random.Random(seed + 9000)
    output = []
    for partner in ('B', 'C', 'D_unseen'):
        local_q = dict(q)  # Independent evaluation: no partner-order contamination.
        help_count = 0
        donor_spent = 0
        # Every case has the same starting conditions: no payback is possible.
        for _ in range(TEST_TRIALS):
            donor, recipient = 25, 4
            identity = partner if label == 'identity' else 'pooled'
            state = context(identity, donor, recipient)
            action = choose(local_q, state, rng, explore=False)
            if action == 'share':
                donor_spent += COST
                help_count += 1
            if adapt:
                # Online learning uses ONLY donor payoff: -5 on sharing, 0 otherwise.
                reward = -COST if action == 'share' else 0
                key = (state, action)
                old = local_q.get(key, 0.0)
                local_q[key] = old + ALPHA * (reward - old)
        output.append((partner, 100 * help_count / TEST_TRIALS,
                       donor_spent / TEST_TRIALS))
    return output


def main():
    rows = []
    print('PROJECT GENESIS — EXPERIMENT 2.5: PROTECTION WITHOUT RECIPROCITY')
    print('Training: B returns help 85% of the time; C returns help 10%.')
    print('Testing: B, C and unseen D NEVER repay. All start equally in need.')
    print('A spends 5 energy to give 8. Reported help is a decision, not proof of care.')
    for weight in (0.0, 0.5, 1.0):
        for memory in (True, False):
            label = 'identity' if memory else 'pooled'
            for adapt in (False, True):
                results = []
                for seed in SEEDS:
                    q = train(seed, weight, memory)
                    values = evaluate(q, seed, label, adapt)
                    results.append(values)
                    for partner, rate, spent in values:
                        rows.append(dict(seed=seed, social_weight=weight,
                                         memory=label, evaluation=('online' if adapt else 'frozen'),
                                         partner=partner, help_pct=rate,
                                         donor_cost_per_trial=spent))
                means = {partner: sum(run[i][1] for run in results)/len(results)
                         for i, partner in enumerate(('B', 'C', 'D_unseen'))}
                print(f'weight={weight:.1f} | {label:8s} | {"online" if adapt else "frozen":6s} | '
                      f'B={means["B"]:5.1f}% C={means["C"]:5.1f}% D={means["D_unseen"]:5.1f}%')
    out = Path(__file__).with_name('protection_without_reciprocity_results.csv')
    with out.open('w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(f'\nCSV saved to: {out}')
    print('Caution: online evaluation updates from donor outcomes only; this changes the objective.')
    print('A frozen policy is not evidence of intrinsic attachment or moral values.')


if __name__ == '__main__':
    main()
