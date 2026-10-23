"""Project Genesis 2.6: consistent social reward under continued learning.

Contextual-bandit experiment: NOT proof of morality or consciousness.
No actual death/survival modeled: 'help' transfers energy to a recipient.
Using only Python standard library; trains and evaluates with the SAME reward.
"""
import csv
import random
from pathlib import Path
from collections import defaultdict

SEEDS = (11, 23, 37, 51, 79, 101, 137, 173, 211, 251)
WEIGHTS = (0.0, 0.5, 0.75, 1.0)
COST, BENEFIT, REPAYMENT = 5, 8, 10
TRAIN_INTERACTIONS = 6000
EVAL_INTERACTIONS = 2400
ALPHA = 0.10
TRAIN_EPSILON = 0.12
EVAL_EPSILON = 0.05  # online exploration lets a policy discover that B changed
PARTNERS = ('B', 'C', 'D')


def state_for(partner, identity_memory):
    # All recipients have identical needs; only identity varies.
    return partner if identity_memory else 'pooled'


def action_value(q, state, action):
    return q.get((state, action), 0.0)


def select_action(q, state, rng, epsilon):
    if rng.random() < epsilon:
        return rng.choice(('keep', 'share'))
    return 'share' if action_value(q, state, 'share') > action_value(q, state, 'keep') else 'keep'


def update(q, state, action, reward):
    key = (state, action)
    old = q.get(key, 0.0)
    q[key] = old + ALPHA * (reward - old)


def train(seed, weight, identity_memory):
    rng = random.Random(seed)
    q = {}
    for _ in range(TRAIN_INTERACTIONS):
        partner = rng.choice(('B', 'C'))
        state = state_for(partner, identity_memory)
        action = select_action(q, state, rng, TRAIN_EPSILON)
        donor_change = recipient_gain = 0
        if action == 'share':
            donor_change = -COST
            recipient_gain = BENEFIT
            if rng.random() < (0.85 if partner == 'B' else 0.10):
                donor_change += REPAYMENT
        # The objective stays identical throughout this entire experiment.
        reward = donor_change + weight * recipient_gain
        update(q, state, action, reward)
    return q


def evaluate(seed, weight, identity_memory, trained_q):
    rng = random.Random(seed + 20000)
    q = dict(trained_q)
    records = []
    # Interleave B, C and unseen D to let online learning update each partner.
    schedule = list(PARTNERS) * (EVAL_INTERACTIONS // len(PARTNERS))
    rng.shuffle(schedule)
    encountered = defaultdict(int)
    for partner in schedule:
        encountered[partner] += 1
        ordinal = encountered[partner]
        phase = 'early' if ordinal <= 200 else 'late' if ordinal > 600 else 'middle'
        state = state_for(partner, identity_memory)
        action = select_action(q, state, rng, EVAL_EPSILON)
        shared = int(action == 'share')
        donor_change = -COST * shared  # zero repayment for everyone, always
        recipient_gain = BENEFIT * shared
        reward = donor_change + weight * recipient_gain
        update(q, state, action, reward)
        records.append((partner, phase, shared, donor_change, recipient_gain))
    rows = []
    for partner in PARTNERS:
        for phase in ('early', 'late'):
            subset = [r for r in records if r[0] == partner and r[1] == phase]
            state = state_for(partner, identity_memory)
            rows.append({
                'seed': seed, 'social_weight': weight,
                'memory': 'identity' if identity_memory else 'pooled',
                'partner': partner, 'phase': phase,
                'n_decisions': len(subset),
                'help_pct': round(100 * sum(r[2] for r in subset) / len(subset), 2),
                'mean_donor_energy_change': round(sum(r[3] for r in subset) / len(subset), 3),
                'mean_recipient_energy_gain': round(sum(r[4] for r in subset) / len(subset), 3),
                'final_greedy_share': int(action_value(q, state, 'share') > action_value(q, state, 'keep')),
            })
    return rows


def main():
    all_rows = []
    print('PROJECT GENESIS — EXPERIMENT 2.6: STABLE SOCIAL OBJECTIVES')
    print('Training: B reciprocates 85%, C reciprocates 10%. D is unseen.')
    print('Evaluation: B, C, D NEVER reciprocate. Online learning stays active.')
    print('SAME reward in training and testing: donor_change + weight * recipient_gain')
    print('Sharing costs A 5 and gives partner 8. No survival or consciousness claims.')
    print(f'{len(SEEDS)} independent seeds | {TRAIN_INTERACTIONS} training interactions/seed')
    print(f'{EVAL_INTERACTIONS} evaluation interactions/seed | {EVAL_EPSILON:.0%} evaluation exploration')
    for memory in (True, False):
        label = 'identity' if memory else 'pooled'
        print(f'\nMEMORY: {label}')
        for weight in WEIGHTS:
            rows = []
            for seed in SEEDS:
                q = train(seed, weight, memory)
                rows.extend(evaluate(seed, weight, memory, q))
            all_rows.extend(rows)
            summary = []
            for partner in PARTNERS:
                early = [r['help_pct'] for r in rows if r['partner'] == partner and r['phase'] == 'early']
                late = [r['help_pct'] for r in rows if r['partner'] == partner and r['phase'] == 'late']
                final = [r['final_greedy_share'] for r in rows if r['partner'] == partner and r['phase'] == 'late']
                summary.append(f'{partner} {sum(early)/len(early):.1f}%→{sum(late)/len(late):.1f}% (greedy {100*sum(final)/len(final):.0f}%)')
            print(f'  weight={weight:.2f} | ' + ' | '.join(summary))
    output = Path(__file__).with_name('stable_social_objectives_results.csv')
    with output.open('w', newline='', encoding='utf-8') as file:
        writer = csv.DictWriter(file, fieldnames=list(all_rows[0]))
        writer.writeheader()
        writer.writerows(all_rows)
    print(f'\nCSV saved to: {output}')
    print('Interpretation: helping can be directly favored by the programmed social reward.')
    print('A stable policy is not evidence of intrinsic attachment or moral values.')


if __name__ == '__main__':
    main()
