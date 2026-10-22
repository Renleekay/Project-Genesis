"""Project Genesis - Stage 2.2: Social-reward learning experiment.

Self-contained standard-library experiment. Does NOT modify Stage 2.1 baseline.
Run: python stage_2/social_rewards.py
"""
import csv
import random
from pathlib import Path

ACTIONS = ('search', 'rest', 'share')
WEIGHTS = (0.0, 0.2, 0.5, 1.0)
TRAIN_EPISODES = 3500
TEST_EPISODES = 400
STEPS = 100
ALPHA = 0.1
GAMMA = 0.9
EPSILON = 0.2


def band(energy):
    if energy <= 0:
        return 0  # Dead
    if energy < 20:
        return 1  # Critical
    if energy < 60:
        return 2  # Low
    return 3  # Healthy


def state(me, partner):
    return (band(me), band(partner))


def new_q():
    return {(own, other, action): 0.0
            for own in range(4) for other in range(4)
            for action in ACTIONS}


def choose(q, observation, rng, explore):
    if explore and rng.random() < EPSILON:
        return rng.choice(ACTIONS)
    best = max(q[(*observation, a)] for a in ACTIONS)
    options = [a for a in ACTIONS if abs(q[(*observation, a)] - best) < 1e-9]
    return rng.choice(options)


def episode(qs, rng, weight, training=False, initial=(50, 50),
            force_partner_low=False):
    # 'weight' affects learning reward only, not evaluation behavior.
    energy = list(initial)
    transfers = 0
    costly = 0
    rescue_opportunities = 0
    rescues = 0
    first_death = None
    for t in range(STEPS):
        # Both agents observe energy bands before taking actions.
        observations = [state(energy[i], energy[1-i]) for i in (0, 1)]
        actions = [choose(qs[i], observations[i], rng, training) if energy[i] > 0 else None
                   for i in (0, 1)]
        # A rescue opportunity is a live partner in critical condition while donor
        # has enough energy to transfer. This is an observational metric, not intent.
        for i in (0, 1):
            j = 1 - i
            if energy[i] > 20 and 0 < energy[j] < 20:
                rescue_opportunities += 1
        # Apply actions in random order; each action is charged its cost.
        for i in rng.sample((0, 1), 2):
            j = 1-i
            if actions[i] is None or energy[i] <= 0:
                continue
            if actions[i] == 'search':
                energy[i] -= 6
                if rng.random() < 0.45:
                    energy[i] += 16
            elif actions[i] == 'rest':
                energy[i] -= 2
            else:
                # Share only with a living partner. Transfers require donor > 12.
                if energy[i] > 12 and 0 < energy[j] < 100:
                    amount = min(10, energy[i]-3, 100-energy[j])
                    if amount > 0:
                        was_costly = energy[i] < 35
                        was_rescue = energy[i] > 20 and energy[j] < 20
                        energy[i] -= amount
                        energy[j] += amount
                        transfers += 1
                        costly += int(was_costly)
                        rescues += int(was_rescue)
                energy[i] -= 2
            energy[i] = min(100, energy[i])
        if first_death is None and min(energy) <= 0:
            first_death = t+1
        if training:
            for i in (0, 1):
                if actions[i] is None:
                    continue
                j = 1-i
                own_alive = energy[i] > 0
                partner_alive = energy[j] > 0
                # Explicit social incentive, not emergent morality.
                reward = (1 if own_alive else -10) + weight * (1 if partner_alive else -10)
                nxt = state(energy[i], energy[j])
                future = max(qs[i][(*nxt, a)] for a in ACTIONS) if own_alive else 0
                k = (*observations[i], actions[i])
                qs[i][k] += ALPHA * (reward + GAMMA*future - qs[i][k])
        if max(energy) <= 0:
            break
    return {
        'both_survive_pct': int(min(energy) > 0)*100,
        'first_death_step': first_death if first_death is not None else STEPS,
        'transfers': transfers,
        'costly_transfers': costly,
        'rescue_opportunities': rescue_opportunities,
        'rescue_transfers': rescues,
    }


def train(weight, seed):
    rng = random.Random(seed)
    qs = [new_q(), new_q()]
    for _ in range(TRAIN_EPISODES):
        episode(qs, rng, weight, training=True)
    return qs


def evaluate(qs, seed, initial=(50, 50)):
    rng = random.Random(seed)
    reports = [episode(qs, rng, weight=0, training=False, initial=initial)
               for _ in range(TEST_EPISODES)]
    keys = reports[0].keys()
    return {key: sum(r[key] for r in reports)/len(reports) for key in keys}


def main():
    rows = []
    print('PROJECT GENESIS - EXPERIMENT 2.2: SOCIAL REWARDS')
    print(f'Training: {TRAIN_EPISODES} episodes/condition; testing: {TEST_EPISODES} episodes/condition')
    print('Testing freezes learning; no social rewards are supplied during evaluation.\n')
    for index, weight in enumerate(WEIGHTS):
        q = train(weight, seed=42)
        ordinary = evaluate(q, seed=2026)
        # Stress condition: A starts comfortable, B starts in critical state.
        stress = evaluate(q, seed=3026, initial=(70, 12))
        row = {'social_weight': weight,
               **{f'normal_{k}': v for k, v in ordinary.items()},
               **{f'stress_{k}': v for k, v in stress.items()}}
        rows.append(row)
        print(f'SOCIAL WEIGHT: {weight:.1f}')
        print(f'  Standard: both survive = {ordinary["both_survive_pct"]:.1f}% | first death = {ordinary["first_death_step"]:.1f} steps')
        print(f'  Standard: shares = {ordinary["transfers"]:.2f} | costly shares = {ordinary["costly_transfers"]:.2f}')
        print(f'  Partner in danger: shares = {stress["transfers"]:.2f} | rescue transfers = {stress["rescue_transfers"]:.2f}')
        print(f'  Partner in danger: both survive = {stress["both_survive_pct"]:.1f}%\n')
    out = Path(__file__).resolve().parent / 'social_rewards_results.csv'
    with out.open('w', newline='', encoding='utf-8') as fh:
        writer = csv.DictWriter(fh, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    print(f'Results saved to: {out}')
    print('NOTE: Partner-helping behavior is not proof of care, consciousness, or moral agency.')


if __name__ == '__main__':
    main()
