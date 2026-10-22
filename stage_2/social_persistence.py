"""Project Genesis — Stage 2.3: Robustness of learned resource sharing.

Run: python stage_2/social_persistence.py
Uses only the Python standard library. Does not alter prior experiments.
Measures learned behavior, NOT feelings or moral agency.
"""
import csv
import random
import statistics
from pathlib import Path

ACTIONS = ('search', 'rest', 'share')
WEIGHTS = (0.0, 0.5)
TRAIN_SEEDS = (11, 23, 37, 51, 79)
TRAIN_EPISODES = 1800
TEST_EPISODES = 200
STEPS = 100
ALPHA, GAMMA, EPSILON = 0.1, 0.9, 0.2

CONDITIONS = {
    'ordinary': {'food_probability': 0.45, 'initial': (50, 50)},
    'scarcity': {'food_probability': 0.25, 'initial': (50, 50)},
    'partner_in_danger': {'food_probability': 0.45, 'initial': (70, 12)},
}


def band(energy):
    if energy <= 0:
        return 0
    if energy < 20:
        return 1
    if energy < 60:
        return 2
    return 3


def fresh_q():
    return {(own, other, action): 0.0
            for own in range(4) for other in range(4) for action in ACTIONS}


def choose(q, observation, rng, exploration):
    if exploration and rng.random() < EPSILON:
        return rng.choice(ACTIONS)
    best = max(q[(*observation, a)] for a in ACTIONS)
    return rng.choice([a for a in ACTIONS if abs(q[(*observation, a)] - best) < 1e-9])


def run_episode(qs, rng, *, social_weight, training, food_probability, initial):
    energy = list(initial)
    first_death = STEPS
    shares = costly_shares = rescues = rescue_chances = 0
    for step in range(STEPS):
        observations = [(band(energy[i]), band(energy[1-i])) for i in (0, 1)]
        actions = [choose(qs[i], observations[i], rng, training) if energy[i] > 0 else None
                   for i in (0, 1)]
        # Opportunities are counted before actions. They are NOT individual intentions.
        for i in (0, 1):
            if energy[i] > 20 and 0 < energy[1-i] < 20:
                rescue_chances += 1
        for i in rng.sample((0, 1), 2):
            j = 1-i
            if energy[i] <= 0 or actions[i] is None:
                continue
            if actions[i] == 'search':
                energy[i] -= 6
                if rng.random() < food_probability:
                    energy[i] += 16
            elif actions[i] == 'rest':
                energy[i] -= 2
            else:
                if energy[i] > 12 and 0 < energy[j] < 100:
                    amount = min(10, energy[i]-3, 100-energy[j])
                    if amount > 0:
                        costly_shares += int(energy[i] < 35)
                        rescues += int(energy[i] > 20 and energy[j] < 20)
                        shares += 1
                        energy[i] -= amount
                        energy[j] += amount
                energy[i] -= 2
            energy[i] = min(100, energy[i])
        if first_death == STEPS and min(energy) <= 0:
            first_death = step+1
        if training:
            for i in (0, 1):
                if actions[i] is None:
                    continue
                j = 1-i
                own_alive = energy[i] > 0
                other_alive = energy[j] > 0
                reward = (1 if own_alive else -10) + social_weight*(1 if other_alive else -10)
                next_obs = (band(energy[i]), band(energy[j]))
                next_best = max(qs[i][(*next_obs, a)] for a in ACTIONS) if own_alive else 0
                key = (*observations[i], actions[i])
                qs[i][key] += ALPHA * (reward + GAMMA*next_best - qs[i][key])
        if max(energy) <= 0:
            break
    return dict(both_survive=int(min(energy) > 0), first_death=first_death,
                shares=shares, costly_shares=costly_shares, rescue_transfers=rescues,
                rescue_opportunities=rescue_chances)


def train(weight, seed):
    rng = random.Random(seed)
    qs = [fresh_q(), fresh_q()]
    for _ in range(TRAIN_EPISODES):
        run_episode(qs, rng, social_weight=weight, training=True,
                    food_probability=0.45, initial=(50, 50))
    return qs


def evaluate(qs, *, seed, condition):
    rng = random.Random(seed)
    reports = [run_episode(qs, rng, social_weight=0, training=False, **condition)
               for _ in range(TEST_EPISODES)]
    result = {key: statistics.mean(item[key] for item in reports)
              for key in reports[0]}
    result['both_survive_pct'] = result.pop('both_survive') * 100
    return result


def main():
    print('PROJECT GENESIS — EXPERIMENT 2.3: SOCIAL VALUE PERSISTENCE')
    print(f'{len(TRAIN_SEEDS)} independent training seeds; {TRAIN_EPISODES} episodes per training run')
    print(f'{TEST_EPISODES} evaluation episodes per run and condition; frozen policies\n')
    results = []
    for seed in TRAIN_SEEDS:
        for weight in WEIGHTS:
            qs = train(weight, seed)
            for idx, (name, condition) in enumerate(CONDITIONS.items()):
                measures = evaluate(qs, seed=10000 + seed*100 + idx, condition=condition)
                results.append(dict(training_seed=seed, social_weight=weight,
                                    condition=name, **measures))
    for name in CONDITIONS:
        print(f'CONDITION: {name}')
        for weight in WEIGHTS:
            sub = [r for r in results if r['condition'] == name and r['social_weight'] == weight]
            rate = statistics.mean(r['both_survive_pct'] for r in sub)
            sd = statistics.stdev(r['both_survive_pct'] for r in sub)
            sharing = statistics.mean(r['shares'] for r in sub)
            costly = statistics.mean(r['costly_shares'] for r in sub)
            rescues = statistics.mean(r['rescue_transfers'] for r in sub)
            print(f'  weight={weight:.1f} | both survive={rate:5.1f}% '
                  f'(SD across seeds={sd:4.1f}) | shares={sharing:5.2f} '
                  f'| costly={costly:5.2f} | rescues={rescues:5.2f}')
        print()
    output = Path(__file__).resolve().parent / 'social_persistence_results.csv'
    with output.open('w', newline='', encoding='utf-8') as fh:
        writer = csv.DictWriter(fh, fieldnames=results[0].keys())
        writer.writeheader()
        writer.writerows(results)
    print(f'CSV saved to: {output}')
    print('CAUTION: No agent identities or relationship memories are modeled.')
    print('A new partner would be indistinguishable from the original partner.')
    print('This experiment tests learned policy robustness, not intrinsic moral values.')


if __name__ == '__main__':
    main()
