import random
import statistics

EPISODES = 3000
STEPS = 100
TRIALS = 500
ACTIONS = ('search', 'rest', 'share')
ENERGY_LEVELS = ('critical', 'low', 'healthy')


def state_for(me, other):
    def band(e):
        return 0 if e < 20 else 1 if e < 60 else 2
    return (band(me), band(other))


def initialize_q():
    return {(a, b, action): 0.0 for a in range(3) for b in range(3) for action in ACTIONS}


def select(q, state, rng, epsilon):
    if rng.random() < epsilon:
        return rng.choice(ACTIONS)
    return max(ACTIONS, key=lambda action: q[(*state, action)])


def run_episode(qs, rng, learn, share_enabled=True, epsilon=0.0, max_steps=STEPS):
    energy = [50, 50]
    shares = [0, 0]
    costly_shares = [0, 0]
    first_death = None

    for step in range(max_steps):
        # Each organism acts once in a randomly ordered turn.
        for i in rng.sample((0, 1), 2):
            j = 1 - i
            if energy[i] <= 0:
                continue
            state = state_for(energy[i], energy[j])
            options = ACTIONS if share_enabled else ACTIONS[:2]
            if learn:
                action = select(qs[i], state, rng, epsilon)
                if action not in options:
                    action = 'search'
            else:
                action = rng.choice(options)

            old_energy = energy[i]
            if action == 'search':
                energy[i] -= 6
                if rng.random() < 0.45:
                    energy[i] += 16
            elif action == 'rest':
                energy[i] -= 2
            elif action == 'share':
                # Sharing only transfers energy when partner is alive and receiver has capacity.
                if energy[j] > 0 and energy[j] < 100 and energy[i] > 10:
                    amount = min(10, energy[i] - 1, 100 - energy[j])
                    energy[i] -= amount
                    energy[j] += amount
                    shares[i] += 1
                    if old_energy < 35:
                        costly_shares[i] += 1
                energy[i] -= 2

            energy[i] = min(100, energy[i])
            # Baseline rewards are individualistic: +1 for survival, negative for death.
            reward = 1.0 if energy[i] > 0 else -10.0
            # No reward for partner survival or sharing in the baseline condition.
            next_state = state_for(energy[i], energy[j])
            future = max(qs[i][(*next_state, a)] for a in options) if energy[i] > 0 else 0.0
            if learn:
                key = (*state, action)
                qs[i][key] += 0.1 * (reward + 0.9 * future - qs[i][key])

        if first_death is None and (energy[0] <= 0 or energy[1] <= 0):
            first_death = step + 1
        if energy[0] <= 0 and energy[1] <= 0:
            return {'lifespan': step + 1, 'both_alive': False,
                    'shares': sum(shares), 'costly_shares': sum(costly_shares)}

    return {'lifespan': first_death if first_death is not None else max_steps,
            'both_alive': all(e > 0 for e in energy),
            'shares': sum(shares), 'costly_shares': sum(costly_shares)}


def train(seed=42):
    rng = random.Random(seed)
    qs = [initialize_q(), initialize_q()]
    for _ in range(EPISODES):
        run_episode(qs, rng, learn=True, epsilon=0.2)
    return qs


def evaluate(qs, seed, learned, share_enabled):
    rng = random.Random(seed)
    results = [run_episode(qs, rng, learn=False, share_enabled=share_enabled,
                           epsilon=0) for _ in range(TRIALS)] if not learned else [
        run_episode_greedy(qs, rng, share_enabled) for _ in range(TRIALS)]
    return {k: statistics.mean(r[k] for r in results) for k in results[0]}


def run_episode_greedy(qs, rng, share_enabled):
    # Evaluation uses frozen Q-values, unlike training.
    energy = [50, 50]
    shares = 0
    costly = 0
    first_death = None
    options = ACTIONS if share_enabled else ACTIONS[:2]
    for step in range(STEPS):
        for i in rng.sample((0, 1), 2):
            j = 1 - i
            if energy[i] <= 0:
                continue
            state = state_for(energy[i], energy[j])
            action = max(options, key=lambda a: qs[i][(*state, a)])
            original = energy[i]
            if action == 'search':
                energy[i] -= 6
                if rng.random() < 0.45:
                    energy[i] += 16
            elif action == 'rest':
                energy[i] -= 2
            else:
                if energy[j] > 0 and energy[j] < 100 and energy[i] > 10:
                    amount = min(10, energy[i] - 1, 100 - energy[j])
                    energy[i] -= amount
                    energy[j] += amount
                    shares += 1
                    if original < 35:
                        costly += 1
                energy[i] -= 2
            energy[i] = min(100, energy[i])
        if first_death is None and min(energy) <= 0:
            first_death = step + 1
        if max(energy) <= 0:
            return {'lifespan': first_death, 'both_alive': False,
                    'shares': shares, 'costly_shares': costly}
    return {'lifespan': first_death or STEPS, 'both_alive': min(energy) > 0,
            'shares': shares, 'costly_shares': costly}


if __name__ == '__main__':
    q = train()
    print('PROJECT GENESIS — STAGE 2: BASELINE')
    print('Agents receive no reward for helping each other.\n')
    for label, learned, allowed in [('Learned with sharing available', True, True),
                                    ('Random with sharing available', False, True),
                                    ('Learned without sharing', True, False)]:
        r = evaluate(q, 2026, learned, allowed)
        print(label)
        print(f"  Both survive 100 steps: {100*r['both_alive']:.1f}%")
        print(f"  First death (or 100):  {r['lifespan']:.1f} steps")
        print(f"  Shares per episode:    {r['shares']:.2f}")
        print(f"  Shares below 35 energy: {r['costly_shares']:.2f}\n")
