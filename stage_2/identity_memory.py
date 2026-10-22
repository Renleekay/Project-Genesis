"""Project Genesis — 2.4: Does interaction history produce partner-specific trust?

Standalone, standard-library-only controlled experiment. Partner behaviors are FIXED
experimental fixtures, not learning organisms; only donor A learns. A's reward
is its own net energy change; it receives no social or identity reward.

Run: python stage_2/identity_memory.py
"""
import csv
import random
import statistics
from pathlib import Path

TRAIN_SEEDS = (11, 23, 37, 51, 79)
TRAIN_ROUNDS = 2400
EVAL_ROUNDS = 400
ALPHA = 0.12
EPSILON = 0.15
SHARE_COST = 5
RETURN_AMOUNT = 10
# Labels do NOT directly determine reward; these are experimental partners'
# observed probabilistic responses to assistance.
PARTNERS = {"B": 0.85, "C": 0.10}
ACTIONS = ("keep", "share")


def key_for(partner, use_identity):
    return partner if use_identity else "all_partners"


def new_values():
    return {}


def action_for(q, partner, use_identity, rng, training):
    key = key_for(partner, use_identity)
    if training and rng.random() < EPSILON:
        return rng.choice(ACTIONS)
    # When unsure, A preserves its own energy. This is a design choice.
    share_value = q.get((key, "share"), 0.0)
    keep_value = q.get((key, "keep"), 0.0)
    return "share" if share_value > keep_value else "keep"


def interaction(q, partner, reliability, use_identity, rng, training):
    """Donor starts at 70 energy; partner at 12 (in need of help).
    A may share five units. Some partners later return ten energy by
    cooperative foraging. The return probability differs by partner.
    Only donor net energy is used as the learning reward.
    """
    action = action_for(q, partner, use_identity, rng, training)
    returned = action == "share" and rng.random() < reliability
    net_energy = (-SHARE_COST + (RETURN_AMOUNT if returned else 0)) if action == "share" else 0
    if training:
        k = (key_for(partner, use_identity), action)
        q[k] = q.get(k, 0.0) + ALPHA * (net_energy - q.get(k, 0.0))
    return action, net_energy


def train(seed, use_identity):
    rng = random.Random(seed)
    q = new_values()
    for _ in range(TRAIN_ROUNDS):
        partner = rng.choice(tuple(PARTNERS))
        interaction(q, partner, PARTNERS[partner], use_identity, rng, True)
    return q


def evaluate(q, seed, use_identity, swapped=False):
    rng = random.Random(seed)
    results = []
    for partner, reliability in PARTNERS.items():
        # If swapped, observed partners' behavior is exchanged but their
        # visible ID labels stay unchanged. The model is NOT retrained.
        actual = PARTNERS["C" if partner == "B" else "B"] if swapped else reliability
        shares = 0
        nets = []
        for _ in range(EVAL_ROUNDS):
            act, net = interaction(q, partner, actual, use_identity, rng, False)
            shares += act == "share"
            nets.append(net)
        results.append(dict(partner=partner, sharing_pct=100*shares/EVAL_ROUNDS,
                            mean_net_energy=statistics.mean(nets)))
    # Never-seen identity D: no learned values, therefore default keep.
    new_shares = sum(interaction(q, "D", .85, use_identity, rng, False)[0] == "share"
                     for _ in range(EVAL_ROUNDS))
    results.append(dict(partner="D_unseen", sharing_pct=100*new_shares/EVAL_ROUNDS,
                        mean_net_energy=0.0))
    return results


def main():
    print('PROJECT GENESIS — EXPERIMENT 2.4: INDIVIDUAL IDENTITY AND MEMORY')
    print('Only donor A learns; B and C are controlled behavioral partners.')
    print('Reward = donor net energy only; no reward for recognizing a name.')
    print(f'{len(TRAIN_SEEDS)} seeds | {TRAIN_ROUNDS} training interactions per seed')
    print('B reciprocates after help with p=.85; C with p=.10\n')
    rows = []
    for seed in TRAIN_SEEDS:
        for identity in (True, False):
            q = train(seed, identity)
            for swapped in (False, True):
                for r in evaluate(q, seed+10000, identity, swapped):
                    rows.append(dict(seed=seed, memory=('identity' if identity else 'pooled'),
                                     condition=('swapped_behaviors' if swapped else 'original'), **r))
    for memory in ('identity', 'pooled'):
        print(f'MEMORY: {memory}')
        for condition in ('original', 'swapped_behaviors'):
            print(f'  {condition}')
            for p in ('B', 'C', 'D_unseen'):
                sub = [r for r in rows if r['memory']==memory and r['condition']==condition and r['partner']==p]
                print(f"    {p:8s} | shares={statistics.mean(r['sharing_pct'] for r in sub):5.1f}%"
                      f" | mean donor net energy={statistics.mean(r['mean_net_energy'] for r in sub):+5.2f}")
        print()
    path = Path(__file__).resolve().parent / 'identity_memory_results.csv'
    with path.open('w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    print('Results saved to:', path)
    print('Interpretation: this tests reward-based partner-specific trust, NOT')
    print('recognition of life, affection, consciousness, or moral value.')
    print('Swapped behaviors tests whether frozen choices follow remembered labels.')


if __name__ == '__main__':
    main()
