import random
import pytest
from inventory_sim import InventorySim, Params

DEMANDS = [18, 23, 15, 27, 19]
ACTIONS = [20, 25, 10, 30, 15]

# Hand-computed ground truth (decision-time inventory, pending order, sales, unmet, end inv, reward)
EXPECTED = [
    (25, (0,),  18, 0,  7,   93),
    (7,  (20,), 7,  16, 0, -126),
    (20, (25,), 15, 0,  5,  105),
    (30, (10,), 27, 0,  3,  147),
    (13, (30,), 13, 6,  0,   34),
]


def make(T=30, L=2, **kw):
    return InventorySim(Params(T=T, L=L, **kw))


def test_five_day_hand_example():
    env = make(T=30)
    env.reset(init_inventory=25, init_pipeline=[0])
    total = 0
    for day, (a, d) in enumerate(zip(ACTIONS, DEMANDS)):
        _, r, term, trunc, info = env.step(a, d)
        inv, pend, s, u, e, rew = EXPECTED[day]
        assert info["inventory_at_decision"] == inv
        assert info["pipeline_at_decision"] == pend
        assert (info["sales"], info["unmet"], info["end_inventory"]) == (s, u, e)
        assert r == rew
        assert not term and not trunc
        total += r
    assert total == 253


def test_order_arrives_exactly_L_days_later():
    for L in (1, 2, 3, 4):
        env = make(T=50, L=L, max_order=40)
        env.reset(init_inventory=0, init_pipeline=[0] * (L - 1))
        env.step(10, 0)                      # order 10 on day 0, no demand
        for day in range(1, L + 1):
            obs = env.observation()          # state at start of `day`, after arrivals
            if day < L:
                assert obs[0] == 0, f"L={L}: arrived early on day {day}"
            else:                            # arrives at START of day L, not before
                assert obs[0] == 10, f"L={L}: not arrived on day {L}"
            env.step(0, 0)


def test_lost_sales_no_backlog():
    env = make()
    env.reset(init_inventory=5, init_pipeline=[0])
    _, r, *_, info = env.step(0, 30)
    assert info["sales"] == 5 and info["unmet"] == 25 and info["end_inventory"] == 0
    # Next day: demand 0 -> unmet demand from yesterday must NOT reappear
    _, _, _, _, info2 = env.step(0, 0)
    assert info2["sales"] == 0 and info2["unmet"] == 0


def test_terminal_rule_masks_late_orders():
    # T=5, L=2: orders on days 3,4 (t+L >= T) must be executed as 0 and not charged
    env = make(T=5, L=2)
    env.reset(init_inventory=25, init_pipeline=[0])
    execd, rewards = [], []
    for a, d in zip([10, 10, 10, 10, 10], [0, 0, 0, 0, 0]):
        _, r, term, trunc, info = env.step(a, d)
        execd.append(info["executed_order"]); rewards.append(r)
    assert execd == [10, 10, 10, 0, 0]
    env.reset(init_inventory=25, init_pipeline=[0])
    for day in range(3):
        env.step(0, 0)
    _, r, *_ , info = env.step(40, 0)       # day 3: proposed 40, executed 0
    assert info["executed_order"] == 0
    assert r == -env.p.h * info["end_inventory"]


def test_truncation_not_termination():
    env = make(T=3)
    env.reset(init_inventory=10, init_pipeline=[0])
    flags = [env.step(0, 1)[2:4] for _ in range(3)]
    assert flags == [(False, False), (False, False), (False, True)]
    with pytest.raises(AssertionError):
        env.step(0, 1)


def test_observation_contains_time_fraction():
    env = make(T=10)
    obs = env.reset(init_inventory=5, init_pipeline=[0])
    assert obs == (5, 0, 0.0)
    obs, *_ = env.step(0, 1)
    assert obs[-1] == pytest.approx(0.1)


@pytest.mark.parametrize("L", [1, 2, 3])
def test_inventory_conservation_random(L):
    """initial stock + executed orders - sales == final on-hand + final pipeline."""
    rng = random.Random(0)
    env = make(T=90, L=L)
    env.reset(init_inventory=25, init_pipeline=[0] * (L - 1))
    total_sales = 0
    trunc = False
    while not trunc:
        _, _, _, trunc, info = env.step(rng.randint(0, 40), rng.randint(0, 40))
        total_sales += info["sales"]
        assert env.inventory >= 0 and all(x >= 0 for x in env.pipeline)
    assert env.initial_stock + env.total_executed_orders - total_sales == \
        env.inventory + sum(env.pipeline)
    # every executed order has arrived by the end (terminal rule), and arrivals are conserved
    assert sum(env.pipeline) == 0
    assert env.total_arrived == env.total_executed_orders


def test_no_executed_order_left_in_pipeline_at_end():
    for L in (1, 2, 3):
        env = make(T=20, L=L)
        env.reset(init_inventory=25, init_pipeline=[0] * (L - 1))
        trunc = False
        while not trunc:
            _, _, _, trunc, _ = env.step(40, 20)
        assert sum(env.pipeline) == 0   # no wasted/undeliverable terminal orders


def test_invalid_action_rejected():
    env = make()
    env.reset()
    with pytest.raises(AssertionError):
        env.step(41, 5)
    with pytest.raises(AssertionError):
        env.step(-1, 5)