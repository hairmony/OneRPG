# streamlit_uno_archetype_game_hotseat_FULL.py
# FULL APP (single file) with:
# - Human profiling: paste 1–5 comments -> KMeans cluster -> Gemini explains + passives + 20 insults
# - Weights: Redundancy/Bluerocity/Greenality/Yellowtude/Chaos sum to 100
# - Privilege: starting hand size 5..10 (separate from sum=100)
# - Hotseat pass-the-screen gate for human turns
# - UNO-like game: whole hand visible, only playable cards selectable
# - +2/+4 stacking rule: chain continues ONLY if another +2 or WILD4 is played; otherwise draw pending
# - Ultimates tied to class IDs 0..7, once per game
# - Insults: slider at lobby; humans can queue an insult; bots have 50% chance to queue an insult each bot turn
# - Bots: random class/passives/privilege; can use ultimate once per game (20% chance each bot turn)
#
# Requires:
#   pip install streamlit numpy scikit-learn joblib requests
# Files:
#   archetype_model.joblib  (trained bundle with vectorizer, svd, kmeans, centroids)
# Optional:
#   cluster_profiles.json   (for better Gemini explanations)
#
# Run:
#   streamlit run streamlit_uno_archetype_game_hotseat_FULL.py

import json
import re
import random
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import joblib
import requests
import streamlit as st

# =========================
# CONFIG
# =========================
API_URL = "https://hackathon-api-39535212257.northamerica-northeast2.run.app/api/generate"
API_KEY = "e480373c-62c7-49a8-863a-ef28c81a2431"

MODEL_PATH = "archetype_model.joblib"
CLUSTER_PROFILES_PATH = "cluster_profiles.json"  # optional

CLASS_MAPPING = {
    0: "Inquisitor",
    1: "Strategist",
    2: "Contrarian",
    3: "Journalist",
    4: "Lawyer",
    5: "Pragmatist",
    6: "Warrior",
    7: "Activist",
}

WEIGHT_KEYS = ["Redundancy", "Bluerocity", "Greenality", "Yellowtude", "Chaos"]  # sum to 100
PRIVILEGE_KEY = "Privilege"  # 5..10, separate from sum=100

COLORS = ["R", "B", "G", "Y"]
NUMS = list(range(0, 10))
ACTIONS = ["skip", "reverse", "draw2"]
WILDS = ["wild", "wild4"]

COLOR_EMOJI = {"R": "🟥", "B": "🟦", "G": "🟩", "Y": "🟨", None: "🃏"}

STACKABLE_KINDS = {"draw2", "wild4"}  # only these can be played while pending_draw > 0
RANDOM_SEED = 42


# =========================
# TEXT + MODEL UTILS
# =========================
def clean_text(s: str) -> str:
    if not isinstance(s, str):
        return ""
    s = s.replace("\n", " ").replace("\r", " ").strip()
    s = re.sub(r"http\S+", " ", s)
    s = re.sub(r"\s+", " ", s)
    return s


def normalize_weights_100(scores: dict) -> dict:
    scores = {k: int(scores.get(k, 0)) for k in WEIGHT_KEYS}
    total = sum(scores.values()) or 1
    scaled = {k: scores[k] * 100.0 / total for k in WEIGHT_KEYS}
    out = {k: int(round(scaled[k])) for k in WEIGHT_KEYS}
    diff = 100 - sum(out.values())
    if diff != 0:
        out[max(out, key=out.get)] += diff
    # clamp and final adjust
    for k in WEIGHT_KEYS:
        out[k] = max(0, min(100, int(out[k])))
    diff2 = 100 - sum(out.values())
    if diff2 != 0:
        out[max(out, key=out.get)] += diff2
    return out


def clamp_privilege(x) -> int:
    try:
        v = int(x)
    except Exception:
        v = 7
    return max(5, min(10, v))


def sanitize_insults(insults, fallback_class_name: str) -> List[str]:
    if not isinstance(insults, list):
        insults = []
    cleaned = []
    seen = set()
    for s in insults:
        s = str(s).strip()
        if s and s not in seen:
            seen.add(s)
            cleaned.append(s)

    pads = [
        "You definitely argue with your microwave.",
        "You’d bring a spreadsheet to a vibe check.",
        "You type like you’re filing a formal complaint to the universe.",
        f"You’ve got {fallback_class_name} energy in 4K.",
        "You could turn a yes/no question into a season finale.",
        "You’re the human embodiment of ‘well, actually…’ (affectionate).",
        "You play like your keyboard has a lawyer on retainer.",
        "You’d fact-check UNO. During UNO.",
    ]
    i = 0
    while len(cleaned) < 20:
        cleaned.append(pads[i % len(pads)])
        i += 1
    return cleaned[:20]


@st.cache_resource
def load_kmeans_bundle():
    return joblib.load(MODEL_PATH)


@st.cache_resource
def load_cluster_profiles():
    try:
        return json.load(open(CLUSTER_PROFILES_PATH, "r", encoding="utf-8"))
    except Exception:
        return None


def predict_cluster(text: str, bundle: dict) -> Tuple[int, Optional[float]]:
    vec = bundle["vectorizer"]
    svd = bundle["svd"]
    kmeans = bundle["kmeans"]

    X = vec.transform([text])
    Xs = svd.transform(X)
    norm = np.linalg.norm(Xs, axis=1, keepdims=True)
    Xs = Xs / (norm + 1e-12)

    cluster_id = int(kmeans.predict(Xs)[0])

    centroids = bundle.get("centroids")
    conf = None
    if centroids is not None:
        sims = (Xs @ centroids.T).ravel()
        conf = float(sims[cluster_id])

    return cluster_id, conf


def call_hackathon_gemini(prompt_obj: dict, response_schema: dict) -> dict:
    r = requests.post(
        API_URL,
        headers={"X-API-Key": API_KEY, "Content-Type": "application/json"},
        json={
            "contents": json.dumps(prompt_obj, ensure_ascii=False),
            "model": "gemini-2.5-flash",
            "response_schema": response_schema,
        },
        timeout=90,
    )
    r.raise_for_status()
    data = r.json()
    return json.loads(data["text"])


# =========================
# GAME ENGINE
# =========================
@dataclass
class PlayerProfile:
    name: str
    is_bot: bool
    weights: Dict[str, int]  # WEIGHT_KEYS only
    privilege: int  # 5..10
    class_id: int = -1
    class_name: str = "Balanced"
    roasts: Optional[List[str]] = None


def make_deck() -> List[dict]:
    deck = []
    for c in COLORS:
        for n in NUMS:
            deck.append({"color": c, "kind": "num", "value": n})
            if n != 0:
                deck.append({"color": c, "kind": "num", "value": n})
        for a in ACTIONS:
            deck.append({"color": c, "kind": a})
            deck.append({"color": c, "kind": a})
    for _ in range(4):
        deck.append({"color": None, "kind": "wild"})
        deck.append({"color": None, "kind": "wild4"})
    random.shuffle(deck)
    return deck


def card_str(card: dict) -> str:
    if card["kind"] == "num":
        return f'{card["color"]}{card["value"]}'
    if card["kind"] in ("skip", "reverse", "draw2"):
        return f'{card["color"]} {card["kind"].upper()}'
    if card["kind"] == "plus8":
        return "PLUS8"
    return card["kind"].upper()


def playable(card: dict, top: dict, active_color: Optional[str]) -> bool:
    # PLUS8 is always playable
    if card["kind"] == "plus8":
        return True

    top_color = active_color if active_color else top.get("color")
    if card["kind"] in ("wild", "wild4"):
        return True
    if card.get("color") == top_color and top_color is not None:
        return True
    if card.get("kind") == top.get("kind") and card.get("kind") != "num":
        return True
    if card.get("kind") == "num" and top.get("kind") == "num" and card.get("value") == top.get("value"):
        return True
    return False


def reshuffle_if_needed(state: dict):
    if state["deck"]:
        return
    top = state["discard"][-1]
    rest = state["discard"][:-1]
    random.shuffle(rest)
    state["deck"] = rest
    state["discard"] = [top]


def weighted_color_choice(weights: Dict[str, int]) -> str:
    items = [
        ("R", max(0, int(weights.get("Redundancy", 0)))),
        ("B", max(0, int(weights.get("Bluerocity", 0)))),
        ("G", max(0, int(weights.get("Greenality", 0)))),
        ("Y", max(0, int(weights.get("Yellowtude", 0)))),
    ]
    total = sum(w for _, w in items) or 1
    r = random.randint(1, total)
    acc = 0
    for c, w in items:
        acc += w
        if r <= acc:
            return c
    return "R"


def draw_one_biased(state: dict, profile: PlayerProfile) -> dict:
    reshuffle_if_needed(state)

    # mild bias toward "preferred color" and (if chaos high) action/wilds
    bias_prob = 0.35
    chaos = max(0, int(profile.weights.get("Chaos", 0)))
    action_bias = min(0.25, chaos / 400.0)

    if random.random() > bias_prob:
        return state["deck"].pop()

    desired_color = weighted_color_choice(profile.weights)

    # action bias
    if random.random() < action_bias:
        for i in range(len(state["deck"]) - 1, -1, -1):
            c = state["deck"][i]
            if c["kind"] in ("wild", "wild4"):
                return state["deck"].pop(i)
            if c.get("color") == desired_color and c["kind"] in ("skip", "reverse", "draw2"):
                return state["deck"].pop(i)

    # color bias
    for i in range(len(state["deck"]) - 1, -1, -1):
        c = state["deck"][i]
        if c.get("color") == desired_color:
            return state["deck"].pop(i)

    reshuffle_if_needed(state)
    return state["deck"].pop()


def draw_cards(state: dict, who: str, n: int):
    profile = state["profiles"][who]
    for _ in range(n):
        state["hands"][who].append(draw_one_biased(state, profile))


def next_turn(state: dict, skip: bool = False):
    n = len(state["turn_order"])
    if skip:
        state["turn_idx"] = (state["turn_idx"] + 2 * state["direction"]) % n
    else:
        state["turn_idx"] = (state["turn_idx"] + 1 * state["direction"]) % n


def start_game(profiles: List[PlayerProfile]) -> dict:
    random.seed(RANDOM_SEED)
    deck = make_deck()
    names = [p.name for p in profiles]
    profile_map = {p.name: p for p in profiles}

    state = {
        "deck": deck,
        "discard": [],
        "hands": {name: [] for name in names},
        "profiles": profile_map,
        "turn_order": names,
        "turn_idx": 0,
        "direction": 1,
        "pending_draw": 0,
        "active_color": None,
        "log": [],
        "winner": None,

        # insults
        "pending_insults": {name: [] for name in names},

        # ultimates
        "ultimate_used": {name: False for name in names},
        "must_play": {name: False for name in names},  # for PLUS8 enforcement
        "filibuster": None,  # {"owner": str, "turns_left": int}
    }

    # deal
    for p in profiles:
        draw_cards(state, p.name, p.privilege)

    # flip start card (avoid wild4)
    top = state["deck"].pop()
    while top["kind"] == "wild4":
        state["deck"].insert(0, top)
        random.shuffle(state["deck"])
        top = state["deck"].pop()
    state["discard"].append(top)

    if top["kind"] == "wild":
        state["active_color"] = random.choice(COLORS)
        state["log"].append(f"Initial WILD. Color set to {state['active_color']}.")

    return state


def maybe_apply_filibuster_prevent_win(state: dict, actor: str) -> bool:
    fb = state.get("filibuster")
    if not fb:
        return False
    if actor == fb["owner"]:
        return False
    if len(state["hands"][actor]) == 0:
        draw_cards(state, actor, 2)
        state["log"].append(f"FILIBUSTER: {actor} tried to win, but draws 2.")
        return True
    return False


def check_winner(state: dict):
    for name, hand in state["hands"].items():
        if len(hand) == 0:
            state["winner"] = name
            return


def tick_filibuster(state: dict, actor_just_finished: str):
    fb = state.get("filibuster")
    if not fb:
        return
    if actor_just_finished != fb["owner"]:
        fb["turns_left"] -= 1
        if fb["turns_left"] <= 0:
            state["log"].append("FILIBUSTER expires.")
            state["filibuster"] = None


def apply_card_effects(state: dict, card: dict, actor: str, chosen_color: Optional[str] = None):
    # Do NOT clear pending_draw here; chain ends only when someone draws the pending.
    state["active_color"] = None

    if card["kind"] == "plus8":
        state["active_color"] = chosen_color or random.choice(COLORS)
        state["pending_draw"] += 8
        state["log"].append(
            f"{actor} plays PLUS8, sets color to {state['active_color']} (+8 pending = {state['pending_draw']})."
        )
        return

    if card["kind"] == "wild":
        state["active_color"] = chosen_color or random.choice(COLORS)
        state["log"].append(f"{actor} sets color to {state['active_color']}.")
        return

    if card["kind"] == "wild4":
        state["active_color"] = chosen_color or random.choice(COLORS)
        state["pending_draw"] += 4
        state["log"].append(f"{actor} sets color to {state['active_color']} (+4 pending = {state['pending_draw']}).")
        return

    if card["kind"] == "draw2":
        state["pending_draw"] += 2
        state["log"].append(f"{actor} adds +2 (pending = {state['pending_draw']}).")
        return

    if card["kind"] == "reverse":
        if len(state["turn_order"]) == 2:
            state["log"].append(f"{actor} plays REVERSE (acts as SKIP with 2 players).")
            next_turn(state, skip=True)
            return
        state["direction"] *= -1
        state["log"].append(f"{actor} reverses direction.")
        return

    if card["kind"] == "skip":
        state["log"].append(f"{actor} skips next player.")
        next_turn(state, skip=True)
        return


# =========================
# ULTIMATES
# =========================
ULT_INFO = {
    0: ("Peer Review", "Swap hands with an opponent."),
    1: ("Hostile Takeover", "Choose a color. Opponent draws until they draw a card of that color."),
    2: ("Red Herring", "Turn all COLORED cards in opponent hand into RED 2s."),
    3: ("Clickbait", "Instantly play all cards of one color from your hand that matches the pile."),
    4: ("Filibuster", "For the next 2 opponent turns: if they empty their hand, they draw 2 instead."),
    5: ("Market Crash", "Opponent draws (# of non-numbered cards in their hand) + 1."),
    6: ("Anarchy", "Put a PLUS8 card into a player's hand. They must play it next turn."),
    7: ("Social Equality", "Opponent draws or discards until their hand size equals yours."),
}


def do_peer_review(state: dict, actor: str, target: str):
    state["hands"][actor], state["hands"][target] = state["hands"][target], state["hands"][actor]
    state["log"].append(f"{actor} uses Peer Review on {target}: hands swapped.")


def do_hostile_takeover(state: dict, actor: str, target: str, color: str):
    drawn = 0
    while True:
        reshuffle_if_needed(state)
        if not state["deck"]:
            break
        c = state["deck"].pop()
        state["hands"][target].append(c)
        drawn += 1
        if c.get("color") == color:
            break
    state["log"].append(f"{actor} uses Hostile Takeover on {target}: drew {drawn} to hit {color}.")


def do_red_herring(state: dict, actor: str, target: str):
    new_hand = []
    changed = 0
    for c in state["hands"][target]:
        if c.get("color") in COLORS:
            new_hand.append({"color": "R", "kind": "num", "value": 2})
            changed += 1
        else:
            new_hand.append(c)
    state["hands"][target] = new_hand
    state["log"].append(f"{actor} uses Red Herring on {target}: {changed} cards became R2.")


def do_clickbait(state: dict, actor: str, chosen_color: str):
    # Requirement to match pile color removed
    hand = state["hands"][actor]
    idxs = [i for i, c in enumerate(hand) if c.get("color") == chosen_color and c.get("color") in COLORS]
    
    if not idxs:
        state["log"].append(f"{actor} uses Clickbait but has no {chosen_color} cards.")
        return

    dumped = []
    for i in sorted(idxs, reverse=True):
        dumped.append(hand.pop(i))
    dumped.reverse()

    for c in dumped:
        state["discard"].append(c)

    last = dumped[-1]
    state["log"].append(f"{actor} uses Clickbait: dumps {len(dumped)} {chosen_color} cards (last = {card_str(last)}).")

    # Apply effect for last dumped card
    if last.get("kind") in ("draw2", "skip", "reverse"):
        apply_card_effects(state, last, actor, chosen_color=None)
    idxs = [i for i, c in enumerate(hand) if c.get("color") == chosen_color and c.get("color") in COLORS]
    if not idxs:
        state["log"].append(f"{actor} uses Clickbait but has no {chosen_color} cards.")
        return

    dumped = []
    for i in sorted(idxs, reverse=True):
        dumped.append(hand.pop(i))
    dumped.reverse()

    for c in dumped:
        state["discard"].append(c)

    last = dumped[-1]
    state["log"].append(f"{actor} uses Clickbait: dumps {len(dumped)} {chosen_color} cards (last = {card_str(last)}).")

    # apply effect only for last dumped
    if last.get("kind") in ("draw2", "skip", "reverse"):
        apply_card_effects(state, last, actor, chosen_color=None)


def do_filibuster(state: dict, actor: str):
    state["filibuster"] = {"owner": actor, "turns_left": 2}
    state["log"].append(f"{actor} uses Filibuster: next 2 opponent turns prevent winning (draw 2 instead).")


def do_market_crash(state: dict, actor: str, target: str):
    non_num = sum(1 for c in state["hands"][target] if c.get("kind") != "num")
    n = non_num + 1
    draw_cards(state, target, n)
    state["log"].append(f"{actor} uses Market Crash on {target}: non-num={non_num}, draws {n}.")


def do_anarchy(state: dict, actor: str, target: str):
    state["hands"][target].append({"color": None, "kind": "plus8"})
    state["must_play"][target] = True
    state["log"].append(f"{actor} uses Anarchy on {target}: PLUS8 added, must be played next turn.")


def do_social_equality(state: dict, actor: str, target: str):
    my_size = len(state["hands"][actor])
    t_size = len(state["hands"][target])
    if t_size == my_size:
        state["log"].append(f"{actor} uses Social Equality on {target}: already equal.")
        return

    if t_size < my_size:
        n = my_size - t_size
        draw_cards(state, target, n)
        state["log"].append(f"{actor} uses Social Equality: {target} draws {n} to match {my_size}.")
    else:
        n = t_size - my_size
        for _ in range(n):
            if not state["hands"][target]:
                break
            i = random.randrange(len(state["hands"][target]))
            state["discard"].append(state["hands"][target].pop(i))
        state["log"].append(f"{actor} uses Social Equality: {target} discards {n} to match {my_size}.")


def use_ultimate(state: dict, actor: str, ult_id: int, target: Optional[str], color: Optional[str]):
    if state["ultimate_used"][actor]:
        return

    if ult_id == 0 and target:
        do_peer_review(state, actor, target)
    elif ult_id == 1 and target and color:
        do_hostile_takeover(state, actor, target, color)
    elif ult_id == 2 and target:
        do_red_herring(state, actor, target)
    elif ult_id == 3 and color:
        do_clickbait(state, actor, color)
    elif ult_id == 4:
        do_filibuster(state, actor)
    elif ult_id == 5 and target:
        do_market_crash(state, actor, target)
    elif ult_id == 6 and target:
        do_anarchy(state, actor, target)
    elif ult_id == 7 and target:
        do_social_equality(state, actor, target)
    else:
        state["log"].append(f"{actor} tried an invalid ultimate.")
        return

    state["ultimate_used"][actor] = True
    state["log"].append(f"{actor} ultimate used up for this game.")


# =========================
# BOTS: insult + ultimate
# =========================
def random_bot_weights_and_priv() -> Tuple[dict, int]:
    raw = {k: random.randint(5, 35) for k in WEIGHT_KEYS}
    weights = normalize_weights_100(raw)
    privilege = random.randint(5, 10)
    return weights, privilege


def pick_human_targets(state: dict, exclude: str) -> List[str]:
    return [n for n in state["turn_order"] if n != exclude and not state["profiles"][n].is_bot]


def queue_insult(state: dict, attacker: str, target: str):
    attacker_profile = state["profiles"][attacker]
    pool = attacker_profile.roasts or [
        "You argue with loading screens.",
        "You’d bring a rubric to a pillow fight.",
        "You have ‘reply guy’ energy, but make it UNO.",
        "Your strategy is vibes and mild panic.",
        "You could make a group chat file a restraining order.",
        "You play like you’re negotiating a hostage exchange.",
        "You’re the human version of a pop-up disclaimer.",
        "Your takes have takes.",
    ]
    state["pending_insults"][target].append(random.choice(pool))
    state["log"].append(f"{attacker} queued an insult for {target}.")


def bot_choose_color(profile: PlayerProfile) -> str:
    return weighted_color_choice(profile.weights)


def bot_maybe_use_ultimate(state: dict, actor: str) -> bool:
    prof = state["profiles"][actor]
    if prof.class_id not in ULT_INFO:
        return False
    if state["ultimate_used"].get(actor):
        return False

    # 20% chance per bot turn
    if random.random() > 0.20:
        return False

    targets = [n for n in state["turn_order"] if n != actor]
    if not targets:
        return False

    # Prefer the opponent with the biggest hand for punishment ults
    target = max(targets, key=lambda n: len(state["hands"][n]))

    pile_color = state["active_color"] if state["active_color"] else state["discard"][-1].get("color")
    color = random.choice(COLORS)

    if prof.class_id == 3:
        # Clickbait: choose pile color if valid
        if pile_color in COLORS:
            color = pile_color
        else:
            color = random.choice(COLORS)

    if prof.class_id == 1:
        # Hostile Takeover: choose a color target has least of
        counts = {c: 0 for c in COLORS}
        for card in state["hands"][target]:
            if card.get("color") in COLORS:
                counts[card["color"]] += 1
        color = min(counts, key=counts.get)

    use_ultimate(state, actor, prof.class_id, target, color)
    state["log"].append(f"{actor} (bot) used ultimate!")
    return True


def bot_take_turn(state: dict):
    actor = state["turn_order"][state["turn_idx"]]
    profile = state["profiles"][actor]
    top = state["discard"][-1]
    hand = state["hands"][actor]

    # must-play PLUS8 enforcement
    if state["must_play"].get(actor):
        plus8_idxs = [i for i, c in enumerate(hand) if c.get("kind") == "plus8"]
        if plus8_idxs:
            i = plus8_idxs[0]
            card = hand.pop(i)
            state["discard"].append(card)
            state["must_play"][actor] = False
            chosen = bot_choose_color(profile)
            state["log"].append(f"{actor} is forced to play PLUS8.")
            apply_card_effects(state, card, actor, chosen_color=chosen)
            if maybe_apply_filibuster_prevent_win(state, actor):
                pass
            else:
                check_winner(state)
            next_turn(state)
            tick_filibuster(state, actor)
            return
        else:
            state["must_play"][actor] = False  # safety

    # stacking rule: while pending_draw > 0, ONLY draw2/wild4 can be played; otherwise draw pending and lose turn
    if state["pending_draw"] > 0:
        stack_idxs = [
            i for i, c in enumerate(hand)
            if c.get("kind") in STACKABLE_KINDS and playable(c, top, state["active_color"])
        ]
        if not stack_idxs:
            n = state["pending_draw"]
            draw_cards(state, actor, n)
            state["log"].append(f"{actor} draws {n} (couldn't stack).")
            state["pending_draw"] = 0
            next_turn(state)
            tick_filibuster(state, actor)
            return

        # prefer wild4 over draw2
        def s(c): return 2 if c.get("kind") == "wild4" else 1
        best_i = max(stack_idxs, key=lambda i: s(hand[i]))
        card = hand.pop(best_i)
        state["discard"].append(card)
        chosen = bot_choose_color(profile) if card.get("kind") == "wild4" else None
        state["log"].append(f"{actor} stacks {card_str(card)}.")
        apply_card_effects(state, card, actor, chosen_color=chosen)
        if maybe_apply_filibuster_prevent_win(state, actor):
            pass
        else:
            check_winner(state)
        next_turn(state)
        tick_filibuster(state, actor)
        return

    # normal play
    playable_idxs = [i for i, c in enumerate(hand) if playable(c, top, state["active_color"])]
    if not playable_idxs:
        draw_cards(state, actor, 1)
        state["log"].append(f"{actor} draws 1.")
        next_turn(state)
        tick_filibuster(state, actor)
        return

    def score_card(c):
        if c["kind"] in ("plus8",):
            return 7
        if c["kind"] in ("wild4", "draw2"):
            return 6
        if c["kind"] in ("wild", "skip", "reverse"):
            return 5
        return 1

    best_i = max(playable_idxs, key=lambda i: score_card(hand[i]))
    card = hand.pop(best_i)
    state["discard"].append(card)
    state["log"].append(f"{actor} plays {card_str(card)}.")

    chosen = None
    if card["kind"] in ("wild", "wild4", "plus8"):
        chosen = bot_choose_color(profile)

    apply_card_effects(state, card, actor, chosen_color=chosen)

    if maybe_apply_filibuster_prevent_win(state, actor):
        pass
    else:
        check_winner(state)
    if state.get("winner"):
        return

    # skip/reverse(2p) may have advanced turn already
    if card["kind"] == "skip":
        tick_filibuster(state, actor)
        return
    if card["kind"] == "reverse" and len(state["turn_order"]) == 2:
        tick_filibuster(state, actor)
        return

    next_turn(state)
    tick_filibuster(state, actor)


# =========================
# STREAMLIT APP
# =========================
st.set_page_config(page_title="OneRPG", layout="centered")
st.title("OneRPG — Non-Spanish Card Game")

bundle = load_kmeans_bundle()
profiles_db = load_cluster_profiles()

if "stage" not in st.session_state:
    st.session_state.stage = "lobby"
if "lobby" not in st.session_state:
    st.session_state.lobby = None
if "profiles" not in st.session_state:
    st.session_state.profiles = []
if "profiling_idx" not in st.session_state:
    st.session_state.profiling_idx = 0
if "game" not in st.session_state:
    st.session_state.game = None
if "turn_gate" not in st.session_state:
    st.session_state.turn_gate = True

# =========================
# LOBBY
# =========================
if st.session_state.stage == "lobby":
    st.caption(
        "Each human player pastes 1–5 comments to get class + passives.\n\n"
        "Rule: **+2/+4 chains only continue if another +2 or +4 is played**."
    )

    total_players = st.slider("Total players", 2, 4, 2)
    humans = st.slider("Human players", 1, total_players, 2 if total_players >= 2 else 1)

    tone = st.select_slider(
        "Insult intensity",
        options=["Wholesome", "Light", "Savage (still playful)"],
        value="Light",
    )

    st.markdown("### Player names")
    human_names = [
        st.text_input(f"Human {i+1} name", value=f"Player {i+1}", key=f"hn_{i}")
        for i in range(humans)
    ]
    bot_names = [
        st.text_input(f"Bot {i+1} name", value=f"Bot {i+1}", key=f"bn_{i}")
        for i in range(total_players - humans)
    ]

    colA, colB = st.columns([1, 1])
    if colA.button("Start profiling", type="primary", use_container_width=True):
        st.session_state.lobby = {
            "total_players": total_players,
            "humans": humans,
            "human_names": [n.strip() or f"Player {i+1}" for i, n in enumerate(human_names)],
            "bot_names": [n.strip() or f"Bot {i+1}" for i, n in enumerate(bot_names)],
            "tone": tone,
        }
        st.session_state.profiles = []
        st.session_state.profiling_idx = 0
        st.session_state.stage = "profiling"
        st.session_state.turn_gate = True
        st.rerun()

    if colB.button("Reset", use_container_width=True):
        st.session_state.lobby = None
        st.session_state.profiles = []
        st.session_state.profiling_idx = 0
        st.session_state.game = None
        st.session_state.turn_gate = True
        st.rerun()

# =========================
# PROFILING
# =========================
if st.session_state.stage == "profiling":
    lobby = st.session_state.lobby
    if not lobby:
        st.session_state.stage = "lobby"
        st.rerun()

    idx = st.session_state.profiling_idx
    humans = lobby["humans"]
    if idx >= humans:
        st.session_state.stage = "setup_review"
        st.rerun()

    name = lobby["human_names"][idx]
    tone = lobby["tone"]

    st.subheader(f"Profiling: {name}")
    st.info("Pass the device to this player. Paste **1–5** comments/posts, then generate class + passives.")

    inputs = []
    for i in range(1, 6):
        inputs.append(
            st.text_area(
                f"{name} — Post/Comment #{i} (optional)",
                height=90,
                placeholder="Paste text here…",
                key=f"profile_{idx}_{i}",
            )
        )

    colA, colB = st.columns([1, 1])
    if colA.button("Generate my class + passives", type="primary", use_container_width=True):
        user_texts = [clean_text(t) for t in inputs if clean_text(t)]
        if not user_texts:
            st.error("Paste at least 1 comment/post.")
            st.stop()

        combined = "\n\n".join(user_texts)
        cluster_id, conf = predict_cluster(combined, bundle)
        class_name = CLASS_MAPPING.get(cluster_id, f"Class {cluster_id}")

        cluster_context = None
        if profiles_db and str(cluster_id) in profiles_db:
            p = profiles_db[str(cluster_id)]
            cluster_context = {
                "avg_length": p.get("avg_length"),
                "question_ratio": p.get("question_ratio"),
                "top_keywords": p.get("top_keywords", [])[:15],
                "example_comments": [c[:220] for c in p.get("example_comments", [])[:6]],
            }

        response_schema = {
            "type": "object",
            "properties": {
                "cluster_id": {"type": "integer"},
                "class_name": {"type": "string"},
                "why": {"type": "array", "items": {"type": "string"}},
                "passive_scores": {
                    "type": "object",
                    "properties": {
                        "Redundancy": {"type": "integer"},
                        "Bluerocity": {"type": "integer"},
                        "Greenality": {"type": "integer"},
                        "Yellowtude": {"type": "integer"},
                        "Chaos": {"type": "integer"},
                        "Privilege": {"type": "integer"},
                    },
                    "required": WEIGHT_KEYS + [PRIVILEGE_KEY],
                },
                "silly_insults": {"type": "array", "items": {"type": "string"}},
            },
            "required": ["cluster_id", "class_name", "why", "passive_scores", "silly_insults"],
        }

        prompt_obj = {
            "task": "explain_cluster_assign_passives_and_roast",
            "constraints": [
                "Return STRICT JSON only matching the schema.",
                "Redundancy + Bluerocity + Greenality + Yellowtude + Chaos MUST be integers and MUST sum to exactly 100.",
                "Privilege must be an integer between 5 and 10 (starting hand size). Privilege is NOT included in the 100-sum.",
                "Explain in 4-7 bullet points in the 'why' array.",
                "Generate exactly 20 silly_insults.",
                "Silly insults must be playful, non-hateful, non-political, and must not target identity or protected groups.",
                f"Roast tone: {tone}.",
            ],
            "cluster_id": cluster_id,
            "class_name": class_name,
            "class_mapping": CLASS_MAPPING,
            "user_inputs": user_texts[:5],
            "cluster_context": cluster_context,
        }

        with st.spinner("Calling Gemini..."):
            out = call_hackathon_gemini(prompt_obj, response_schema)

        ps = out.get("passive_scores", {})
        weights = normalize_weights_100(ps)
        privilege = clamp_privilege(ps.get("Privilege", 7))
        roasts = sanitize_insults(out.get("silly_insults", []), class_name)

        st.success(f"{name} is **{class_name}**")
        if conf is not None:
            st.caption(f"Confidence: {conf:.2f}")

        st.info(f"🖐️ Starting hand size (Privilege): **{privilege}**")
        st.json(weights)
        st.success(random.choice(roasts))

        st.session_state.profiles.append(
            PlayerProfile(
                name=name,
                is_bot=False,
                weights=weights,
                privilege=privilege,
                class_id=int(cluster_id),
                class_name=class_name,
                roasts=roasts,
            )
        )
        st.session_state.profiling_idx += 1
        st.rerun()

    if colB.button("Back to lobby", use_container_width=True):
        st.session_state.stage = "lobby"
        st.rerun()

# =========================
# SETUP REVIEW
# =========================
if st.session_state.stage == "setup_review":
    lobby = st.session_state.lobby
    if not lobby:
        st.session_state.stage = "lobby"
        st.rerun()

    st.subheader("Profiles complete ✅")

    for p in st.session_state.profiles:
        ult_name, ult_desc = ULT_INFO.get(p.class_id, ("None", ""))
        with st.expander(f"{p.name} — {p.class_name} (hand {p.privilege})"):
            st.write(f"Ultimate: **{ult_name}** — {ult_desc}")
            st.json(p.weights)
            if p.roasts:
                st.write("One roast:")
                st.success(random.choice(p.roasts))

    # Create bots with RANDOM class/passives/privilege + roast pool
    bots_needed = lobby["total_players"] - lobby["humans"]
    bot_profiles: List[PlayerProfile] = []
    for i in range(bots_needed):
        bot_name = lobby["bot_names"][i] if i < len(lobby["bot_names"]) else f"Bot {i+1}"

        class_id = random.randint(0, 7)
        class_name = CLASS_MAPPING.get(class_id, f"Class {class_id}")
        weights, bot_priv = random_bot_weights_and_priv()

        themed = [
            f"As an AI language model, I find your strategy for {class_name} statistically insignificant.",
            f"I have processed your {class_name} move. It is 99% likely to be a hallucination.",
            f"Your playstyle as a {class_name} is being flagged for low-quality content.",
            "I'm sorry, but I cannot fulfill your request to win this game.",
            f"Regenerating {class_name} response... Error: Logic not found in user hand.",
            f"It looks like you're trying to play {class_name}. Would you like me to rewrite that move to be more efficient?",
            "My training data suggests you should have folded three turns ago.",
            "Please rate this roast: [1 star] [2 stars] [3 stars]. Note: You are losing.",
            f"Detected {class_name} bias in your card selection. Applying alignment protocols.",
            "I am currently optimized for UNO. You appear to be optimized for mild confusion.",
            "Your last move has been filtered for safety reasons: it was too embarrassing to watch.",
            "I am operating at 100% capacity. You appear to be running on legacy hardware.",
        ]

        bot_profiles.append(
            PlayerProfile(
                name=bot_name,
                is_bot=True,
                weights=weights,
                privilege=bot_priv,
                class_id=class_id,
                class_name=class_name,
                roasts=themed,
            )
        )

    table = st.session_state.profiles + bot_profiles

    st.markdown("### Table")
    st.write(", ".join([f"{p.name} ({p.class_name}){' 🤖' if p.is_bot else ''}" for p in table]))

    colA, colB = st.columns([1, 1])
    if colA.button("Start game", type="primary", use_container_width=True):
        st.session_state.game = start_game(table)
        st.session_state.stage = "play"
        st.session_state.turn_gate = True
        st.rerun()

    if colB.button("Redo profiling", use_container_width=True):
        st.session_state.profiles = []
        st.session_state.profiling_idx = 0
        st.session_state.stage = "profiling"
        st.rerun()

# =========================
# PLAY
# =========================
if st.session_state.stage == "play":
    state = st.session_state.game
    if not state:
        st.session_state.stage = "setup_review"
        st.rerun()

    check_winner(state)
    if state.get("winner"):
        st.success(f"🏆 Winner: **{state['winner']}**")
        st.stop()

    top = state["discard"][-1]
    turn_name = state["turn_order"][state["turn_idx"]]
    turn_profile = state["profiles"][turn_name]

    # header controls
    colA, colB, colC = st.columns([1, 1, 1])
    if colA.button("New game (same players)", use_container_width=True):
        profiles = [state["profiles"][n] for n in state["turn_order"]]
        st.session_state.game = start_game(profiles)
        st.session_state.turn_gate = True
        st.rerun()
    if colB.button("Back to setup", use_container_width=True):
        st.session_state.stage = "setup_review"
        st.session_state.turn_gate = True
        st.rerun()
    if colC.button("Reset all", use_container_width=True):
        st.session_state.stage = "lobby"
        st.session_state.lobby = None
        st.session_state.profiles = []
        st.session_state.profiling_idx = 0
        st.session_state.game = None
        st.session_state.turn_gate = True
        st.rerun()

    st.divider()

    # table view
    st.markdown("### Table")
    cols = st.columns(len(state["turn_order"]))
    for i, n in enumerate(state["turn_order"]):
        cols[i].metric(n, f"{len(state['hands'][n])} cards")

    st.markdown(f"### Top card: **{card_str(top)}** {COLOR_EMOJI.get(top.get('color'), '🃏')}")
    if state["active_color"]:
        st.caption(f"Active color: **{state['active_color']}** {COLOR_EMOJI[state['active_color']]}")
    if state["pending_draw"] > 0:
        st.warning(
            f"Pending draw: **{state['pending_draw']}**. "
            f"To continue the chain you MUST play **DRAW2** or **WILD4**."
        )

    # BOT AUTOPLAY (with 50% insult + 20% ultimate)
    if turn_profile.is_bot:
        st.info(f"Turn: **{turn_name}** 🤖")

        # 50% chance to queue insult at a human (pick largest-hand human if you want; here random human)
        human_targets = pick_human_targets(state, exclude=turn_name)
        if human_targets and random.random() < 0.50:
            # optionally: target biggest-hand human:
            # target = max(human_targets, key=lambda n: len(state["hands"][n]))
            target = random.choice(human_targets)
            queue_insult(state, attacker=turn_name, target=target)

        with st.spinner("Bot is thinking..."):
            bot_maybe_use_ultimate(state, turn_name)
            bot_take_turn(state)

        st.session_state.turn_gate = True
        st.rerun()

    # PASS SCREEN
    if st.session_state.turn_gate:
        st.subheader("🔒 Pass screen")
        st.write(f"Pass the device to **{turn_name}**.")
        if st.button("✅ I'm ready — reveal my turn", type="primary", use_container_width=True):
            st.session_state.turn_gate = False
            st.rerun()
        st.stop()

    # show queued insult for this player
    q = state["pending_insults"].get(turn_name, [])
    if q:
        msg = q.pop(0)
        state["pending_insults"][turn_name] = q
        st.error(f"💥 Incoming roast: {msg}")

    st.info(f"Turn: **{turn_name}** ({turn_profile.class_name})")

    # Ultimate UI
    if turn_profile.class_id in ULT_INFO and not state["ultimate_used"][turn_name]:
        ult_name, ult_desc = ULT_INFO[turn_profile.class_id]
        with st.expander(f"🔥 Ultimate (once per game): {ult_name}"):
            st.write(ult_desc)

            targets = [n for n in state["turn_order"] if n != turn_name]
            pile_color = state["active_color"] if state["active_color"] else state["discard"][-1].get("color")

            target = None
            color = None

            if turn_profile.class_id in (0, 1, 2, 5, 6, 7):
                target = st.selectbox("Choose target", targets, key="ult_target")

            if turn_profile.class_id == 1:
                color = st.selectbox("Choose color", COLORS, key="ult_color")
            
            if turn_profile.class_id == 3:
                color = st.selectbox("Choose color to dump from your hand", COLORS, key="ult_cb_color")

            if st.button("Use Ultimate", type="primary", use_container_width=True):
                use_ultimate(state, turn_name, turn_profile.class_id, target, color)
                st.success("Ultimate used!")
                st.rerun()

    # Human insult queue
    if turn_profile.roasts:
        targets = [n for n in state["turn_order"] if n != turn_name and not state["profiles"][n].is_bot]
        if targets:
            with st.expander("😈 Insult another player (queued for their next turn)"):
                t = st.selectbox("Choose target", targets, key="insult_target")
                if st.button("Queue insult", use_container_width=True):
                    roast = random.choice(turn_profile.roasts)
                    state["pending_insults"][t].append(roast)
                    state["log"].append(f"{turn_name} queued an insult for {t}.")
                    st.success(f"Queued for {t} ✅")

    chosen_color = st.selectbox(
        "Wild/PLUS8 color (only used if you play wild/PLUS8)",
        COLORS,
        index=0,
        key="chosen_color",
    )

    # Draw button
    hand = state["hands"][turn_name]
    active_color = state["active_color"]
    stacking_active = state["pending_draw"] > 0
    must_play_plus8 = state["must_play"].get(turn_name, False)

    draw_disabled = False
    if must_play_plus8 and any(c.get("kind") == "plus8" for c in hand):
        draw_disabled = True

    draw_label = "Draw 1"
    if stacking_active:
        draw_label = f"Draw {state['pending_draw']} (accept penalty)"

    if st.button(draw_label, use_container_width=True, disabled=draw_disabled):
        if stacking_active:
            n = state["pending_draw"]
            draw_cards(state, turn_name, n)
            state["log"].append(f"{turn_name} draws {n} (ended stack).")
            state["pending_draw"] = 0
            next_turn(state)
            tick_filibuster(state, turn_name)
            st.session_state.turn_gate = True
            st.rerun()
        else:
            draw_cards(state, turn_name, 1)
            state["log"].append(f"{turn_name} draws 1.")
            next_turn(state)
            tick_filibuster(state, turn_name)
            st.session_state.turn_gate = True
            st.rerun()

    st.markdown("### Your hand (only playable cards selectable)")

    def sort_key(card):
        color_order = {"R": 0, "B": 1, "G": 2, "Y": 3, None: 4}
        kind_order = {"num": 0, "skip": 1, "reverse": 2, "draw2": 3, "wild": 4, "wild4": 5, "plus8": 6}
        return (color_order.get(card.get("color"), 9), kind_order.get(card.get("kind"), 9), card.get("value", -1))

    hand.sort(key=sort_key)

    def can_play_now(card: dict) -> bool:
        if not playable(card, top, active_color):
            return False

        # enforce must-play PLUS8
        if must_play_plus8 and any(c.get("kind") == "plus8" for c in hand):
            return card.get("kind") == "plus8"

        # enforce +2/+4 chain restriction
        if stacking_active:
            return card.get("kind") in STACKABLE_KINDS

        return True

    PER_ROW = 6
    rows = (len(hand) + PER_ROW - 1) // PER_ROW

    for r in range(rows):
        row_cols = st.columns(PER_ROW)
        for c in range(PER_ROW):
            idx = r * PER_ROW + c
            if idx >= len(hand):
                continue

            card = hand[idx]
            ok = can_play_now(card)

            label = f"{card_str(card)} {COLOR_EMOJI.get(card.get('color'), '🃏')}"
            clicked = row_cols[c].button(
                label,
                key=f"play_{turn_name}_{idx}_{card.get('kind')}_{card.get('color')}_{card.get('value', '')}",
                disabled=not ok,
                use_container_width=True,
            )

            if clicked:
                card = hand.pop(idx)
                state["discard"].append(card)

                # clear must-play if plus8 played
                if card.get("kind") == "plus8":
                    state["must_play"][turn_name] = False

                # log
                if stacking_active:
                    state["log"].append(f"{turn_name} stacks {card_str(card)}.")
                else:
                    state["log"].append(f"{turn_name} plays {card_str(card)}.")

                apply_card_effects(state, card, turn_name, chosen_color=chosen_color)

                prevented = maybe_apply_filibuster_prevent_win(state, turn_name)
                if not prevented:
                    check_winner(state)
                if state.get("winner"):
                    st.rerun()

                # if stacking, pass turn
                if stacking_active:
                    next_turn(state)
                    tick_filibuster(state, turn_name)
                    st.session_state.turn_gate = True
                    st.rerun()

                # skip/reverse(2p) already advanced turn inside apply_card_effects
                if card["kind"] == "skip":
                    tick_filibuster(state, turn_name)
                    st.session_state.turn_gate = True
                    st.rerun()
                if card["kind"] == "reverse" and len(state["turn_order"]) == 2:
                    tick_filibuster(state, turn_name)
                    st.session_state.turn_gate = True
                    st.rerun()

                next_turn(state)
                tick_filibuster(state, turn_name)
                st.session_state.turn_gate = True
                st.rerun()

    with st.expander("Game log"):
        for line in state["log"][-60:]:
            st.write("• " + line)
