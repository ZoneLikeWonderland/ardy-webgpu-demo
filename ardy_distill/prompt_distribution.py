"""Deterministic, control-aware prompt distribution for ARDY distillation.

The text encoders are deliberately external to the motion student.  This
module only defines the finite prompt bank that is encoded once by the
teacher's Llama/LLM2Vec encoder and the student's FLUX.2/Qwen3 encoder.
"""

from __future__ import annotations

import hashlib
import json
import random
from collections import Counter
from dataclasses import asdict, dataclass
from typing import Iterable


DEFAULT_PROMPT_BANK_SIZE = 8192
DEFAULT_PROMPT_BANK_SEED = 20260715

OFFICIAL_ARDY_PROMPTS = (
    "A person is walking.",
    "A person jumps backwards.",
    "A person side steps to the right.",
    "A person is walking backwards.",
    "A person is kicking with their right leg.",
    "A person is standing.",
    "A young lady walks forward elegantly.",
    "A person bows down and then stands upright.",
    "A ballet dancer, performs a forward, turn joining feet, in a repeating loop",
    "a performer gives high bow, with arms to the side, right leg crossed behind the left",
)


@dataclass(frozen=True)
class PromptFamily:
    name: str
    group: str
    sampling_weight: float
    control_profile: str
    speed_min_mps: float
    speed_max_mps: float
    actions: tuple[str, ...]
    manners: tuple[str, ...]
    contexts: tuple[str, ...]


SUBJECTS = (
    "A person",
    "An adult",
    "A performer",
    "A human figure",
    "A character",
    "Someone",
    "A young person",
    "An athlete",
    "A dancer",
    "A trained performer",
    "A relaxed person",
    "An energetic person",
)

FAMILIES = (
    PromptFamily(
        "walk_forward", "locomotion", 0.14, "mobile", 0.35, 1.65,
        (
            "walking forward", "taking steady steps forward", "strolling ahead",
            "walking briskly forward", "moving straight ahead on foot",
            "walking forward with long strides", "walking ahead with short steps",
            "advancing at a walking pace", "walking forward naturally",
            "moving forward with an even gait",
        ),
        (
            "at a relaxed pace", "at a natural pace", "briskly", "slowly and carefully",
            "with confident steps", "with light steps", "with heavy deliberate steps",
            "with an elegant gait", "with an upright posture", "with casual arm swings",
        ),
        (
            "across an open space", "along a straight path", "through a quiet room",
            "while looking ahead", "while keeping good balance", "toward a distant point",
            "over level ground", "in a continuous motion",
        ),
    ),
    PromptFamily(
        "run_forward", "locomotion", 0.07, "fast_mobile", 1.20, 4.00,
        (
            "jogging forward", "running straight ahead", "sprinting forward",
            "running with quick strides", "accelerating into a run", "bounding forward",
            "running at a steady pace", "jogging with compact steps",
            "charging forward", "running forward energetically",
        ),
        (
            "at a moderate pace", "quickly", "at full speed", "with powerful strides",
            "with a light jogging gait", "with strong arm swings", "with rapid short steps",
            "with controlled momentum", "with athletic form", "with rising speed",
        ),
        (
            "across an open space", "along a clear path", "toward a distant point",
            "over level ground", "while looking ahead", "in one continuous burst",
            "while maintaining balance", "through a wide corridor",
        ),
    ),
    PromptFamily(
        "backward", "locomotion", 0.055, "reverse_mobile", 0.25, 1.45,
        (
            "walking backward", "taking careful steps backward", "retreating on foot",
            "jogging backward", "moving backward cautiously", "backpedaling steadily",
            "stepping straight backward", "walking in reverse", "backing away",
            "moving backward with short steps",
        ),
        (
            "slowly", "at a natural pace", "carefully", "with measured steps",
            "while staying balanced", "with quick light steps", "with a guarded posture",
            "without turning around", "with steady rhythm", "with controlled momentum",
        ),
        (
            "along a straight line", "across an open space", "away from a nearby point",
            "over level ground", "while facing forward", "through a clear area",
            "in a continuous motion", "while keeping an upright torso",
        ),
    ),
    PromptFamily(
        "lateral", "locomotion", 0.06, "lateral_mobile", 0.25, 1.70,
        (
            "side stepping to the left", "side stepping to the right", "shuffling left",
            "shuffling right", "moving laterally with crossing steps",
            "taking wide steps to the side", "sliding sideways", "sidestepping quickly",
            "moving left while facing forward", "moving right while facing forward",
        ),
        (
            "at a steady pace", "slowly and carefully", "with quick athletic steps",
            "with knees slightly bent", "while keeping the torso forward",
            "with wide controlled steps", "with small precise steps", "with a light bounce",
            "while staying balanced", "in a smooth rhythm",
        ),
        (
            "across an open space", "along a lateral line", "over level ground",
            "while watching forward", "through a clear area", "in one continuous motion",
            "as if avoiding an obstacle", "while maintaining a stable stance",
        ),
    ),
    PromptFamily(
        "turning", "locomotion", 0.06, "turning", 0.05, 1.35,
        (
            "turning left while walking", "turning right while walking", "making a wide turn",
            "pivoting to face the opposite direction", "walking in a tight circle",
            "changing direction smoothly", "making a half turn", "making a full turn",
            "turning in place", "curving around to the side",
        ),
        (
            "slowly", "smoothly", "with quick precise footwork", "with measured steps",
            "while keeping balance", "with a relaxed posture", "without stopping",
            "with a small turning radius", "with a wide turning radius", "gracefully",
        ),
        (
            "in an open space", "around an imaginary marker", "over level ground",
            "while looking in the new direction", "in one continuous motion",
            "before continuing forward", "while keeping the torso upright",
            "as part of a repeated route",
        ),
    ),
    PromptFamily(
        "curved_route", "locomotion", 0.04, "mobile", 0.35, 2.20,
        (
            "walking along a curved route", "jogging along a gentle arc",
            "following a winding path", "weaving from side to side while moving forward",
            "walking in an S-shaped path", "circling around an obstacle",
            "moving diagonally forward", "following a broad loop", "walking in a spiral",
            "running around a wide bend",
        ),
        (
            "at a natural pace", "smoothly", "with controlled turns", "briskly",
            "while keeping steady momentum", "with light footwork", "with broad steps",
            "with short careful steps", "while leaning gently into turns", "gracefully",
        ),
        (
            "across an open space", "around several imaginary markers", "over level ground",
            "without coming to a stop", "while looking along the route",
            "in one continuous motion", "while maintaining balance", "through a wide area",
        ),
    ),
    PromptFamily(
        "start_stop", "transition", 0.075, "transition", 0.0, 2.60,
        (
            "starting to walk from a standstill", "slowing from a walk to a stop",
            "accelerating from walking into running", "decelerating from a run into a walk",
            "stopping abruptly after moving forward", "pausing and then continuing to walk",
            "taking a few steps and coming to rest", "starting, stopping, and starting again",
            "changing from standing to jogging", "settling into a stand after walking",
        ),
        (
            "smoothly", "with controlled balance", "gradually", "quickly",
            "with natural weight shifts", "with a brief hesitation", "with decisive movement",
            "without losing balance", "with small adjustment steps", "in a relaxed manner",
        ),
        (
            "on level ground", "in an open space", "while facing forward",
            "at a marked point", "after a short distance", "before changing direction",
            "as part of a repeated sequence", "while keeping the torso upright",
        ),
    ),
    PromptFamily(
        "idle_stance", "stationary", 0.06, "stationary", 0.0, 0.25,
        (
            "standing still", "waiting in a relaxed stance", "standing at attention",
            "shifting weight while standing", "resting in place", "holding a neutral pose",
            "standing with feet apart", "standing with hands relaxed", "breathing while standing",
            "making small idle adjustments",
        ),
        (
            "calmly", "with an upright posture", "with relaxed shoulders", "with subtle motion",
            "while looking forward", "with balanced weight", "with arms at the sides",
            "with a slight sway", "without stepping away", "naturally",
        ),
        (
            "in one spot", "on level ground", "in an open space", "while waiting",
            "for several seconds", "before beginning another action", "while staying alert",
            "as if listening to someone",
        ),
    ),
    PromptFamily(
        "crouch_floor", "low_motion", 0.04, "limited", 0.0, 0.85,
        (
            "crouching down", "rising from a crouch", "walking in a low crouch",
            "kneeling carefully", "standing up from one knee", "crawling forward",
            "ducking low and recovering", "squatting and standing again",
            "moving forward on hands and knees", "lowering toward the floor",
        ),
        (
            "slowly", "with controlled balance", "carefully", "with a low center of gravity",
            "with deliberate motion", "while keeping the torso stable", "smoothly",
            "with small steps", "with athletic control", "without rushing",
        ),
        (
            "on level ground", "in one spot", "across a short distance", "under an obstacle",
            "before returning upright", "while looking forward", "in a continuous motion",
            "as part of an exercise",
        ),
    ),
    PromptFamily(
        "jump_hop", "dynamic", 0.075, "ballistic", 0.0, 2.50,
        (
            "jumping straight up", "jumping forward", "jumping backward", "hopping on one foot",
            "making repeated small hops", "leaping to the side", "performing a broad jump",
            "landing after a high jump", "bounding forward with two feet", "skipping forward",
        ),
        (
            "energetically", "with a soft landing", "with powerful leg drive", "carefully",
            "with arms helping the motion", "with athletic form", "in quick succession",
            "with controlled balance", "as high as possible", "with light springy motion",
        ),
        (
            "on level ground", "across a short distance", "over an imaginary line",
            "before coming to a stop", "and recovering to an upright stance",
            "in an open space", "as part of a repeated sequence", "while facing forward",
        ),
    ),
    PromptFamily(
        "dance", "expressive", 0.08, "expressive_mobile", 0.0, 1.80,
        (
            "performing a graceful dance", "doing a lively freestyle dance",
            "performing a ballet turn", "dancing with rhythmic side steps",
            "spinning and joining the feet", "performing a looping dance phrase",
            "swaying to an imagined beat", "performing quick footwork",
            "dancing forward with elegant steps", "performing a celebratory dance",
        ),
        (
            "gracefully", "with energetic rhythm", "with flowing arm movements",
            "with precise foot placement", "with a light playful style", "dramatically",
            "with repeated turns", "with expressive whole-body motion", "smoothly",
            "with a strong musical pulse",
        ),
        (
            "across an open floor", "mostly in one place", "in a repeating loop",
            "while facing an audience", "with a final balanced pose", "over a small area",
            "as part of a short routine", "while changing direction several times",
        ),
    ),
    PromptFamily(
        "gesture", "expressive", 0.06, "stationary", 0.0, 0.30,
        (
            "waving with the right hand", "waving with the left hand", "clapping repeatedly",
            "pointing forward", "raising both arms overhead", "giving a thumbs-up gesture",
            "bowing and returning upright", "shaking hands with an imaginary person",
            "gesturing while speaking", "opening both arms in welcome",
        ),
        (
            "cheerfully", "politely", "with broad expressive motion", "subtly",
            "with an upright posture", "with relaxed shoulders", "enthusiastically",
            "slowly and clearly", "with balanced weight", "in a friendly manner",
        ),
        (
            "while standing in place", "while facing forward", "as a greeting",
            "as if addressing an audience", "before relaxing the arms", "for several seconds",
            "with small natural weight shifts", "in an open space",
        ),
    ),
    PromptFamily(
        "combat", "dynamic", 0.05, "limited", 0.0, 1.30,
        (
            "kicking forward with the right leg", "kicking forward with the left leg",
            "throwing a straight punch", "performing a side kick", "taking a guarded fighting stance",
            "ducking and countering with a punch", "performing a short boxing combination",
            "stepping back defensively", "lunging forward with a strike", "practicing martial arts footwork",
        ),
        (
            "with athletic control", "powerfully", "quickly", "with precise technique",
            "while keeping balance", "with a guarded upper body", "with sharp motion",
            "in a controlled practice style", "with a fast recovery", "with deliberate timing",
        ),
        (
            "in an open training area", "against an imaginary opponent", "on level ground",
            "before returning to guard", "as part of a repeated drill", "while facing forward",
            "over a small area", "without losing balance",
        ),
    ),
    PromptFamily(
        "sport", "dynamic", 0.04, "limited", 0.0, 2.20,
        (
            "swinging an imaginary tennis racket", "throwing an imaginary ball",
            "catching an imaginary ball", "taking a basketball jump shot",
            "practicing a golf swing", "serving an imaginary volleyball",
            "doing rapid agility steps", "performing a skating-like stride",
            "winding up for a throw", "celebrating a successful play",
        ),
        (
            "with athletic form", "powerfully", "with precise timing", "quickly",
            "with a full follow-through", "while keeping balance", "with energetic motion",
            "in a controlled practice style", "with light footwork", "confidently",
        ),
        (
            "in an open training area", "against an imaginary target", "on level ground",
            "before returning to a ready stance", "as part of a repeated drill",
            "while facing forward", "over a small area", "with a stable landing",
        ),
    ),
    PromptFamily(
        "everyday", "activity", 0.04, "limited", 0.0, 1.10,
        (
            "reaching down to pick up an object", "placing an object on a high shelf",
            "sitting down on an imaginary chair", "standing up from an imaginary chair",
            "opening a heavy door", "carrying an imaginary box", "looking around while waiting",
            "bending to tie a shoe", "pulling an imaginary rope", "pushing an imaginary cart",
        ),
        (
            "naturally", "carefully", "with both hands", "with controlled balance", "slowly",
            "with realistic weight shifts", "without rushing", "with a slight effort",
            "smoothly", "with a stable stance",
        ),
        (
            "in one spot", "across a short distance", "in an open room", "before standing upright",
            "while looking at the object", "as part of a normal routine", "on level ground",
            "and then relaxing",
        ),
    ),
    PromptFamily(
        "balance_pose", "stationary", 0.03, "stationary", 0.0, 0.20,
        (
            "balancing on the right leg", "balancing on the left leg", "holding a wide lunge pose",
            "standing on tiptoe", "holding a yoga-like balance pose", "stretching both arms upward",
            "leaning sideways and recovering", "holding a deep squat", "standing with arms extended",
            "maintaining a poised finishing pose",
        ),
        (
            "steadily", "with controlled breathing", "with an upright torso", "carefully",
            "with arms helping balance", "with minimal movement", "for several seconds",
            "with focused control", "gracefully", "without stepping away",
        ),
        (
            "in one spot", "on level ground", "while facing forward", "before returning to neutral",
            "as part of a warm-up", "in an open space", "with small natural corrections",
            "as a final pose",
        ),
    ),
)


def _control_modes(profile: str) -> tuple[str, ...]:
    if profile in {"mobile", "fast_mobile", "reverse_mobile", "lateral_mobile"}:
        return ("none", "mouse_sparse", "mouse_dense", "keyboard_velocity", "keyboard_heading")
    if profile == "turning":
        return ("none", "mouse_sparse", "mouse_dense", "keyboard_heading")
    if profile == "transition":
        return ("none", "mouse_sparse", "mouse_dense", "keyboard_velocity", "keyboard_heading")
    if profile == "expressive_mobile":
        return ("none", "mouse_sparse", "mouse_dense", "keyboard_velocity")
    if profile == "ballistic":
        return ("none", "mouse_sparse", "mouse_dense")
    if profile == "limited":
        return ("none", "mouse_sparse", "keyboard_velocity")
    if profile == "stationary":
        return ("none", "mouse_sparse", "keyboard_velocity")
    raise ValueError(f"unknown control profile: {profile}")


def _official_metadata(text: str) -> tuple[str, str, str, float, float]:
    lowered = text.lower()
    if "standing" in lowered or "bow" in lowered or "kick" in lowered:
        family = "idle_stance" if "standing" in lowered else "gesture" if "bow" in lowered else "combat"
    elif "ballet" in lowered or "performer" in lowered:
        family = "dance"
    elif "backward" in lowered or "backwards" in lowered:
        family = "backward" if "walk" in lowered else "jump_hop"
    elif "side step" in lowered:
        family = "lateral"
    else:
        family = "walk_forward"
    spec = next(item for item in FAMILIES if item.name == family)
    return spec.name, spec.group, spec.control_profile, spec.speed_min_mps, spec.speed_max_mps


def _candidate_prompts(spec: PromptFamily, seed: int) -> list[str]:
    candidates = [
        f"{subject} is {action} {manner} {context}."
        for subject in SUBJECTS
        for action in spec.actions
        for manner in spec.manners
        for context in spec.contexts
    ]
    rng = random.Random(seed)
    rng.shuffle(candidates)
    return candidates


def _allocate_counts(total: int) -> dict[str, int]:
    weights = {spec.name: spec.sampling_weight for spec in FAMILIES}
    normalizer = sum(weights.values())
    raw = {name: total * weight / normalizer for name, weight in weights.items()}
    counts = {name: int(value) for name, value in raw.items()}
    remaining = total - sum(counts.values())
    priority = sorted(raw, key=lambda name: raw[name] - counts[name], reverse=True)
    for name in priority[:remaining]:
        counts[name] += 1
    return counts


def build_prompt_bank(
    size: int = DEFAULT_PROMPT_BANK_SIZE,
    seed: int = DEFAULT_PROMPT_BANK_SEED,
) -> list[dict]:
    """Build a unique prompt bank with compatibility metadata.

    Prompt id zero is the true unconditional case.  The ten released-demo
    presets are pinned next, and generated prompts fill the remaining slots.
    """

    minimum = 1 + len(OFFICIAL_ARDY_PROMPTS)
    if size < minimum:
        raise ValueError(f"prompt bank size must be at least {minimum}")
    records: list[dict] = [
        {
            "prompt_id": 0,
            "text": "",
            "family": "unconditional",
            "group": "unconditional",
            "control_profile": "unconditional",
            "speed_min_mps": 0.0,
            "speed_max_mps": 4.0,
            "control_modes": ["none", "mouse_sparse", "mouse_dense", "keyboard_velocity", "keyboard_heading"],
            "source": "unconditional",
        }
    ]
    seen = {""}
    for text in OFFICIAL_ARDY_PROMPTS:
        family, group, profile, speed_min, speed_max = _official_metadata(text)
        records.append(
            {
                "prompt_id": len(records),
                "text": text,
                "family": family,
                "group": group,
                "control_profile": profile,
                "speed_min_mps": speed_min,
                "speed_max_mps": speed_max,
                "control_modes": list(_control_modes(profile)),
                "source": "official_ardy_preset",
            }
        )
        seen.add(text)

    generated_total = size - len(records)
    counts = _allocate_counts(generated_total)
    for family_index, spec in enumerate(FAMILIES):
        accepted = 0
        for text in _candidate_prompts(spec, seed + 10_007 * family_index):
            if text in seen:
                continue
            records.append(
                {
                    "prompt_id": len(records),
                    "text": text,
                    "family": spec.name,
                    "group": spec.group,
                    "control_profile": spec.control_profile,
                    "speed_min_mps": spec.speed_min_mps,
                    "speed_max_mps": spec.speed_max_mps,
                    "control_modes": list(_control_modes(spec.control_profile)),
                    "source": "template_v1",
                }
            )
            seen.add(text)
            accepted += 1
            if accepted == counts[spec.name]:
                break
        if accepted != counts[spec.name]:
            raise RuntimeError(
                f"family {spec.name} only produced {accepted}/{counts[spec.name]} unique prompts"
            )
    if len(records) != size:
        raise AssertionError(f"built {len(records)} prompts, expected {size}")
    return records


def prompt_bank_jsonl(records: Iterable[dict]) -> str:
    return "".join(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n" for record in records)


def summarize_prompt_bank(records: list[dict], seed: int) -> dict:
    payload = prompt_bank_jsonl(records).encode("utf-8")
    return {
        "schema": "ardy_prompt_bank_v1",
        "count": len(records),
        "seed": seed,
        "sha256": hashlib.sha256(payload).hexdigest(),
        "family_counts": dict(sorted(Counter(record["family"] for record in records).items())),
        "group_counts": dict(sorted(Counter(record["group"] for record in records).items())),
        "control_profile_counts": dict(
            sorted(Counter(record["control_profile"] for record in records).items())
        ),
        "family_definitions": [asdict(spec) for spec in FAMILIES],
    }

