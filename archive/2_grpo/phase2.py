"""Entry point of the GRPO runs: set STEP and run.

    calibrate        cache the entropy targets of the base model
    smoke            short GRPO run to check that training works
    train            full GRPO run
    eval             evaluate on the P131 test set
    controllability  ask for a city, region or country; does the level move?
    inspect          print trained answers
    diagnose         GranuScore against the matched level
"""

from src.config import REWARD_VARIANT
from src.calibrate import step_calibrate
from src.capability import step_capability
from src.sft import step_sft
from src.dpo import step_dpo
from src.train import step_train
from src.eval import step_eval
from src.generalization import step_generalization
from src.controllability import step_controllability
from src.inspect import step_inspect, step_diagnose

STEP = "dpo"  # capability | sft | dpo | generalization | calibrate | smoke | train | eval | controllability | inspect | diagnose


def main():
    if STEP == "capability":
        step_capability()
    elif STEP == "capability_smoke":
        step_capability(limit=10)
    elif STEP == "sft":
        step_sft()
    elif STEP == "sft_smoke":
        step_sft(smoke_test=True)
    elif STEP == "dpo":
        step_dpo()
    elif STEP == "dpo_smoke":
        step_dpo(smoke_test=True)
    elif STEP == "generalization":
        step_generalization("sft")
    elif STEP == "calibrate":
        step_calibrate()
    elif STEP == "smoke":
        step_train(smoke_test=True)
    elif STEP == "train":
        step_train(smoke_test=False)
    elif STEP == "eval":
        step_eval(REWARD_VARIANT)
    elif STEP == "controllability":
        step_controllability(REWARD_VARIANT)
    elif STEP == "inspect":
        step_inspect()
    elif STEP == "diagnose":
        step_diagnose()
    else:
        raise ValueError(f"Unknown STEP: {STEP!r}")


if __name__ == "__main__":
    main()
