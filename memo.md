0.25
python - <<'PY'
from adversarial.embedding import adv_emb_attack
adv_emb_attack(
    "data/in",
    encoder="sdxlvae",
    strength=31.875,
    output_path="data/out0.25",
    device=torch.device("cuda:1"),
)
PY

0.1
python - <<'PY'
from adversarial.embedding import adv_emb_attack
adv_emb_attack(
    "data/in",
    encoder="sdxlvae",
    strength=12.75,
    output_path="data/out0.1", 
    device=torch.device("cuda:2"),
)
PY