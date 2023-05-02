from importlib.resources import files
import json
import tidytcells as tt

AMINO_ACIDS = (
    "A",
    "C",
    "D",
    "E",
    "F",
    "G",
    "H",
    "I",
    "K",
    "L",
    "M",
    "N",
    "P",
    "Q",
    "R",
    "S",
    "T",
    "V",
    "W",
    "Y",
)

FUNCTIONAL_TRAVS = sorted(
    list(tt.tcr.query(precision="gene", functionality="F", contains="TRAV"))
)
FUNCTIONAL_TRAJS = sorted(
    list(tt.tcr.query(precision="gene", functionality="F", contains="TRAJ"))
)
FUNCTIONAL_TRBVS = sorted(
    list(tt.tcr.query(precision="gene", functionality="F", contains="TRBV"))
)
FUNCTIONAL_TRBJS = sorted(
    list(tt.tcr.query(precision="gene", functionality="F", contains="TRBJ"))
)

with files(__name__).joinpath("v_cdrs.json").open("rb") as f:
    V_CDRS = json.load(f)
