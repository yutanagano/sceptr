import json
from pkg_resources import resource_stream
import tidytcells as tt

AMINO_ACIDS = (
    'A','C','D','E','F','G','H','I','K','L',
    'M','N','P','Q','R','S','T','V','W','Y'
)

FUNCTIONAL_TRAVS = sorted(list(tt.tcr.query(precision='gene', functionality='F', contains='TRAV')))
FUNCTIONAL_TRAJS = sorted(list(tt.tcr.query(precision='gene', functionality='F', contains='TRAJ')))
FUNCTIONAL_TRBVS = sorted(list(tt.tcr.query(precision='gene', functionality='F', contains='TRBV')))
FUNCTIONAL_TRBJS = sorted(list(tt.tcr.query(precision='gene', functionality='F', contains='TRBJ')))

with resource_stream(__name__, 'v_cdrs.json') as f:
    V_CDRS = json.load(f)