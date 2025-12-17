
from archive.icd_transformer import icdTransformer
from archive.icd_transformer_hier import icdTransformerHier
from archive.icd_transformer_lstm import icdTransformerLSTM
from archive.agg_mm import AggMM
from archive.agg_mm_2mod import AggMM as AggMM_2M
from .unis_mmc import uniSMMC
from .ps_fusion import FlexMMF
from .ps_fusion_v2 import FlexMMF as FlexMMFV2
from .mocca import MOCCA
from .mocca_ucl import MOCCA_UCL
from .mocca_2mod import MOCCA as MOCCA_2M
from .mocca_gnn import MOCCA_GNN
from .mocca_note import MOCCA_NOTE
from archive.da_mmc import DAMMC
from archive.da_mmc_v2 import DAMMC as DAMMCV2
from archive.da_mmc_v3 import DAMMC as DAMMCV3
from archive.da_mmc_v4 import DAMMC as DAMMCV4
from .tmc import TMCClassifier
from .att_mm import AttMM
from .att_mm_v2 import AttMMV2
from .mt import MT
from .gmc import GMC
from .cmc import CMC
from .f_comorb import ComorbOnly
from .muse import MUSE
from .concert import ConCert
from .agg_mm import AggMM


all_models = {
    'ICDTransformer': icdTransformer,
    "ICDTransformerHier": icdTransformerHier,
    "ICDTransformerLSTM": icdTransformerLSTM,
    "AggMM": AggMM,
    "AggMM_2M": AggMM_2M,
    "UniS-MMC": uniSMMC,
    "PSFusion": FlexMMF,
    "PSFusionV2": FlexMMFV2,
    "MOCCA": MOCCA,
    "MOCCA_UCL": MOCCA_UCL,
    "MOCCA_2M": MOCCA_2M,
    "MOCCA_GNN": MOCCA_GNN,
    "MOCCA_NOTE": MOCCA_NOTE,
    "DA-MMC": DAMMC,
    "DA-MMC-V2": DAMMCV2,
    "DA-MMC-V3": DAMMCV3,
    "DA-MMC-V4": DAMMCV4,
    "TMC": TMCClassifier,
    "AttMM": AttMM,
    "AttMMV2": AttMMV2,
    "MT": MT,
    "GMC": GMC,
    "CMC": CMC,
    "ComorbOnly": ComorbOnly,
    "MUSE": MUSE,
    "ConCert": ConCert,
}