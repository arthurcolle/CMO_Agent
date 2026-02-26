"""
Client Demand Generation for Multi-Client CMO Structuring.

Models realistic institutional investor demand for CMO tranches.
Each client type has distinct preferences for tranche types, WAL targets,
duration constraints, and price sensitivity - mirroring the real GSE REMIC
dealer desk where 2-5 investors simultaneously bid for different parts
of the capital structure.

Client types based on actual CMO buyer profiles:
- Insurance: Z-bonds, VADM, long duration (match liabilities)
- Bank: floaters, short PAC, NAS (asset-liability mgmt, reg capital)
- Hedge fund: inverse floaters, IO, PO, support (relative value)
- Money manager: PAC, PAC-II, schedule bonds (predictable cash flows)
- Pension fund: VADM, Z-PAC, Z-bonds (ultra-long duration)
"""
import numpy as np
from dataclasses import dataclass, field
from enum import IntEnum
from .cmo_structure import PrincipalType, InterestType, TrancheSpec


class ClientType(IntEnum):
    INSURANCE = 0
    BANK = 1
    HEDGE_FUND = 2
    MONEY_MANAGER = 3
    PENSION_FUND = 4


class DifficultyLevel(IntEnum):
    EASY = 0      # 1-2 aligned clients
    MEDIUM = 1    # 2-3 moderate conflicts
    HARD = 2      # 3-4 competing demands
    EXPERT = 3    # 4-5 adversarial demands


@dataclass
class ClientDemand:
    """Represents a single client's demand for CMO tranches."""
    client_type: ClientType
    notional: float  # Target notional in dollars

    # Preference scores 0-10 for each principal type
    principal_prefs: dict[str, float] = field(default_factory=dict)
    # Preference scores 0-10 for each interest type
    interest_prefs: dict[str, float] = field(default_factory=dict)

    # WAL constraints (years)
    wal_min: float = 0.0
    wal_max: float = 30.0

    # Duration constraints (years)
    dur_min: float = 0.0
    dur_max: float = 25.0

    # Acceptance threshold: minimum match score to accept a tranche
    acceptance_threshold: float = 0.5

    # Price sensitivity: how much they discount for imperfect match (0=insensitive, 1=very)
    price_sensitivity: float = 0.5

    # Whether this client is active in this scenario
    active: bool = True

    def to_vector(self) -> np.ndarray:
        """Encode client demand as a 15-dim vector for observation space."""
        # 5-dim one-hot for client type
        type_onehot = np.zeros(5)
        type_onehot[int(self.client_type)] = 1.0

        # Top principal preference (index + score, normalized)
        if self.principal_prefs:
            top_princ = max(self.principal_prefs.items(), key=lambda x: x[1])
            princ_idx = _PRINCIPAL_KEYS.index(top_princ[0]) / len(_PRINCIPAL_KEYS) if top_princ[0] in _PRINCIPAL_KEYS else 0.0
        else:
            princ_idx = 0.0

        # Top interest preference (index + score, normalized)
        if self.interest_prefs:
            top_int = max(self.interest_prefs.items(), key=lambda x: x[1])
            int_idx = _INTEREST_KEYS.index(top_int[0]) / len(_INTEREST_KEYS) if top_int[0] in _INTEREST_KEYS else 0.0
        else:
            int_idx = 0.0

        return np.array([
            *type_onehot,                    # 5 dims
            self.notional / 1e8,             # normalized to $100M
            self.wal_min / 30.0,             # normalized WAL range
            self.wal_max / 30.0,
            self.dur_min / 25.0,             # normalized duration range
            self.dur_max / 25.0,
            princ_idx,                       # top principal type
            int_idx,                         # top interest type
            self.acceptance_threshold,       # 0-1
            self.price_sensitivity,          # 0-1
            1.0 if self.active else 0.0,     # active flag
        ], dtype=np.float32)


# Keys for indexing preferences
_PRINCIPAL_KEYS = [pt.value for pt in PrincipalType]
_INTEREST_KEYS = [it.value for it in InterestType]


def _make_principal_prefs(**overrides) -> dict[str, float]:
    """Create principal preference dict with defaults at 0."""
    prefs = {pt.value: 0.0 for pt in PrincipalType}
    for k, v in overrides.items():
        prefs[k] = v
    return prefs


def _make_interest_prefs(**overrides) -> dict[str, float]:
    """Create interest preference dict with defaults at 0."""
    prefs = {it.value: 0.0 for it in InterestType}
    for k, v in overrides.items():
        prefs[k] = v
    return prefs


def generate_client_demand(
    client_type: ClientType,
    rng: np.random.RandomState,
    collateral_balance: float = 100_000_000,
) -> ClientDemand:
    """Generate realistic demand for a given client type."""

    if client_type == ClientType.INSURANCE:
        return ClientDemand(
            client_type=client_type,
            notional=rng.uniform(0.15, 0.35) * collateral_balance,
            principal_prefs=_make_principal_prefs(
                Z=rng.uniform(8, 10), VADM=rng.uniform(7, 10),
                ZPAC=rng.uniform(6, 9), PAC=rng.uniform(3, 5),
                SEQ=rng.uniform(1, 3),
            ),
            interest_prefs=_make_interest_prefs(
                Z=rng.uniform(8, 10), FIX=rng.uniform(5, 7),
            ),
            wal_min=7.0 + rng.uniform(0, 3),
            wal_max=25.0 + rng.uniform(0, 5),
            dur_min=5.0 + rng.uniform(0, 3),
            dur_max=20.0 + rng.uniform(0, 5),
            acceptance_threshold=rng.uniform(0.35, 0.55),
            price_sensitivity=rng.uniform(0.2, 0.5),
        )

    elif client_type == ClientType.BANK:
        return ClientDemand(
            client_type=client_type,
            notional=rng.uniform(0.15, 0.30) * collateral_balance,
            principal_prefs=_make_principal_prefs(
                SEQ=rng.uniform(5, 8), PAC=rng.uniform(6, 9),
                NAS=rng.uniform(5, 8), SCHED=rng.uniform(3, 6),
            ),
            interest_prefs=_make_interest_prefs(
                FLT=rng.uniform(8, 10), FIX=rng.uniform(4, 7),
            ),
            wal_min=0.5,
            wal_max=5.0 + rng.uniform(0, 2),
            dur_min=0.3,
            dur_max=4.0 + rng.uniform(0, 2),
            acceptance_threshold=rng.uniform(0.40, 0.60),
            price_sensitivity=rng.uniform(0.3, 0.6),
        )

    elif client_type == ClientType.HEDGE_FUND:
        return ClientDemand(
            client_type=client_type,
            notional=rng.uniform(0.10, 0.25) * collateral_balance,
            principal_prefs=_make_principal_prefs(
                SUP=rng.uniform(5, 8), PT=rng.uniform(4, 7),
                SEQ=rng.uniform(2, 5), SPO=rng.uniform(4, 7),
            ),
            interest_prefs=_make_interest_prefs(
                INV=rng.uniform(8, 10), IO=rng.uniform(7, 10),
                PO=rng.uniform(6, 9),
            ),
            wal_min=0.5,
            wal_max=15.0 + rng.uniform(0, 10),
            dur_min=0.0,
            dur_max=20.0,
            acceptance_threshold=rng.uniform(0.25, 0.45),
            price_sensitivity=rng.uniform(0.5, 0.9),
        )

    elif client_type == ClientType.MONEY_MANAGER:
        return ClientDemand(
            client_type=client_type,
            notional=rng.uniform(0.20, 0.40) * collateral_balance,
            principal_prefs=_make_principal_prefs(
                PAC=rng.uniform(8, 10), PAC2=rng.uniform(6, 9),
                SCHED=rng.uniform(5, 8), SEQ=rng.uniform(4, 7),
                VADM=rng.uniform(3, 6),
            ),
            interest_prefs=_make_interest_prefs(
                FIX=rng.uniform(8, 10),
            ),
            wal_min=3.0 + rng.uniform(0, 1),
            wal_max=8.0 + rng.uniform(0, 3),
            dur_min=2.0 + rng.uniform(0, 1),
            dur_max=7.0 + rng.uniform(0, 2),
            acceptance_threshold=rng.uniform(0.45, 0.65),
            price_sensitivity=rng.uniform(0.3, 0.6),
        )

    else:  # PENSION_FUND
        return ClientDemand(
            client_type=client_type,
            notional=rng.uniform(0.15, 0.35) * collateral_balance,
            principal_prefs=_make_principal_prefs(
                VADM=rng.uniform(8, 10), ZPAC=rng.uniform(7, 10),
                Z=rng.uniform(7, 9), PAC=rng.uniform(3, 5),
            ),
            interest_prefs=_make_interest_prefs(
                Z=rng.uniform(8, 10), FIX=rng.uniform(5, 7),
            ),
            wal_min=10.0 + rng.uniform(0, 5),
            wal_max=30.0,
            dur_min=8.0 + rng.uniform(0, 4),
            dur_max=25.0,
            acceptance_threshold=rng.uniform(0.35, 0.55),
            price_sensitivity=rng.uniform(0.2, 0.4),
        )


def generate_client_scenario(
    difficulty: DifficultyLevel,
    regime: str,
    rng: np.random.RandomState,
    collateral_balance: float = 100_000_000,
    max_clients: int = 5,
) -> list[ClientDemand]:
    """Generate a mix of clients based on difficulty and market regime.

    Market regime influences client mix:
    - crisis: more hedge funds (distressed opportunities)
    - steep: more banks (NIM) + insurance (duration)
    - flat/inverted: more money managers (stability)
    - easing: more pension funds (lock in duration)
    """
    # Base type weights by regime
    regime_weights = {
        "normal":     [0.20, 0.20, 0.20, 0.20, 0.20],
        "steep":      [0.25, 0.30, 0.10, 0.15, 0.20],
        "flat":       [0.15, 0.15, 0.15, 0.35, 0.20],
        "inverted":   [0.10, 0.20, 0.25, 0.30, 0.15],
        "crisis":     [0.10, 0.10, 0.40, 0.20, 0.20],
        "easing":     [0.25, 0.15, 0.10, 0.20, 0.30],
        "tightening": [0.15, 0.25, 0.20, 0.25, 0.15],
    }
    weights = np.array(regime_weights.get(regime, regime_weights["normal"]))
    weights = weights / weights.sum()

    # Number of clients by difficulty
    n_clients_range = {
        DifficultyLevel.EASY: (1, 2),
        DifficultyLevel.MEDIUM: (2, 3),
        DifficultyLevel.HARD: (3, 4),
        DifficultyLevel.EXPERT: (4, 5),
    }
    lo, hi = n_clients_range[difficulty]
    n_active = rng.randint(lo, hi + 1)
    n_active = min(n_active, max_clients)

    # Sample client types
    client_types = rng.choice(
        list(ClientType), size=n_active, replace=False if n_active <= 5 else True,
        p=weights,
    )

    # For higher difficulty, tighten acceptance thresholds and increase conflicts
    threshold_bump = {
        DifficultyLevel.EASY: -0.10,     # more lenient
        DifficultyLevel.MEDIUM: 0.0,
        DifficultyLevel.HARD: 0.05,      # tighter
        DifficultyLevel.EXPERT: 0.10,    # very tight
    }

    clients = []
    for ct_raw in client_types:
        ct = ClientType(int(ct_raw))
        demand = generate_client_demand(ct, rng, collateral_balance)
        demand.acceptance_threshold = np.clip(
            demand.acceptance_threshold + threshold_bump[difficulty], 0.15, 0.85
        )
        clients.append(demand)

    # Pad to max_clients with inactive clients
    while len(clients) < max_clients:
        inactive = ClientDemand(
            client_type=ClientType.MONEY_MANAGER,
            notional=0.0,
            active=False,
        )
        clients.append(inactive)

    return clients


def compute_match_score(
    tranche: TrancheSpec,
    client: ClientDemand,
    tranche_wal: float = 5.0,
    tranche_duration: float = 4.0,
) -> float:
    """Score how well a tranche matches a client's demand.

    Scoring weights:
    - 30% principal type preference
    - 30% interest type preference
    - 20% WAL fit
    - 20% duration fit

    Returns float in [0, 1].
    """
    if not client.active:
        return 0.0

    # Principal type score (0-1)
    princ_key = tranche.principal_type.value
    princ_score = client.principal_prefs.get(princ_key, 0.0) / 10.0

    # Interest type score (0-1)
    int_key = tranche.interest_type.value
    int_score = client.interest_prefs.get(int_key, 0.0) / 10.0

    # WAL fit (1 if in range, decays outside)
    if client.wal_min <= tranche_wal <= client.wal_max:
        wal_score = 1.0
    else:
        # Decay: lose 0.1 per year outside range
        dist = min(abs(tranche_wal - client.wal_min), abs(tranche_wal - client.wal_max))
        wal_score = max(0.0, 1.0 - dist * 0.1)

    # Duration fit (same logic)
    if client.dur_min <= tranche_duration <= client.dur_max:
        dur_score = 1.0
    else:
        dist = min(abs(tranche_duration - client.dur_min), abs(tranche_duration - client.dur_max))
        dur_score = max(0.0, 1.0 - dist * 0.1)

    # Weighted average
    score = 0.30 * princ_score + 0.30 * int_score + 0.20 * wal_score + 0.20 * dur_score
    return float(np.clip(score, 0.0, 1.0))
