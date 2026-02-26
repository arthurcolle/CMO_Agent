"""
CGC 53601 Compliance Checker for California Local Government Investment Programs.

Implements all authorized investment types, maturity limits, concentration limits,
credit quality requirements, and prohibited instruments per California Government Code
Section 53601, including amendments from:
  - SB 998 (effective Jan 1, 2021): CP 40% for large entities, 10% single-issuer CP+MTN
  - SB 1489 (effective Jan 1, 2023): Settlement-date maturity, 45-day forward limit
  - SB 882 (effective Jan 1, 2024): Agency MBS exempt from private-label restrictions
"""
from __future__ import annotations

from dataclasses import dataclass, field, asdict
from datetime import date, timedelta
from enum import Enum
from typing import Optional


class CGCSubdivision(str, Enum):
    """CGC 53601 subdivision identifiers."""
    A_LOCAL_AGENCY_BONDS = "a"
    B_US_TREASURY = "b"
    C_CA_STATE = "c"
    D_OTHER_STATE = "d"
    E_OTHER_LOCAL_AGENCY = "e"
    F_FEDERAL_AGENCY_GSE = "f"
    G_BANKERS_ACCEPTANCE = "g"
    H_COMMERCIAL_PAPER = "h"
    I_NEGOTIABLE_CD = "i"
    J_REPO = "j_repo"
    J_REVERSE_REPO = "j_reverse"
    K_MEDIUM_TERM_NOTE = "k"
    L_MUTUAL_FUND = "l"
    M_TRUSTEE_INDENTURE = "m"
    N_SECURED_OBLIGATIONS = "n"
    O_AGENCY_MBS = "o_agency"
    O_NON_AGENCY_ABS = "o_non_agency"
    P_JPA_SHARES = "p"
    Q_SUPRANATIONAL = "q"
    R_PUBLIC_BANK = "r"


SUBDIVISION_LABELS = {
    CGCSubdivision.A_LOCAL_AGENCY_BONDS: "Local Agency Bonds (a)",
    CGCSubdivision.B_US_TREASURY: "US Treasury (b)",
    CGCSubdivision.C_CA_STATE: "California State Obligations (c)",
    CGCSubdivision.D_OTHER_STATE: "Other State Bonds (d)",
    CGCSubdivision.E_OTHER_LOCAL_AGENCY: "Other Local Agency (e)",
    CGCSubdivision.F_FEDERAL_AGENCY_GSE: "Federal Agency/GSE (f)",
    CGCSubdivision.G_BANKERS_ACCEPTANCE: "Bankers' Acceptances (g)",
    CGCSubdivision.H_COMMERCIAL_PAPER: "Commercial Paper (h)",
    CGCSubdivision.I_NEGOTIABLE_CD: "Negotiable CD (i)",
    CGCSubdivision.J_REPO: "Repurchase Agreement (j)",
    CGCSubdivision.J_REVERSE_REPO: "Reverse Repo / Securities Lending (j)",
    CGCSubdivision.K_MEDIUM_TERM_NOTE: "Medium-Term Notes (k)",
    CGCSubdivision.L_MUTUAL_FUND: "Mutual Fund (l)",
    CGCSubdivision.M_TRUSTEE_INDENTURE: "Trustee/Indenture (m)",
    CGCSubdivision.N_SECURED_OBLIGATIONS: "Secured Obligations (n)",
    CGCSubdivision.O_AGENCY_MBS: "Agency MBS (o) — SB 882 exempt",
    CGCSubdivision.O_NON_AGENCY_ABS: "Non-Agency ABS/MBS (o)",
    CGCSubdivision.P_JPA_SHARES: "JPA Shares (p)",
    CGCSubdivision.Q_SUPRANATIONAL: "Supranational (q)",
    CGCSubdivision.R_PUBLIC_BANK: "Public Bank (r)",
}


# ── Rating scale mapping (for comparison) ─────────────────────────────────────

RATING_RANK = {
    "AAA": 1, "Aaa": 1,
    "AA+": 2, "Aa1": 2,
    "AA": 3,  "Aa2": 3,
    "AA-": 4, "Aa3": 4,
    "A+": 5,  "A1": 5,
    "A": 6,   "A2": 6,
    "A-": 7,  "A3": 7,
    "BBB+": 8, "Baa1": 8,
    "BBB": 9,  "Baa2": 9,
    "BBB-": 10, "Baa3": 10,
    "BB+": 11, "Ba1": 11,
    "BB": 12,  "Ba2": 12,
    "BB-": 13, "Ba3": 13,
    "B+": 14,  "B1": 14,
    "B": 15,   "B2": 15,
    "B-": 16,  "B3": 16,
    "CCC": 17, "Caa": 17,
    "CC": 18,  "Ca": 18,
    "C": 19,
    "D": 20, "NR": 99,
}

SHORT_TERM_RANK = {
    "A-1+": 1, "P-1": 1, "F1+": 1,
    "A-1": 2,  "F1": 2,
    "A-2": 3,  "P-2": 3, "F2": 3,
    "A-3": 4,  "P-3": 4, "F3": 4,
}


def rating_meets_minimum(actual: str, minimum: str) -> bool:
    """Check if actual rating meets or exceeds minimum. Lower rank number = better."""
    if minimum.startswith("A-1") or minimum.startswith("P-1") or minimum.startswith("F1"):
        a = SHORT_TERM_RANK.get(actual, 99)
        m = SHORT_TERM_RANK.get(minimum, 99)
    else:
        a = RATING_RANK.get(actual, 99)
        m = RATING_RANK.get(minimum, 99)
    return a <= m


# ── Subdivision rules ─────────────────────────────────────────────────────────

@dataclass
class SubdivisionRule:
    """Investment rules for a CGC 53601 subdivision."""
    subdivision: CGCSubdivision
    max_maturity_days: Optional[int]  # None = no statutory limit / per indenture
    max_portfolio_pct: Optional[float]  # None = unlimited
    min_rating: Optional[str]  # None = no rating requirement
    single_issuer_pct: Optional[float]  # None = no single-issuer cap
    single_issuer_note: str = ""
    notes: str = ""


RULES: dict[CGCSubdivision, SubdivisionRule] = {
    CGCSubdivision.A_LOCAL_AGENCY_BONDS: SubdivisionRule(
        CGCSubdivision.A_LOCAL_AGENCY_BONDS, 1825, None, None, None,
    ),
    CGCSubdivision.B_US_TREASURY: SubdivisionRule(
        CGCSubdivision.B_US_TREASURY, 1825, None, None, None,
        notes="Maturity extendable with legislative body approval",
    ),
    CGCSubdivision.C_CA_STATE: SubdivisionRule(
        CGCSubdivision.C_CA_STATE, 1825, None, None, None,
    ),
    CGCSubdivision.D_OTHER_STATE: SubdivisionRule(
        CGCSubdivision.D_OTHER_STATE, 1825, None, None, None,
    ),
    CGCSubdivision.E_OTHER_LOCAL_AGENCY: SubdivisionRule(
        CGCSubdivision.E_OTHER_LOCAL_AGENCY, 1825, None, None, None,
    ),
    CGCSubdivision.F_FEDERAL_AGENCY_GSE: SubdivisionRule(
        CGCSubdivision.F_FEDERAL_AGENCY_GSE, 1825, None, None, None,
        notes="Maturity extendable with legislative body approval",
    ),
    CGCSubdivision.G_BANKERS_ACCEPTANCE: SubdivisionRule(
        CGCSubdivision.G_BANKERS_ACCEPTANCE, 180, 0.40, None, 0.30,
        single_issuer_note="30% of investable moneys per bank",
    ),
    CGCSubdivision.H_COMMERCIAL_PAPER: SubdivisionRule(
        CGCSubdivision.H_COMMERCIAL_PAPER, 270, 0.25, "A-1", 0.10,
        single_issuer_note="Combined CP+MTN per issuer (SB 998)",
        notes="40% for entities >= $100M (SB 998). Issuer must be US-organized, >$500M assets, non-CP debt rated A+",
    ),
    CGCSubdivision.I_NEGOTIABLE_CD: SubdivisionRule(
        CGCSubdivision.I_NEGOTIABLE_CD, 1825, 0.30, None, None,
    ),
    CGCSubdivision.J_REPO: SubdivisionRule(
        CGCSubdivision.J_REPO, 365, None, None, None,
        notes="Collateral >= 102% market value, adjusted quarterly",
    ),
    CGCSubdivision.J_REVERSE_REPO: SubdivisionRule(
        CGCSubdivision.J_REVERSE_REPO, 92, 0.20, None, None,
        notes="Security must be owned >= 30 days prior. 92-day limit unless written min-spread guarantee",
    ),
    CGCSubdivision.K_MEDIUM_TERM_NOTE: SubdivisionRule(
        CGCSubdivision.K_MEDIUM_TERM_NOTE, 1825, 0.30, "A", 0.10,
        single_issuer_note="Combined CP+MTN per issuer (SB 998)",
    ),
    CGCSubdivision.L_MUTUAL_FUND: SubdivisionRule(
        CGCSubdivision.L_MUTUAL_FUND, None, 0.20, "AAA", 0.10,
        single_issuer_note="10% per fund",
        notes="Highest ranking by >= 2 NRSROs, or qualified adviser (SEC-registered, 5yr exp, >$500M AUM)",
    ),
    CGCSubdivision.M_TRUSTEE_INDENTURE: SubdivisionRule(
        CGCSubdivision.M_TRUSTEE_INDENTURE, None, None, None, None,
        notes="Per indenture/trust agreement terms",
    ),
    CGCSubdivision.N_SECURED_OBLIGATIONS: SubdivisionRule(
        CGCSubdivision.N_SECURED_OBLIGATIONS, 1825, None, None, None,
        notes="Secured by CGC 53651-eligible collateral",
    ),
    CGCSubdivision.O_AGENCY_MBS: SubdivisionRule(
        CGCSubdivision.O_AGENCY_MBS, 1825, None, None, None,
        notes="Issued/guaranteed by sub(b) or sub(f) entities (GNMA/FNMA/FHLMC). Exempt per SB 882",
    ),
    CGCSubdivision.O_NON_AGENCY_ABS: SubdivisionRule(
        CGCSubdivision.O_NON_AGENCY_ABS, 1825, 0.20, "AA", None,
    ),
    CGCSubdivision.P_JPA_SHARES: SubdivisionRule(
        CGCSubdivision.P_JPA_SHARES, None, None, None, None,
        notes="JPA must retain qualified adviser (SEC-registered, 5yr exp, >$500M AUM)",
    ),
    CGCSubdivision.Q_SUPRANATIONAL: SubdivisionRule(
        CGCSubdivision.Q_SUPRANATIONAL, 1825, 0.30, "AA", None,
        notes="IBRD, IFC, or IADB only. USD senior unsecured unsubordinated",
    ),
    CGCSubdivision.R_PUBLIC_BANK: SubdivisionRule(
        CGCSubdivision.R_PUBLIC_BANK, 1825, None, None, None,
        notes="Public bank per GC 57600",
    ),
}


# ── Prohibited instruments (CGC 53601.6) ──────────────────────────────────────

PROHIBITED_TYPES = [
    "inverse_floater",
    "range_note",
    "mortgage_derived_io_strip",
    "zero_interest_accrual",
]

PROHIBITED_LABELS = {
    "inverse_floater": "Inverse floaters (CGC 53601.6)",
    "range_note": "Range notes (CGC 53601.6)",
    "mortgage_derived_io_strip": "Mortgage-derived interest-only strips (CGC 53601.6)",
    "zero_interest_accrual": "Zero-interest-accrual securities (CGC 53601.6) — exception for USG-backed during negative rate periods (SB 998, sunsets Jan 2026)",
}


# ── Security holding ──────────────────────────────────────────────────────────

@dataclass
class SecurityHolding:
    """A single security in a portfolio, for compliance checking."""
    name: str
    subdivision: CGCSubdivision
    par_value: float
    market_value: float
    issuer: str = ""
    credit_rating: str = ""
    settlement_date: Optional[date] = None
    maturity_date: Optional[date] = None
    trade_date: Optional[date] = None
    instrument_subtype: str = ""  # e.g. "inverse_floater", "range_note", etc.
    is_agency_backed: bool = False  # For sub(o): True = GNMA/FNMA/FHLMC


@dataclass
class Violation:
    """A single compliance violation."""
    security_name: str
    rule: str
    detail: str
    severity: str = "FAIL"  # FAIL or WARN


@dataclass
class ComplianceResult:
    """Result of a full portfolio compliance check."""
    compliant: bool
    violations: list[Violation] = field(default_factory=list)
    warnings: list[Violation] = field(default_factory=list)
    summary: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "compliant": self.compliant,
            "violations": [asdict(v) for v in self.violations],
            "warnings": [asdict(w) for w in self.warnings],
            "summary": self.summary,
        }


@dataclass
class EntityProfile:
    """Profile of the investing entity, needed for entity-size-dependent rules."""
    name: str = "Local Agency"
    total_investable_assets: float = 0.0
    has_extended_maturity_authority: bool = False
    is_county: bool = False


# ── Core compliance engine ────────────────────────────────────────────────────

def check_security(
    holding: SecurityHolding,
    entity: EntityProfile,
    as_of: Optional[date] = None,
) -> list[Violation]:
    """Check a single security against its subdivision rules."""
    violations: list[Violation] = []
    as_of = as_of or date.today()
    rule = RULES.get(holding.subdivision)
    if rule is None:
        violations.append(Violation(
            holding.name, "UNKNOWN_SUBDIVISION",
            f"Subdivision '{holding.subdivision}' not recognized under CGC 53601",
        ))
        return violations

    # ── Prohibited instrument check ───────────────────────────────────────
    if holding.instrument_subtype in PROHIBITED_TYPES:
        label = PROHIBITED_LABELS.get(holding.instrument_subtype, holding.instrument_subtype)
        violations.append(Violation(
            holding.name, "PROHIBITED_INSTRUMENT",
            f"Prohibited under CGC 53601.6: {label}",
        ))

    # ── Maturity check ────────────────────────────────────────────────────
    if rule.max_maturity_days is not None and holding.maturity_date and holding.settlement_date:
        # SB 1489: maturity measured from settlement date
        days_to_maturity = (holding.maturity_date - holding.settlement_date).days
        if days_to_maturity > rule.max_maturity_days:
            if not entity.has_extended_maturity_authority:
                violations.append(Violation(
                    holding.name, "MATURITY_EXCEEDED",
                    f"Maturity {days_to_maturity}d exceeds {rule.max_maturity_days}d limit for "
                    f"{SUBDIVISION_LABELS[holding.subdivision]}. "
                    f"Legislative body approval required for extension.",
                ))
            else:
                violations.append(Violation(
                    holding.name, "MATURITY_EXTENDED",
                    f"Maturity {days_to_maturity}d exceeds {rule.max_maturity_days}d default — "
                    f"entity has extended authority",
                    severity="WARN",
                ))

    # ── Forward settlement check (SB 1489) ────────────────────────────────
    if holding.trade_date and holding.settlement_date:
        fwd_days = (holding.settlement_date - holding.trade_date).days
        if fwd_days > 45:
            violations.append(Violation(
                holding.name, "FORWARD_SETTLEMENT",
                f"Forward settlement of {fwd_days} days exceeds 45-day limit (SB 1489)",
            ))

    # ── Credit rating check ───────────────────────────────────────────────
    if rule.min_rating and holding.credit_rating:
        if not rating_meets_minimum(holding.credit_rating, rule.min_rating):
            violations.append(Violation(
                holding.name, "CREDIT_RATING",
                f"Rating {holding.credit_rating} below minimum {rule.min_rating} for "
                f"{SUBDIVISION_LABELS[holding.subdivision]}",
            ))
    elif rule.min_rating and not holding.credit_rating:
        violations.append(Violation(
            holding.name, "CREDIT_RATING_MISSING",
            f"No credit rating provided; minimum {rule.min_rating} required for "
            f"{SUBDIVISION_LABELS[holding.subdivision]}",
            severity="WARN",
        ))

    return violations


def check_portfolio(
    holdings: list[SecurityHolding],
    entity: EntityProfile,
    as_of: Optional[date] = None,
) -> ComplianceResult:
    """Full portfolio compliance check against CGC 53601."""
    as_of = as_of or date.today()
    violations: list[Violation] = []
    warnings: list[Violation] = []

    total_mv = sum(h.market_value for h in holdings)
    if total_mv <= 0:
        return ComplianceResult(
            compliant=True,
            summary={"total_market_value": 0, "num_holdings": 0},
        )

    # ── Per-security checks ───────────────────────────────────────────────
    for h in holdings:
        for v in check_security(h, entity, as_of):
            if v.severity == "WARN":
                warnings.append(v)
            else:
                violations.append(v)

    # ── Concentration by subdivision ──────────────────────────────────────
    sub_totals: dict[CGCSubdivision, float] = {}
    for h in holdings:
        sub_totals[h.subdivision] = sub_totals.get(h.subdivision, 0.0) + h.market_value

    concentration_detail: dict[str, dict] = {}
    for sub, mv in sub_totals.items():
        rule = RULES.get(sub)
        if rule is None:
            continue
        pct = mv / total_mv
        max_pct = rule.max_portfolio_pct

        # SB 998: CP 40% for entities >= $100M
        if sub == CGCSubdivision.H_COMMERCIAL_PAPER and entity.total_investable_assets >= 100_000_000:
            max_pct = 0.40

        label = SUBDIVISION_LABELS.get(sub, sub.value)
        concentration_detail[label] = {
            "market_value": round(mv, 2),
            "pct_of_portfolio": round(pct * 100, 2),
            "limit_pct": round(max_pct * 100, 2) if max_pct else None,
        }

        if max_pct is not None and pct > max_pct:
            violations.append(Violation(
                f"[PORTFOLIO]", "CONCENTRATION_LIMIT",
                f"{label} at {pct*100:.1f}% exceeds {max_pct*100:.0f}% limit "
                f"(${mv:,.0f} / ${total_mv:,.0f})",
            ))

    # ── Single-issuer checks ──────────────────────────────────────────────
    issuer_totals: dict[str, dict[CGCSubdivision, float]] = {}
    for h in holdings:
        if h.issuer:
            if h.issuer not in issuer_totals:
                issuer_totals[h.issuer] = {}
            issuer_totals[h.issuer][h.subdivision] = (
                issuer_totals[h.issuer].get(h.subdivision, 0.0) + h.market_value
            )

    for issuer, sub_map in issuer_totals.items():
        # Per-subdivision single-issuer limits
        for sub, mv in sub_map.items():
            rule = RULES.get(sub)
            if rule and rule.single_issuer_pct is not None:
                pct = mv / total_mv
                if pct > rule.single_issuer_pct:
                    violations.append(Violation(
                        f"[ISSUER: {issuer}]", "SINGLE_ISSUER_LIMIT",
                        f"{issuer} {SUBDIVISION_LABELS.get(sub, sub.value)} at {pct*100:.1f}% "
                        f"exceeds {rule.single_issuer_pct*100:.0f}% single-issuer limit",
                    ))

        # SB 998: Combined CP + MTN single-issuer limit = 10%
        cp_mv = sub_map.get(CGCSubdivision.H_COMMERCIAL_PAPER, 0.0)
        mtn_mv = sub_map.get(CGCSubdivision.K_MEDIUM_TERM_NOTE, 0.0)
        combined = cp_mv + mtn_mv
        if combined > 0:
            combined_pct = combined / total_mv
            if combined_pct > 0.10:
                violations.append(Violation(
                    f"[ISSUER: {issuer}]", "SB998_CP_MTN_COMBINED",
                    f"{issuer} combined CP+MTN at {combined_pct*100:.1f}% exceeds "
                    f"10% single-issuer limit (SB 998)",
                ))

    # ── Mutual fund per-fund limit ────────────────────────────────────────
    fund_holdings: dict[str, float] = {}
    for h in holdings:
        if h.subdivision == CGCSubdivision.L_MUTUAL_FUND and h.issuer:
            fund_holdings[h.issuer] = fund_holdings.get(h.issuer, 0.0) + h.market_value
    for fund, mv in fund_holdings.items():
        pct = mv / total_mv
        if pct > 0.10:
            violations.append(Violation(
                f"[FUND: {fund}]", "MUTUAL_FUND_SINGLE_FUND",
                f"Fund '{fund}' at {pct*100:.1f}% exceeds 10% per-fund limit (sub l)",
            ))

    # ── Build summary ─────────────────────────────────────────────────────
    summary = {
        "as_of": as_of.isoformat(),
        "entity": entity.name,
        "total_investable_assets": entity.total_investable_assets,
        "cp_limit_applies": "40%" if entity.total_investable_assets >= 100_000_000 else "25%",
        "extended_maturity_authority": entity.has_extended_maturity_authority,
        "total_market_value": round(total_mv, 2),
        "num_holdings": len(holdings),
        "num_violations": len(violations),
        "num_warnings": len(warnings),
        "concentration": concentration_detail,
    }

    is_compliant = len(violations) == 0
    return ComplianceResult(
        compliant=is_compliant,
        violations=violations,
        warnings=warnings,
        summary=summary,
    )


def classify_cmo_tranche_subdivision(
    agency: str,
    principal_type: str,
    interest_type: str,
) -> tuple[CGCSubdivision, list[str]]:
    """
    Classify a CMO tranche under CGC 53601 and flag prohibited types.

    Returns (subdivision, list_of_prohibition_flags).
    """
    prohibitions: list[str] = []
    agency_upper = agency.upper() if agency else ""

    # Determine if agency-backed
    is_agency = agency_upper in ("GNMA", "FNMA", "FHLMC", "GINNIE", "FANNIE", "FREDDIE",
                                  "GOVERNMENT NATIONAL", "FEDERAL NATIONAL",
                                  "FEDERAL HOME LOAN")

    # Check for prohibited types
    pt = principal_type.upper() if principal_type else ""
    it = interest_type.upper() if interest_type else ""

    if it in ("INV", "INVERSE") or "INVERSE" in it:
        prohibitions.append("inverse_floater")
    if "IO" in it and is_agency is False:
        # Mortgage-derived IO strips are prohibited; agency IO may be structured differently
        # but for local gov compliance, flag conservatively
        prohibitions.append("mortgage_derived_io_strip")
    if "RANGE" in pt or "RANGE" in it:
        prohibitions.append("range_note")

    if is_agency:
        return CGCSubdivision.O_AGENCY_MBS, prohibitions
    else:
        return CGCSubdivision.O_NON_AGENCY_ABS, prohibitions


def check_deal_compliance(
    deal_tranches: list[dict],
    entity: EntityProfile,
    existing_holdings: Optional[list[SecurityHolding]] = None,
    as_of: Optional[date] = None,
) -> ComplianceResult:
    """
    Check a proposed CMO deal for CGC 53601 compliance.

    deal_tranches: list of dicts with keys:
        name, par_value, market_value, agency, principal_type, interest_type,
        credit_rating, settlement_date, maturity_date, trade_date, issuer
    existing_holdings: current portfolio (for concentration checks)
    """
    as_of = as_of or date.today()
    holdings = list(existing_holdings or [])

    for t in deal_tranches:
        sub, prohibitions = classify_cmo_tranche_subdivision(
            t.get("agency", ""),
            t.get("principal_type", ""),
            t.get("interest_type", ""),
        )

        settle = t.get("settlement_date")
        if isinstance(settle, str):
            settle = date.fromisoformat(settle)
        mat = t.get("maturity_date")
        if isinstance(mat, str):
            mat = date.fromisoformat(mat)
        trade = t.get("trade_date")
        if isinstance(trade, str):
            trade = date.fromisoformat(trade)

        instrument_subtype = prohibitions[0] if prohibitions else ""

        holdings.append(SecurityHolding(
            name=t.get("name", "Unknown"),
            subdivision=sub,
            par_value=t.get("par_value", 0),
            market_value=t.get("market_value", t.get("par_value", 0)),
            issuer=t.get("issuer", t.get("agency", "")),
            credit_rating=t.get("credit_rating", ""),
            settlement_date=settle,
            maturity_date=mat,
            trade_date=trade,
            instrument_subtype=instrument_subtype,
            is_agency_backed=(sub == CGCSubdivision.O_AGENCY_MBS),
        ))

    return check_portfolio(holdings, entity, as_of)


# ── Quick single-security check ───────────────────────────────────────────────

def quick_check(
    subdivision: str,
    par_value: float = 0,
    credit_rating: str = "",
    maturity_days: Optional[int] = None,
    settlement_date: Optional[str] = None,
    maturity_date: Optional[str] = None,
    trade_date: Optional[str] = None,
    entity_assets: float = 0,
    instrument_subtype: str = "",
) -> dict:
    """
    Quick compliance check for a single security.
    Returns dict with rule details, violations, and pass/fail.
    """
    try:
        sub = CGCSubdivision(subdivision)
    except ValueError:
        # Try matching by prefix
        sub = None
        for s in CGCSubdivision:
            if subdivision.lower().startswith(s.value.split("_")[0]):
                sub = s
                break
        if sub is None:
            return {"error": f"Unknown subdivision: {subdivision}. Use one of: {[s.value for s in CGCSubdivision]}"}

    rule = RULES[sub]
    entity = EntityProfile(total_investable_assets=entity_assets)

    settle = date.fromisoformat(settlement_date) if settlement_date else None
    mat = date.fromisoformat(maturity_date) if maturity_date else None
    trd = date.fromisoformat(trade_date) if trade_date else None

    # If maturity specified in days, compute maturity_date from settlement
    if maturity_days is not None and settle and mat is None:
        mat = settle + timedelta(days=maturity_days)

    holding = SecurityHolding(
        name="test_security",
        subdivision=sub,
        par_value=par_value,
        market_value=par_value,
        credit_rating=credit_rating,
        settlement_date=settle,
        maturity_date=mat,
        trade_date=trd,
        instrument_subtype=instrument_subtype,
    )

    violations = check_security(holding, entity)

    max_pct = rule.max_portfolio_pct
    if sub == CGCSubdivision.H_COMMERCIAL_PAPER and entity_assets >= 100_000_000:
        max_pct = 0.40

    return {
        "subdivision": sub.value,
        "label": SUBDIVISION_LABELS[sub],
        "rule": {
            "max_maturity_days": rule.max_maturity_days,
            "max_portfolio_pct": f"{max_pct*100:.0f}%" if max_pct else "unlimited",
            "min_rating": rule.min_rating or "none",
            "single_issuer_pct": f"{rule.single_issuer_pct*100:.0f}%" if rule.single_issuer_pct else "none",
            "notes": rule.notes,
        },
        "compliant": len(violations) == 0,
        "violations": [asdict(v) for v in violations],
    }


def get_all_rules() -> dict:
    """Return all CGC 53601 rules in a serializable format."""
    result = {}
    for sub, rule in RULES.items():
        max_pct = rule.max_portfolio_pct
        result[SUBDIVISION_LABELS[sub]] = {
            "subdivision": sub.value,
            "max_maturity_days": rule.max_maturity_days,
            "max_maturity_years": round(rule.max_maturity_days / 365.25, 1) if rule.max_maturity_days else None,
            "max_portfolio_pct": f"{max_pct*100:.0f}%" if max_pct else "unlimited",
            "min_rating": rule.min_rating or "none",
            "single_issuer_pct": f"{rule.single_issuer_pct*100:.0f}%" if rule.single_issuer_pct else "none",
            "single_issuer_note": rule.single_issuer_note,
            "notes": rule.notes,
        }
    result["_prohibited_instruments"] = list(PROHIBITED_LABELS.values())
    result["_global_rules"] = {
        "default_max_maturity": "5 years (1825 days)",
        "maturity_measured_from": "settlement date (SB 1489)",
        "max_forward_settlement": "45 days (SB 1489)",
        "sb998_cp_40pct_threshold": "$100,000,000",
        "sb998_single_issuer_cp_mtn": "10% combined",
        "sb882_agency_mbs_exempt": "Agency MBS (GNMA/FNMA/FHLMC) exempt from sub(o)(2) restrictions",
    }
    return result
