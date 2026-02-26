"""
Ginnie Mae Approved Multiclass Participants.

Reference data for REMIC sponsor/co-sponsor validation, trustee lookup,
and deal structuring. Source: ginniemae.gov Multiclass Resources.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional


@dataclass(frozen=True)
class Participant:
    firm: str
    contact: str
    phone: str


@dataclass(frozen=True)
class ParticipantDirectory:
    sf_sponsors: tuple[Participant, ...]
    reverse_sponsors: tuple[Participant, ...]
    mf_sponsors: tuple[Participant, ...]
    co_sponsors: tuple[Participant, ...]
    accountants: tuple[Participant, ...]
    trust_counsel: tuple[Participant, ...]
    co_trust_counsel: tuple[Participant, ...]
    trustees: tuple[Participant, ...]
    trustee_tax_contacts: tuple[Participant, ...]
    remic_info_agents: tuple[Participant, ...]


# ── Single-Family Sponsors ────────────────────────────────────────────────────

SF_SPONSORS = (
    Participant("Bank of America, N.A.", "Sergio Rodriguez", "646-855-8340"),
    Participant("Barclays Capital Inc.", "Hemanth Nagaraj", "212-526-0387"),
    Participant("BMO Capital Markets Corp.", "Michael Forlenza", "212-702-1981"),
    Participant("BNP Paribas Securities Corp.", "Bassel Kikano", "212-841-2713"),
    Participant("BofA Securities, Inc.", "Sergio Rodriguez", "646-855-8340"),
    Participant("Cantor Fitzgerald & Co.", "Craig Kazmierczak", "212-829-5259"),
    Participant("Citigroup Global Markets Inc.", "Jerome Langella", "212-723-6621"),
    Participant("Goldman Sachs & Co. LLC", "Virginia Xu", "972-368-9147"),
    Participant("Hilltop Securities Inc.", "Jason Lisec", "214-859-6815"),
    Participant("Jefferies, LLC", "Sasan Soleimani", "212-778-8265"),
    Participant("J.P. Morgan Securities LLC", "Samantha Leenas", "212-834-4477"),
    Participant("Mizuho Securities USA LLC", "John Critelli", "212-205-7598"),
    Participant("Morgan Stanley & Co. LLC", "Christopher Noe", "212-761-1471"),
    Participant("MUFG Securities Americas, Inc.", "Scott Mooney", "212-405-7065"),
    Participant("Nomura Securities International, Inc.", "Christina McCarthy", "212-667-2280"),
    Participant("PNC Capital Markets LLC", "Abi Tobun", "212-210-9979"),
    Participant("Santander US Capital Markets LLC", "John Armitage", "646-776-7810"),
    Participant("Stifel, Nicolaus & Company, Inc.", "Sam Rothwell", "212-847-6023"),
    Participant("StoneX Financial Inc.", "Nick Sferrazza", "212-379-5552"),
    Participant("Wells Fargo Bank, N.A.", "Zachary Teachey", "704-410-3331"),
)

# ── Reverse Mortgage (REMIC) Sponsors ─────────────────────────────────────────

REVERSE_SPONSORS = (
    Participant("Bank of America, N.A.", "Sergio Rodriguez", "646-855-8340"),
    Participant("BMO Capital Markets Corp.", "Michael Forlenza", "212-702-1981"),
    Participant("BofA Securities, Inc.", "Sergio Rodriguez", "646-855-8340"),
    Participant("Brean Capital, LLC", "Vanessa Warren", "212-702-6656"),
    Participant("Cantor Fitzgerald & Co.", "Craig Kazmierczak", "212-829-5259"),
    Participant("Citigroup Global Markets Inc.", "Jerome Langella", "212-723-6621"),
    Participant("Nomura Securities International, Inc.", "Christina McCarthy", "212-667-2280"),
    Participant("PNC Capital Markets LLC", "Mohammad Khan", "703-893-5194"),
    Participant("Stifel, Nicolaus & Company, Inc.", "Sam Rothwell", "212-847-6023"),
)

# ── Multifamily Sponsors ──────────────────────────────────────────────────────

MF_SPONSORS = (
    Participant("Bank of America, N.A.", "Sergio Rodriguez", "646-855-8340"),
    Participant("Barclays Capital Inc.", "Hemanth Nagaraj", "212-526-0387"),
    Participant("BMO Capital Markets Corp.", "Michael Forlenza", "212-702-1981"),
    Participant("BofA Securities, Inc.", "Sergio Rodriguez", "646-855-8340"),
    Participant("Cantor Fitzgerald & Co.", "Yahli Becker", "212-829-5259"),
    Participant("Citigroup Global Markets Inc.", "Jerome Langella", "212-723-6621"),
    Participant("Goldman Sachs & Co. LLC", "Virginia Xu", "972-368-9147"),
    Participant("Hilltop Securities Inc.", "Mitchell Simon", "212-699-3306"),
    Participant("Jefferies, LLC", "Sasan Soleimani", "212-778-8265"),
    Participant("J.P. Morgan Securities LLC", "Michael Gottlieb", "212-834-2296"),
    Participant("Mizuho Securities USA LLC", "John Critelli", "212-205-7598"),
    Participant("Morgan Stanley & Co. LLC", "James Palmer", "212-761-3090"),
    Participant("MUFG Securities Americas, Inc.", "Scott Mooney", "212-405-7065"),
    Participant("Nomura Securities International, Inc.", "Christina McCarthy", "212-667-2280"),
    Participant("PNC Capital Markets LLC", "Abi Tobun", "212-210-9979"),
    Participant("Santander US Capital Markets LLC", "John Armitage", "646-776-7810"),
    Participant("Stifel, Nicolaus & Company, Inc.", "Sam Rothwell", "212-847-6023"),
    Participant("StoneX Financial Inc.", "Nick Sferrazza", "212-379-5552"),
    Participant("Wells Fargo Bank, N.A.", "Tyler Barnes", "704-410-3350"),
)

# ── Co-Sponsors ───────────────────────────────────────────────────────────────

CO_SPONSORS = (
    Participant("Academy Securities, Inc.", "Michael Boyd", "646-736-3995"),
    Participant("AmeriVet Securities, Inc.", "Timothy Wilson", "646-809-6933"),
    Participant("Blaylock Van, LLC", "Mark Noble", "212-715-3305"),
    Participant("CastleOak Securities, L.P.", "Patrick Decatalogne", "212-829-5439"),
    Participant("Drexel Hamilton, LLC", "Alex Kim", "646-412-1548"),
    Participant("Great Pacific Securities", "Christopher Vinck", "714-619-3000"),
    Participant("Independence Point Securities LLC", "Meghan Siripurapu", "332-282-2607"),
    Participant("Mischler Financial Group, Inc.", "Dean Chamberlain", "203-276-6646"),
    Participant("Roberts & Ryan Investments Inc.", "Ed D'Alessandro", "646-542-0018"),
    Participant("Samuel A. Ramirez & Company, Inc.", "Eric Kurschus", "212-378-7181"),
)

# ── Accountants ───────────────────────────────────────────────────────────────

ACCOUNTANTS = (
    Participant("Deloitte & Touche LLP", "John Pasvankias", "203-905-2822"),
    Participant("Ernst & Young LLP", "Eli Stern", "212-773-5752"),
    Participant("KPMG, LLP", "Aileen Gregory", "703-286-8189"),
)

# ── Trust Counsel ─────────────────────────────────────────────────────────────

TRUST_COUNSEL = (
    Participant("Cadwalader, Wickersham & Taft, LLP", "Gary Schuler", "704-348-5304"),
    Participant("Cleary, Gottlieb, Steen & Hamilton, LLP", "Mitch Dupler", "202-974-1630"),
    Participant("Dentons US LLP", "Marlo Young", "212-768-5338"),
    Participant("K & L Gates, LLP", "Virginia Stevenson", "704-331-7512"),
    Participant("Morgan, Lewis & Bockius, LLP", "Jeff Johnson", "202-373-6626"),
    Participant("Orrick, Herrington & Sutcliffe, LLP", "Leah Sanzari", "212-506-3798"),
)

# ── Co-Trust Counsel ──────────────────────────────────────────────────────────

CO_TRUST_COUNSEL = (
    Participant("Marcell Solomon & Associates, PC", "Marcell Solomon", "301-486-0700"),
)

# ── Trustees ──────────────────────────────────────────────────────────────────

TRUSTEES = (
    Participant("Bank of New York Mellon Trust Company N.A.", "Hector Herrera", "212-815-4293"),
    Participant("The Bank of New York Mellon", "Hector Herrera", "212-815-4293"),
    Participant("U.S. Bank National Association", "Beth Nally", "617-603-6882"),
)

# ── Trustee Tax Contacts ──────────────────────────────────────────────────────

TRUSTEE_TAX = (
    Participant("The Bank of New York Mellon Trust Company, N.A.", "Darron Huls", "713-483-7792"),
    Participant("U.S. Bank National Association", "Carrie Reynolds", "949-224-7046"),
)

# ── REMIC Information Agent ───────────────────────────────────────────────────

REMIC_INFO_AGENTS = (
    Participant("The Bank of New York Mellon Trust Company, N.A.", "Kathryn Corbett", "315-414-3830"),
)

# ── Assembled directory ───────────────────────────────────────────────────────

DIRECTORY = ParticipantDirectory(
    sf_sponsors=SF_SPONSORS,
    reverse_sponsors=REVERSE_SPONSORS,
    mf_sponsors=MF_SPONSORS,
    co_sponsors=CO_SPONSORS,
    accountants=ACCOUNTANTS,
    trust_counsel=TRUST_COUNSEL,
    co_trust_counsel=CO_TRUST_COUNSEL,
    trustees=TRUSTEES,
    trustee_tax_contacts=TRUSTEE_TAX,
    remic_info_agents=REMIC_INFO_AGENTS,
)

# ── Canonical firm name mapping (for fuzzy matching) ──────────────────────────

_FIRM_ALIASES: dict[str, str] = {
    "bofa": "BofA Securities, Inc.",
    "bofa securities": "BofA Securities, Inc.",
    "bank of america": "Bank of America, N.A.",
    "bac": "Bank of America, N.A.",
    "barclays": "Barclays Capital Inc.",
    "bmo": "BMO Capital Markets Corp.",
    "bnp": "BNP Paribas Securities Corp.",
    "bnp paribas": "BNP Paribas Securities Corp.",
    "cantor": "Cantor Fitzgerald & Co.",
    "cantor fitzgerald": "Cantor Fitzgerald & Co.",
    "citi": "Citigroup Global Markets Inc.",
    "citigroup": "Citigroup Global Markets Inc.",
    "goldman": "Goldman Sachs & Co. LLC",
    "goldman sachs": "Goldman Sachs & Co. LLC",
    "gs": "Goldman Sachs & Co. LLC",
    "hilltop": "Hilltop Securities Inc.",
    "jefferies": "Jefferies, LLC",
    "jpmorgan": "J.P. Morgan Securities LLC",
    "jp morgan": "J.P. Morgan Securities LLC",
    "jpm": "J.P. Morgan Securities LLC",
    "mizuho": "Mizuho Securities USA LLC",
    "morgan stanley": "Morgan Stanley & Co. LLC",
    "ms": "Morgan Stanley & Co. LLC",
    "mufg": "MUFG Securities Americas, Inc.",
    "nomura": "Nomura Securities International, Inc.",
    "pnc": "PNC Capital Markets LLC",
    "santander": "Santander US Capital Markets LLC",
    "stifel": "Stifel, Nicolaus & Company, Inc.",
    "stonex": "StoneX Financial Inc.",
    "wells": "Wells Fargo Bank, N.A.",
    "wells fargo": "Wells Fargo Bank, N.A.",
    "wfc": "Wells Fargo Bank, N.A.",
    "brean": "Brean Capital, LLC",
    "academy": "Academy Securities, Inc.",
    "amerivet": "AmeriVet Securities, Inc.",
    "blaylock": "Blaylock Van, LLC",
    "castleoak": "CastleOak Securities, L.P.",
    "drexel hamilton": "Drexel Hamilton, LLC",
    "great pacific": "Great Pacific Securities",
    "independence point": "Independence Point Securities LLC",
    "mischler": "Mischler Financial Group, Inc.",
    "roberts & ryan": "Roberts & Ryan Investments Inc.",
    "ramirez": "Samuel A. Ramirez & Company, Inc.",
    "bnym": "Bank of New York Mellon Trust Company N.A.",
    "bny mellon": "Bank of New York Mellon Trust Company N.A.",
    "us bank": "U.S. Bank National Association",
    "cadwalader": "Cadwalader, Wickersham & Taft, LLP",
    "cleary": "Cleary, Gottlieb, Steen & Hamilton, LLP",
    "dentons": "Dentons US LLP",
    "kl gates": "K & L Gates, LLP",
    "k&l gates": "K & L Gates, LLP",
    "morgan lewis": "Morgan, Lewis & Bockius, LLP",
    "orrick": "Orrick, Herrington & Sutcliffe, LLP",
}


def _all_sponsor_firms() -> set[str]:
    """All unique firm names across SF, MF, and Reverse sponsor lists."""
    firms: set[str] = set()
    for p in SF_SPONSORS + MF_SPONSORS + REVERSE_SPONSORS:
        firms.add(p.firm)
    return firms


def _all_co_sponsor_firms() -> set[str]:
    return {p.firm for p in CO_SPONSORS}


def resolve_firm(query: str) -> Optional[str]:
    """Resolve a firm name from alias or substring match. Returns canonical name or None."""
    q = query.strip().lower()
    # Exact alias
    if q in _FIRM_ALIASES:
        return _FIRM_ALIASES[q]
    # Substring match against all known firms
    all_firms = set()
    for group in (SF_SPONSORS, MF_SPONSORS, REVERSE_SPONSORS, CO_SPONSORS,
                  ACCOUNTANTS, TRUST_COUNSEL, CO_TRUST_COUNSEL, TRUSTEES,
                  TRUSTEE_TAX, REMIC_INFO_AGENTS):
        for p in group:
            all_firms.add(p.firm)
    for firm in all_firms:
        if q in firm.lower():
            return firm
    return None


def validate_sponsor(dealer: str, program: str = "all") -> dict:
    """
    Validate whether a dealer is an approved Ginnie Mae REMIC sponsor.

    Args:
        dealer: Firm name (exact, alias, or substring)
        program: "sf" (single-family), "mf" (multifamily), "reverse", or "all"

    Returns dict with approved status, roles, and contact info.
    """
    canonical = resolve_firm(dealer)

    sponsor_lists = {
        "sf": ("Single-Family Sponsor", SF_SPONSORS),
        "mf": ("Multifamily Sponsor", MF_SPONSORS),
        "reverse": ("Reverse Mortgage Sponsor", REVERSE_SPONSORS),
    }

    roles: list[dict] = []

    # Check sponsor lists
    for key, (label, participants) in sponsor_lists.items():
        if program not in ("all", key):
            continue
        for p in participants:
            if canonical and p.firm == canonical:
                roles.append({"role": label, "firm": p.firm, "contact": p.contact, "phone": p.phone})
                break

    # Always check co-sponsor
    for p in CO_SPONSORS:
        if canonical and p.firm == canonical:
            roles.append({"role": "Co-Sponsor", "firm": p.firm, "contact": p.contact, "phone": p.phone})
            break

    return {
        "query": dealer,
        "resolved_firm": canonical,
        "approved": len(roles) > 0,
        "roles": roles,
    }


def get_directory_summary() -> dict:
    """Return full directory as serializable dict."""
    def _group(participants: tuple[Participant, ...]) -> list[dict]:
        return [{"firm": p.firm, "contact": p.contact, "phone": p.phone} for p in participants]

    return {
        "single_family_sponsors": _group(SF_SPONSORS),
        "reverse_mortgage_sponsors": _group(REVERSE_SPONSORS),
        "multifamily_sponsors": _group(MF_SPONSORS),
        "co_sponsors": _group(CO_SPONSORS),
        "accountants": _group(ACCOUNTANTS),
        "trust_counsel": _group(TRUST_COUNSEL),
        "co_trust_counsel": _group(CO_TRUST_COUNSEL),
        "trustees": _group(TRUSTEES),
        "trustee_tax_contacts": _group(TRUSTEE_TAX),
        "remic_info_agents": _group(REMIC_INFO_AGENTS),
        "counts": {
            "sf_sponsors": len(SF_SPONSORS),
            "reverse_sponsors": len(REVERSE_SPONSORS),
            "mf_sponsors": len(MF_SPONSORS),
            "co_sponsors": len(CO_SPONSORS),
            "accountants": len(ACCOUNTANTS),
            "trust_counsel": len(TRUST_COUNSEL),
            "trustees": len(TRUSTEES),
        },
    }
