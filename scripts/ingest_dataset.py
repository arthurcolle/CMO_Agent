#!/usr/bin/env python3
"""Stream a Fannie Mae pipe-delimited dataset (CAS/CIRT/HARP) for calibration."""
import sys, zipfile, io, json, time
import numpy as np
from collections import defaultdict

zip_path = sys.argv[1]
csv_name = sys.argv[2]
out_path = sys.argv[3]
label = sys.argv[4] if len(sys.argv) > 4 else csv_name

print(f"Streaming {label} from {zip_path}...", flush=True)

age_prepay = defaultdict(lambda: [0,0])
age_default = defaultdict(lambda: [0,0])
coupon_prepay = defaultdict(lambda: [0,0])
fico_default = defaultdict(lambda: [0,0])
fico_prepay = defaultdict(lambda: [0,0])
ltv_default = defaultdict(lambda: [0,0])
ltv_prepay = defaultdict(lambda: [0,0])
state_default = defaultdict(lambda: [0,0])
vintage_stats = defaultdict(lambda: [0,0,0])
purpose_stats = defaultdict(lambda: [0,0,0])
monthly_prepay = defaultdict(lambda: [0,0])
loss_data = []  # (net_loss, upb) for LGD

n_rows = 0
n_loans = 0
n_prepaid = 0
n_defaulted = 0
n_ages_computed = 0
seen_loans = set()
t0 = time.time()

def compute_age(orig_date_str, report_period_str):
    try:
        om, oy = int(orig_date_str[:2]), int(orig_date_str[2:])
        rm, ry = int(report_period_str[:2]), int(report_period_str[2:])
        if oy < 100: oy += 2000
        if ry < 100: ry += 2000
        return (ry - oy) * 12 + (rm - om)
    except:
        return -1

z = zipfile.ZipFile(zip_path)
with z.open(csv_name) as f:
    for line in io.TextIOWrapper(f, 'utf-8', errors='replace'):
        n_rows += 1
        fields = line.strip().split('|')
        if len(fields) < 44:
            continue

        loan_id = fields[1] if len(fields) > 1 else ''
        if loan_id and loan_id not in seen_loans:
            seen_loans.add(loan_id)
            n_loans += 1

        # Age
        age = -1
        age_str = fields[15] if len(fields) > 15 else ''
        if age_str.strip():
            try: age = int(age_str)
            except: pass
        if age < 0:
            orig_date = fields[13] if len(fields) > 13 else ''
            report_period = fields[2] if len(fields) > 2 else ''
            if orig_date and report_period:
                age = compute_age(orig_date, report_period)
                if age >= 0:
                    n_ages_computed += 1

        # Attributes
        try: rate = round(float(fields[7]) * 2) / 2 if fields[7].strip() else 0
        except: rate = 0
        try: fico = int(fields[23]) if fields[23].strip() else 0
        except: fico = 0
        try: ltv = int(fields[19]) if fields[19].strip() else 0
        except: ltv = 0
        state = fields[30].strip() if len(fields) > 30 else ''
        purpose = fields[26].strip() if len(fields) > 26 else ''
        orig_date = fields[13] if len(fields) > 13 else ''
        report_period = fields[2] if len(fields) > 2 else ''

        try: current_upb = float(fields[10]) if fields[10].strip() else 0
        except: current_upb = 0
        try: net_loss = float(fields[48]) if len(fields) > 48 and fields[48].strip() else 0
        except: net_loss = 0

        try:
            vy = int(orig_date[2:]) if orig_date else 0
            if vy < 100: vy += 2000
        except: vy = 0

        zbc = fields[43].strip() if len(fields) > 43 else ''
        is_prepaid = zbc == '01'
        is_defaulted = zbc in ('03', '09', '06', '15')

        if is_prepaid: n_prepaid += 1
        if is_defaulted:
            n_defaulted += 1
            if current_upb > 0 and net_loss != 0:
                loss_data.append((abs(net_loss), current_upb))

        # Age stats
        if age >= 0 and age < 360:
            age_prepay[age][1] += 1
            if is_prepaid: age_prepay[age][0] += 1
            age_default[age][1] += 1
            if is_defaulted: age_default[age][0] += 1

        if rate > 0:
            coupon_prepay[rate][1] += 1
            if is_prepaid: coupon_prepay[rate][0] += 1

        if fico >= 300:
            if fico < 620: fb = '<620'
            elif fico < 660: fb = '620-659'
            elif fico < 700: fb = '660-699'
            elif fico < 740: fb = '700-739'
            elif fico < 780: fb = '740-779'
            else: fb = '780+'
            fico_default[fb][1] += 1
            if is_defaulted: fico_default[fb][0] += 1
            fico_prepay[fb][1] += 1
            if is_prepaid: fico_prepay[fb][0] += 1

        if ltv > 0:
            if ltv <= 60: lb = '<=60'
            elif ltv <= 70: lb = '60-70'
            elif ltv <= 80: lb = '70-80'
            elif ltv <= 90: lb = '80-90'
            elif ltv <= 95: lb = '90-95'
            else: lb = '95+'
            ltv_default[lb][1] += 1
            if is_defaulted: ltv_default[lb][0] += 1

        if state:
            state_default[state][1] += 1
            if is_defaulted: state_default[state][0] += 1

        if vy >= 2005:
            vintage_stats[vy][2] += 1
            if is_defaulted: vintage_stats[vy][0] += 1
            if is_prepaid: vintage_stats[vy][1] += 1

        if purpose:
            purpose_stats[purpose][2] += 1
            if is_defaulted: purpose_stats[purpose][0] += 1
            if is_prepaid: purpose_stats[purpose][1] += 1

        try:
            rm = int(report_period[:2]) if report_period else 0
            if 1 <= rm <= 12:
                monthly_prepay[rm][1] += 1
                if is_prepaid: monthly_prepay[rm][0] += 1
        except: pass

        if n_rows % 10_000_000 == 0:
            elapsed = (time.time() - t0) / 60
            rate_mpm = n_rows / elapsed / 1e6 if elapsed > 0 else 0
            print(f"  {n_rows/1e6:.0f}M | {elapsed:.1f}min | {rate_mpm:.1f}M/min | loans={n_loans:,} | prepay={n_prepaid:,} default={n_defaulted:,}", flush=True)

elapsed = (time.time() - t0) / 60
print(f"\n{'='*70}", flush=True)
print(f"{label} DONE: {n_rows:,} rows | {n_loans:,} loans | {elapsed:.1f} min", flush=True)
print(f"Prepaid: {n_prepaid:,} | Defaulted: {n_defaulted:,} | Ages computed: {n_ages_computed:,}", flush=True)
print(f"{'='*70}", flush=True)

# Summaries
print(f"\n--- Seasoning ---")
for age in [1,3,6,12,18,24,30,36,48,60,72,84,96,108,120]:
    if age in age_prepay:
        p, t = age_prepay[age]
        cpr = p/t*100 if t>0 else 0
        d, td = age_default.get(age, [0,0])
        cdr = d/td*100 if td>0 else 0
        print(f"  Age {age:3d}mo: CPR={cpr:7.2f}%  CDR={cdr:.4f}%  (n={t:,})")

# Seasonal
print(f"\n--- Seasonal Factors ---")
total_p = sum(v[0] for v in monthly_prepay.values())
seasonal_factors = []
months_names = ['Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','Dec']
for m in range(1, 13):
    p, t = monthly_prepay.get(m, [0,0])
    factor = (p / (total_p/12)) if total_p > 0 else 1.0
    seasonal_factors.append(round(factor, 4))
    print(f"  {months_names[m-1]}: {factor:.3f}  (prepaid={p:,}  active={t:,})")

print(f"\n--- CPR by Coupon ---")
for r in sorted(coupon_prepay.keys()):
    p, t = coupon_prepay[r]
    if t > 50000:
        print(f"  {r:.1f}%: CPR={p/t*100:.2f}%  (n={t:,})")

print(f"\n--- CDR by FICO ---")
for fb in ['<620','620-659','660-699','700-739','740-779','780+']:
    if fb in fico_default:
        d, t = fico_default[fb]
        print(f"  {fb:>8s}: CDR={d/t*100:.4f}%  (n={t:,})")

print(f"\n--- CDR by LTV ---")
for lb in ['<=60','60-70','70-80','80-90','90-95','95+']:
    if lb in ltv_default:
        d, t = ltv_default[lb]
        print(f"  {lb:>5s}: CDR={d/t*100:.4f}%  (n={t:,})")

# States
print(f"\n--- Top 10 Default States ---")
state_cdr = [(st, d/t*100, d, t) for st, (d, t) in state_default.items() if t > 100000 and d > 0]
state_cdr.sort(key=lambda x: -x[1])
for st, cdr, d, t in state_cdr[:10]:
    print(f"  {st}: {cdr:.4f}% CDR  ({d:,} / {t:,})")

print(f"\n--- Vintage ---")
for v in sorted(vintage_stats.keys()):
    d, p, t = vintage_stats[v]
    if t > 10000:
        print(f"  {v}: def={d/t*100:.4f}% pre={p/t*100:.2f}% (n={t:,})")

print(f"\n--- Purpose ---")
for pur in sorted(purpose_stats.keys()):
    d, p, t = purpose_stats[pur]
    if t > 10000:
        print(f"  {pur}: def={d/t*100:.4f}% pre={p/t*100:.2f}% (n={t:,})")

# LGD
if loss_data:
    losses = np.array([l[0] for l in loss_data[:200000]])
    upbs = np.array([l[1] for l in loss_data[:200000]])
    valid = upbs > 0
    if valid.any():
        lgd = losses[valid] / upbs[valid]
        lgd = lgd[(lgd >= 0) & (lgd <= 2.0)]
        if len(lgd) > 0:
            print(f"\n--- LGD ({len(lgd):,} observations) ---")
            print(f"  Mean: {np.mean(lgd)*100:.1f}%")
            print(f"  Median: {np.median(lgd)*100:.1f}%")
            print(f"  25th/75th: {np.percentile(lgd,25)*100:.1f}% / {np.percentile(lgd,75)*100:.1f}%")

# Build seasoning curves — convert monthly SMM/MDR to annualized CPR/CDR
seasoning_cpr = []
seasoning_cdr = []
for i in range(180):
    p, t = age_prepay.get(i, [0,0])
    if t > 0:
        smm = p / t  # Monthly prepayment rate (fraction)
        cpr = (1 - (1 - smm) ** 12) * 100  # Annualized CPR (%)
        seasoning_cpr.append(round(cpr, 4))
    else:
        seasoning_cpr.append(0.0)
    d, t = age_default.get(i, [0,0])
    if t > 0:
        mdr = d / t  # Monthly default rate (fraction)
        cdr = (1 - (1 - mdr) ** 12) * 100  # Annualized CDR (%)
        seasoning_cdr.append(round(cdr, 6))
    else:
        seasoning_cdr.append(0.0)

result = {
    'dataset': label,
    'total_rows': n_rows,
    'unique_loans': n_loans,
    'total_prepaid': n_prepaid,
    'total_defaulted': n_defaulted,
    'elapsed_min': round(elapsed, 1),
    'seasoning_cpr': seasoning_cpr,
    'seasoning_cdr': seasoning_cdr,
    'seasonal_factors': seasonal_factors,
    'cpr_by_coupon': {str(k): round(v[0]/v[1]*100, 4) if v[1]>0 else 0 for k,v in sorted(coupon_prepay.items()) if v[1]>50000},
    'cdr_by_fico': {k: round(v[0]/v[1]*100, 6) if v[1]>0 else 0 for k,v in sorted(fico_default.items())},
    'cdr_by_ltv': {k: round(v[0]/v[1]*100, 6) if v[1]>0 else 0 for k,v in sorted(ltv_default.items())},
    'top_default_states': {st: round(cdr, 4) for st, cdr, _, _ in state_cdr[:15]} if state_cdr else {},
    'vintage': {str(k): {'def': round(v[0]/v[2]*100,4) if v[2]>0 else 0, 'pre': round(v[1]/v[2]*100,2) if v[2]>0 else 0, 'n': v[2]} for k,v in sorted(vintage_stats.items()) if v[2]>10000},
    'purpose': {k: {'def': round(v[0]/v[2]*100,4) if v[2]>0 else 0, 'pre': round(v[1]/v[2]*100,2) if v[2]>0 else 0, 'n': v[2]} for k,v in sorted(purpose_stats.items()) if v[2]>10000},
}

with open(out_path, 'w') as f:
    json.dump(result, f, indent=2)
print(f"\nSaved to {out_path}", flush=True)
