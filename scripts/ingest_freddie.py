#!/usr/bin/env python3
"""Stream Freddie Mac Single-Family Loan-Level data for calibration.

Handles nested zips: year.zip -> quarterly.zip -> origination.txt + performance.txt
Joins origination (FICO, LTV, state) with performance (ZBC, age, loss amounts).
"""
import sys, zipfile, io, json, time
import numpy as np
from collections import defaultdict

import os

out_path = sys.argv[1] if len(sys.argv) > 1 else '/Users/arthurcolle/CMO_Agent/freddie_calibration.json'
master_zip = sys.argv[2] if len(sys.argv) > 2 else '/Users/arthurcolle/Downloads/full_set_standard_historical_data.zip'

# Support both: master zip containing year zips, or individual year zips
if os.path.basename(master_zip).startswith('full_set') or os.path.basename(master_zip).startswith('non_std'):
    print(f"Processing master zip: {master_zip} ({os.path.getsize(master_zip)/1e9:.1f}GB)", flush=True)
    z_master = zipfile.ZipFile(master_zip)
    year_zip_names = sorted([n for n in z_master.namelist() if n.endswith('.zip')])
    print(f"  Contains {len(year_zip_names)} year files", flush=True)
    use_master = True
else:
    # Individual year files passed as arguments
    zip_files = [master_zip] + sys.argv[3:]
    z_master = None
    year_zip_names = zip_files
    use_master = False
    print(f"Processing {len(zip_files)} Freddie Mac year files...", flush=True)

# Accumulators
age_prepay = defaultdict(lambda: [0,0])
age_default = defaultdict(lambda: [0,0])
fico_default = defaultdict(lambda: [0,0])
fico_prepay = defaultdict(lambda: [0,0])
ltv_default = defaultdict(lambda: [0,0])
ltv_prepay = defaultdict(lambda: [0,0])
state_default = defaultdict(lambda: [0,0])
vintage_stats = defaultdict(lambda: [0,0,0])  # [defaults, prepays, total]
coupon_prepay = defaultdict(lambda: [0,0])
delinq_stats = defaultdict(int)
monthly_prepay = defaultdict(lambda: [0,0])
loss_amounts = []  # (loss, upb) for LGD

n_rows_total = 0
n_loans_total = 0
n_prepaid_total = 0
n_defaulted_total = 0
t0 = time.time()

def fico_bucket(fico):
    if fico < 620: return '<620'
    elif fico < 660: return '620-659'
    elif fico < 700: return '660-699'
    elif fico < 740: return '700-739'
    elif fico < 780: return '740-779'
    else: return '780+'

def ltv_bucket(ltv):
    if ltv <= 60: return '<=60'
    elif ltv <= 70: return '60-70'
    elif ltv <= 80: return '70-80'
    elif ltv <= 90: return '80-90'
    elif ltv <= 95: return '90-95'
    else: return '95+'

for year_zip_name in year_zip_names:
    year_label = year_zip_name.split('_')[-1].replace('.zip','')
    print(f"\n=== {year_label} ===", flush=True)

    try:
        if use_master:
            # Read year zip from master zip into memory
            year_bytes = z_master.read(year_zip_name)
            z_year = zipfile.ZipFile(io.BytesIO(year_bytes))
        else:
            z_year = zipfile.ZipFile(year_zip_name)
    except Exception as e:
        print(f"  ERROR: {e}", flush=True)
        continue

    quarterly_zips = sorted(z_year.namelist())

    for q_name in quarterly_zips:
        q_label = q_name.replace('.zip','').replace('historical_data_','')
        print(f"  Processing {q_label}...", flush=True)

        try:
            q_bytes = z_year.read(q_name)
            z_q = zipfile.ZipFile(io.BytesIO(q_bytes))
        except Exception as e:
            print(f"    ERROR reading {q_name}: {e}", flush=True)
            continue

        # Find origination and performance files
        orig_file = None
        perf_file = None
        for n in z_q.namelist():
            if '_time_' in n: perf_file = n
            else: orig_file = n

        if not orig_file or not perf_file:
            print(f"    Missing files in {q_name}", flush=True)
            continue

        # Step 1: Load origination data into lookup (small file, ~50MB)
        loan_info = {}  # loan_id -> {fico, ltv, state, ...}
        n_orig = 0
        with z_q.open(orig_file) as f:
            for line in io.TextIOWrapper(f, 'utf-8', errors='replace'):
                fields = line.strip().split('|')
                if len(fields) < 20:
                    continue
                n_orig += 1

                # Freddie origination layout:
                # [0]=FICO [1]=first_pay_date [2]=first_time_buyer [3]=maturity_date
                # [4]=MSA [5]=MI_pct [6]=units [7]=occupancy [8]=CLTV [9]=DTI
                # [10]=orig_UPB [11]=LTV [12]=orig_rate [13]=channel [14]=PPM
                # [15]=amort_type(?) [16]=state [17]=prop_type [18]=zip3 [19]=loan_seq_num
                # ...more fields

                try:
                    loan_id = fields[19].strip() if len(fields) > 19 else ''
                    if not loan_id:
                        # Some versions use different field positions
                        # Try to find the loan sequence number
                        for idx in range(15, min(25, len(fields))):
                            if fields[idx].strip().startswith('F'):
                                loan_id = fields[idx].strip()
                                break

                    fico = int(fields[0]) if fields[0].strip() else 0
                    ltv = int(fields[11]) if fields[11].strip() else 0
                    rate = float(fields[12]) if fields[12].strip() else 0
                    state = fields[16].strip() if len(fields) > 16 else ''
                    first_pay = fields[1].strip() if fields[1].strip() else ''

                    # Vintage year from first payment date
                    vy = 0
                    if first_pay:
                        try:
                            vy = int(first_pay[:4]) if len(first_pay) >= 4 else int(first_pay[2:])
                            if vy < 100: vy += 2000
                        except: pass

                    if loan_id:
                        loan_info[loan_id] = {
                            'fico': fico, 'ltv': ltv, 'rate': rate,
                            'state': state, 'vintage': vy,
                        }
                except Exception:
                    continue

        print(f"    Origination: {n_orig:,} loans loaded", flush=True)
        n_loans_total += n_orig

        # Step 2: Stream performance file
        n_perf = 0
        n_prep_q = 0
        n_def_q = 0

        with z_q.open(perf_file) as f:
            for line in io.TextIOWrapper(f, 'utf-8', errors='replace'):
                fields = line.strip().split('|')
                if len(fields) < 10:
                    continue
                n_perf += 1
                n_rows_total += 1

                # Freddie performance layout:
                # [0]=loan_seq [1]=report_period [2]=current_UPB [3]=delinq_status
                # [4]=loan_age [5]=remaining_months [6]=repurchase_date [7]=mod_flag
                # [8]=zero_bal_code [9]=zero_bal_date [10]=current_rate
                # [11]=current_deferred_UPB [12]=DDLPI [13]=MI_recoveries
                # [14]=net_sale_proceeds [15]=non_MI_recoveries [16]=expenses
                # [17]=legal_costs [18]=maint_costs [19]=taxes_insurance
                # [20]=misc_expenses [21]=actual_loss [22]=mod_cost

                loan_id = fields[0].strip()
                zbc = fields[8].strip() if len(fields) > 8 else ''
                report_period = fields[1].strip() if fields[1].strip() else ''

                # Age
                age = -1
                if fields[4].strip():
                    try: age = int(fields[4])
                    except: pass

                # Delinquency
                delinq = fields[3].strip() if fields[3].strip() else '0'
                delinq_stats[delinq] += 1

                is_prepaid = zbc == '01'
                is_defaulted = zbc in ('02', '03', '09', '15', '16')

                if is_prepaid:
                    n_prepaid_total += 1
                    n_prep_q += 1
                if is_defaulted:
                    n_defaulted_total += 1
                    n_def_q += 1

                    # Actual loss for LGD
                    try:
                        actual_loss = float(fields[21]) if len(fields) > 21 and fields[21].strip() else 0
                        current_upb = float(fields[2]) if fields[2].strip() else 0
                        orig_upb_str = fields[2]
                        if actual_loss != 0:
                            loss_amounts.append((abs(actual_loss), max(current_upb, 1)))
                    except: pass

                # Get loan attributes from origination lookup
                info = loan_info.get(loan_id, {})
                fico = info.get('fico', 0)
                ltv = info.get('ltv', 0)
                state = info.get('state', '')
                rate = info.get('rate', 0)
                vy = info.get('vintage', 0)

                # Age stats
                if age >= 0 and age < 360:
                    age_prepay[age][1] += 1
                    if is_prepaid: age_prepay[age][0] += 1
                    age_default[age][1] += 1
                    if is_defaulted: age_default[age][0] += 1

                # FICO
                if fico >= 300:
                    fb = fico_bucket(fico)
                    fico_default[fb][1] += 1
                    if is_defaulted: fico_default[fb][0] += 1
                    fico_prepay[fb][1] += 1
                    if is_prepaid: fico_prepay[fb][0] += 1

                # LTV
                if ltv > 0:
                    lb = ltv_bucket(ltv)
                    ltv_default[lb][1] += 1
                    if is_defaulted: ltv_default[lb][0] += 1

                # State
                if state:
                    state_default[state][1] += 1
                    if is_defaulted: state_default[state][0] += 1

                # Coupon
                if rate > 0:
                    rb = round(rate * 2) / 2
                    coupon_prepay[rb][1] += 1
                    if is_prepaid: coupon_prepay[rb][0] += 1

                # Vintage
                if vy >= 1999:
                    vintage_stats[vy][2] += 1
                    if is_defaulted: vintage_stats[vy][0] += 1
                    if is_prepaid: vintage_stats[vy][1] += 1

                # Seasonal
                try:
                    rm = int(report_period[4:6]) if len(report_period) >= 6 else 0
                    if 1 <= rm <= 12:
                        monthly_prepay[rm][1] += 1
                        if is_prepaid: monthly_prepay[rm][0] += 1
                except: pass

        print(f"    Performance: {n_perf:,} rows | prepaid={n_prep_q:,} default={n_def_q:,}", flush=True)

        # Free memory
        del z_q, q_bytes

    if not use_master:
        z_year.close()

elapsed = (time.time() - t0) / 60
print(f"\n{'='*70}", flush=True)
print(f"FREDDIE MAC DONE: {n_rows_total:,} rows | {n_loans_total:,} loans | {elapsed:.1f} min", flush=True)
print(f"Prepaid: {n_prepaid_total:,} | Defaulted: {n_defaulted_total:,}", flush=True)
print(f"{'='*70}", flush=True)

# Print summaries
print(f"\n--- Seasoning ---")
for age in [1,3,6,12,18,24,30,36,48,60,72,84,96,108,120]:
    if age in age_prepay:
        p, t = age_prepay[age]
        cpr = p/t*100 if t>0 else 0
        d, td = age_default.get(age, [0,0])
        cdr = d/td*100 if td>0 else 0
        print(f"  Age {age:3d}mo: CPR={cpr:7.2f}%  CDR={cdr:.4f}%  (n={t:,})")

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

print(f"\n--- Top Default States ---")
state_cdr = [(st, d/t*100, d, t) for st, (d, t) in state_default.items() if t > 50000 and d > 0]
state_cdr.sort(key=lambda x: -x[1])
for st, cdr, d, t in state_cdr[:15]:
    print(f"  {st}: {cdr:.4f}% CDR  ({d:,} / {t:,})")

print(f"\n--- Vintage ---")
for v in sorted(vintage_stats.keys()):
    d, p, t = vintage_stats[v]
    if t > 1000:
        print(f"  {v}: def={d/t*100:.4f}% pre={p/t*100:.2f}% (n={t:,})")

print(f"\n--- Delinquency Distribution ---")
for status in sorted(delinq_stats.keys()):
    c = delinq_stats[status]
    if c > 1000:
        print(f"  {status}: {c:,} ({c/n_rows_total*100:.2f}%)")

# Seasonal
print(f"\n--- Seasonal ---")
total_p = sum(v[0] for v in monthly_prepay.values())
seasonal_factors = []
months_names = ['Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','Dec']
for m in range(1, 13):
    p, t = monthly_prepay.get(m, [0,0])
    factor = (p / (total_p/12)) if total_p > 0 else 1.0
    seasonal_factors.append(round(factor, 4))
    print(f"  {months_names[m-1]}: {factor:.3f}  (prepaid={p:,})")

# LGD
if loss_amounts:
    losses = np.array([l[0] for l in loss_amounts[:500000]])
    upbs = np.array([l[1] for l in loss_amounts[:500000]])
    valid = (upbs > 0) & (losses > 0)
    if valid.any():
        lgd = losses[valid] / upbs[valid]
        lgd = lgd[(lgd >= 0) & (lgd <= 5.0)]
        if len(lgd) > 0:
            print(f"\n--- LGD ({len(lgd):,} observations) ---")
            print(f"  Mean: {np.mean(lgd)*100:.1f}%")
            print(f"  Median: {np.median(lgd)*100:.1f}%")
            print(f"  25th/75th: {np.percentile(lgd,25)*100:.1f}% / {np.percentile(lgd,75)*100:.1f}%")
            print(f"  90th: {np.percentile(lgd,90)*100:.1f}%")

# CPR by coupon
print(f"\n--- CPR by Coupon ---")
for r in sorted(coupon_prepay.keys()):
    p, t = coupon_prepay[r]
    if t > 50000:
        print(f"  {r:.1f}%: CPR={p/t*100:.2f}%  (n={t:,})")

# Save JSON — convert monthly SMM/MDR to annualized CPR/CDR
seasoning_cpr = []
seasoning_cdr = []
for i in range(180):
    p, t = age_prepay.get(i, [0,0])
    if t > 0:
        smm = p / t
        cpr = (1 - (1 - smm) ** 12) * 100
        seasoning_cpr.append(round(cpr, 4))
    else:
        seasoning_cpr.append(0.0)
    d, t = age_default.get(i, [0,0])
    if t > 0:
        mdr = d / t
        cdr = (1 - (1 - mdr) ** 12) * 100
        seasoning_cdr.append(round(cdr, 6))
    else:
        seasoning_cdr.append(0.0)

result = {
    'dataset': 'Freddie Mac Single-Family 1999-2008',
    'total_rows': n_rows_total,
    'unique_loans': n_loans_total,
    'total_prepaid': n_prepaid_total,
    'total_defaulted': n_defaulted_total,
    'elapsed_min': round(elapsed, 1),
    'seasoning_cpr': seasoning_cpr,
    'seasoning_cdr': seasoning_cdr,
    'seasonal_factors': seasonal_factors,
    'cpr_by_coupon': {str(k): round(v[0]/v[1]*100, 4) if v[1]>0 else 0 for k,v in sorted(coupon_prepay.items()) if v[1]>50000},
    'cdr_by_fico': {k: round(v[0]/v[1]*100, 6) if v[1]>0 else 0 for k,v in sorted(fico_default.items())},
    'cdr_by_ltv': {k: round(v[0]/v[1]*100, 6) if v[1]>0 else 0 for k,v in sorted(ltv_default.items())},
    'top_default_states': {st: round(cdr, 4) for st, cdr, _, _ in state_cdr[:20]} if state_cdr else {},
    'vintage': {str(k): {'def': round(v[0]/v[2]*100,4) if v[2]>0 else 0, 'pre': round(v[1]/v[2]*100,2) if v[2]>0 else 0, 'n': v[2]} for k,v in sorted(vintage_stats.items()) if v[2]>1000},
    'delinquency': {k: c for k, c in sorted(delinq_stats.items()) if c > 1000},
    'seasonal_factors': seasonal_factors,
}

# Add LGD if available
if loss_amounts:
    losses = np.array([l[0] for l in loss_amounts[:500000]])
    upbs = np.array([l[1] for l in loss_amounts[:500000]])
    valid = (upbs > 0) & (losses > 0)
    if valid.any():
        lgd = losses[valid] / upbs[valid]
        lgd = lgd[(lgd >= 0) & (lgd <= 5.0)]
        if len(lgd) > 0:
            result['lgd'] = {
                'mean': round(float(np.mean(lgd)), 4),
                'median': round(float(np.median(lgd)), 4),
                'p25': round(float(np.percentile(lgd, 25)), 4),
                'p75': round(float(np.percentile(lgd, 75)), 4),
                'p90': round(float(np.percentile(lgd, 90)), 4),
                'n': int(len(lgd)),
            }

with open(out_path, 'w') as f:
    json.dump(result, f, indent=2)
print(f"\nSaved to {out_path}", flush=True)
