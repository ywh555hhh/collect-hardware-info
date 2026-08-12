import json, pathlib, statistics, sys
root=pathlib.Path(sys.argv[1])
base_dir=root/'baseline_warp32'

def load(d):
    p=d/'benchmark.jsonl'
    return [json.loads(x) for x in p.read_text().splitlines() if x.strip()] if p.exists() else []
rows=[]
base=load(base_dir)
base_g=statistics.geometric_mean([r['time_us'] for r in base]) if base else None
for d in sorted(root.iterdir()):
    if not d.is_dir(): continue
    recs=load(d)
    if not recs: continue
    times=[r['time_us'] for r in recs]
    bws=[r['bandwidth_gbs'] for r in recs if r.get('bandwidth_gbs',0)>0]
    row={'variant':d.name,'rows':len(recs),'time_us_gmean':statistics.geometric_mean(times),'bw_gbs_gmean':statistics.geometric_mean(bws),'bw_gbs_max':max(bws),'speed_vs_baseline':(base_g/statistics.geometric_mean(times) if base_g else 1.0)}
    for group, pred in [('plain_out', lambda r: not r['params'].get('is_fused_cast_back')), ('fused_cast_back', lambda r: r['params'].get('is_fused_cast_back'))]:
        sub=[r for r in recs if pred(r)]
        if sub:
            row[group+'_us_gmean']=statistics.geometric_mean([r['time_us'] for r in sub])
            row[group+'_bw_gmean']=statistics.geometric_mean([r['bandwidth_gbs'] for r in sub])
    rows.append(row)
md=['# per_channel_cast_fused 4090 Sweep - 2026-08-11\n\n']
md.append('| variant | rows | gmean us | gmean BW GB/s | max BW GB/s | speed vs baseline | plain BW | rescale BW |\n')
md.append('|---|---:|---:|---:|---:|---:|---:|---:|\n')
for r in sorted(rows, key=lambda x: x['time_us_gmean']):
    md.append('| {variant} | {rows} | {time_us_gmean:.2f} | {bw_gbs_gmean:.1f} | {bw_gbs_max:.1f} | {speed_vs_baseline:.3f}x | {plain_out_bw_gmean:.1f} | {fused_cast_back_bw_gmean:.1f} |\n'.format(**r))
(root/'SUMMARY.md').write_text(''.join(md))
print((root/'SUMMARY.md').read_text())
