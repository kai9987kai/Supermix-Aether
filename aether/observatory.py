"""A portable HTML observatory generated solely from recorded study data."""

from html import escape
import json
from pathlib import Path


def render_report(result: dict, path: Path):
    rows, cards = [], []
    aggregate = result['aggregates']
    max_seconds = max(v['mean_train_seconds'] for v in aggregate.values()) or 1
    for protocol, values in aggregate.items():
        label = escape(protocol.replace('_', ' '))
        success = values['mean_success']
        rows.append(f"<tr><th>{label}</th><td>{success:.1%}</td>"
                    f"<td>{values['mean_retention_success']:.1%}</td>"
                    f"<td>{values['parameter_count']:,}</td>"
                    f"<td>{values['mean_train_seconds']:.2f}s</td></tr>")
        cards.append(f'<div class="barrow"><span>{label}</span><div class="track">'
                     f'<div class="bar" style="width:{success * 100:.3f}%"></div></div>'
                     f'<b>{success:.1%}</b></div>')
    effects = []
    for name, value in result['paired_effects'].items():
        lo, hi = value['ci95']
        effects.append(f"<tr><th>wave − {escape(name)}</th><td>{value['mean']:+.3f}</td>"
                       f"<td>[{lo:+.3f}, {hi:+.3f}]</td><td>{value['n']}</td></tr>")
    ablations = []
    for run in result['runs']:
        if run['protocol'] == 'wave':
            for name, evaluation in run['interventions'].items():
                ablations.append(f"<tr><td>{run['seed']}</td><th>{escape(name)}</th>"
                                 f"<td>{evaluation['success_rate']:.1%}</td>"
                                 f"<td>{evaluation['success_rate'] - run['evaluation']['success_rate']:+.3f}</td></tr>")
    curves = []
    palette = ['#6885a6', '#b8b8cc', '#ce9e67', '#df7864', '#b787d9', '#64af9a', '#79ddcc']
    max_loss = max(p['loss'] for r in result['runs'] for p in r['training_curve']) * 1.05
    for index, name in enumerate(aggregate):
        runs = [r for r in result['runs'] if r['protocol'] == name]
        series = [sum(r['training_curve'][i]['loss'] for r in runs) / len(runs)
                  for i in range(len(runs[0]['training_curve']))]
        points = ' '.join(f'{45 + i * 700 / max(1, len(series)-1):.1f},{205 - value / max_loss * 175:.1f}'
                          for i, value in enumerate(series))
        curves.append(f'<polyline points="{points}" fill="none" stroke="{palette[index % len(palette)]}" '
                      f'stroke-width="2"><title>{escape(name)}</title></polyline>')
    reasons = ''.join(f'<li>{escape(str(reason))}</li>' for reason in result['promotion']['reasons'])
    limits = ''.join(f'<li>{escape(item)}</li>' for item in result['limitations'])
    model = result['config']['model']
    neurons = model.get('neuron_count', 512)
    seeds = len(result['config']['seeds'])
    replay_count = sum(r['replay_valid'] for r in result['runs'])
    html = '''<!doctype html><html lang="en"><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>SUPERMiX AETHER · NeuroWave-X</title>
<style>
:root{color-scheme:dark;--bg:#0b1219;--panel:#111e29;--line:#263744;--ink:#e2ebf0;--muted:#93a8b6;--accent:#79ddcc}
*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--ink);font:15px/1.6 system-ui,sans-serif}
main{max-width:1180px;margin:auto;padding:42px 32px 70px}a{color:var(--accent)}header{border-bottom:1px solid var(--line);padding-bottom:32px}
.eyebrow{font-size:12px;letter-spacing:.19em;color:var(--accent)}h1{font-size:clamp(30px,5vw,54px);letter-spacing:-.045em;line-height:1.1;margin:18px 0}h2{font-size:21px;letter-spacing:-.02em;margin:0 0 15px}p{max-width:850px;color:var(--muted)}
.tag{display:inline-block;border:1px solid #635035;color:#f0c98e;border-radius:20px;padding:3px 12px;font-size:12px}.stats{display:grid;grid-template-columns:repeat(4,1fr);gap:14px;margin:26px 0}.stat,.panel{background:var(--panel);border:1px solid var(--line);border-radius:12px;padding:22px}.stat b{display:block;font-size:29px;color:var(--accent);line-height:1.3}.stat span{font-size:12px;color:var(--muted)}.grid{display:grid;grid-template-columns:1.1fr 1fr;gap:18px}.panel{margin-bottom:18px;min-width:0}.wide{grid-column:1/-1}
table{width:100%;border-collapse:collapse;text-align:left;font-size:13px}th,td{padding:10px 9px;border-bottom:1px solid var(--line)}th{font-weight:500}thead{color:var(--muted)}.scroll{overflow-x:auto}.barrow{display:grid;grid-template-columns:110px 1fr 48px;gap:12px;align-items:center;font-size:12px;margin:14px 0}.barrow b{text-align:right}.track{height:10px;background:#213341;border-radius:3px;overflow:hidden}.bar{height:100%;background:var(--accent)}svg{width:100%;height:auto}.note,li,footer{font-size:13px;color:var(--muted)}li{margin-bottom:6px}ul{padding-left:20px}footer{border-top:1px solid var(--line);padding-top:24px;margin-top:16px}code{font-size:12px;overflow-wrap:anywhere}details{margin-top:15px}summary{cursor:pointer;color:var(--accent)}pre{white-space:pre-wrap;font-size:11px;max-height:400px;overflow:auto}@media(max-width:750px){main{padding:24px 16px}.stats{grid-template-columns:1fr 1fr}.grid{grid-template-columns:1fr}.panel{padding:16px}h1{font-size:36px}}
</style><main><header><div class="eyebrow">SUPERMiX / AETHER / RESEARCH OBSERVATORY</div>
<h1>Communication, under test.</h1><span class="tag">Exploratory pilot · no promotion</span>
<p>A CPU-local study of learned communication in a sparse neural tissue. These are recorded outcomes from a delayed-cue navigation task. The policy acts in the world; the evidence determines what follows.</p>
<a href="results.json">Study data</a> &nbsp;·&nbsp; <a href="config.json">Configuration</a> &nbsp;·&nbsp; <a href="receipt.json">Artifact receipt</a>
</header>'''
    html += f'<section class="stats"><div class="stat"><b>{neurons}</b><span>neurons × {model.get("state_dimension",32)} state components</span></div>'
    html += f'<div class="stat"><b>{len(aggregate)}</b><span>trained protocol controls</span></div><div class="stat"><b>{seeds}</b><span>independent paired training seeds</span></div>'
    html += f'<div class="stat"><b>{replay_count}/{len(result["runs"])}</b><span>checkpoints replayed exactly locally</span></div></section><div class="grid">'
    html += '<section class="panel"><h2>Executed query success</h2>' + ''.join(cards) + '<p class="note">Mean over paired training seeds. Each seed contains separate held-out policy rollouts.</p></section>'
    html += '<section class="panel"><h2>Learning on support trajectories</h2><svg viewBox="0 0 790 240" role="img" aria-label="Mean support loss by training update"><path d="M45 20V205H755" fill="none" stroke="#526879"/>' + ''.join(curves) + '<text x="45" y="230" fill="#93a8b6" font-size="12">First update</text><text x="670" y="230" fill="#93a8b6" font-size="12">Last update</text></svg><p class="note">Imitation loss on executed expert actions. Lower support loss does not establish query advantage.</p></section>'
    html += '<section class="panel wide"><h2>Behaviour and resource ledger</h2><div class="scroll"><table><thead><tr><th>Protocol</th><th>Query success</th><th>Longer delay</th><th>Parameters</th><th>Training / seed</th></tr></thead><tbody>' + ''.join(rows) + '</tbody></table></div><p class="note">Updates and examples are matched. Wall time is measured; compute equivalence is unverified. All auxiliary protocols use the same payload shape. The scalar-only control sends no wave packets.</p></section>'
    html += '<section class="panel"><h2>Paired wave effects</h2><div class="scroll"><table><thead><tr><th>Control</th><th>Mean Δ</th><th>95% bootstrap interval</th><th>Seeds</th></tr></thead><tbody>' + ''.join(effects) + '</tbody></table></div><p class="note">Bootstrap units are independent training seeds. Small-sample intervals are exploratory and cannot establish superiority.</p></section>'
    html += '<section class="panel"><h2>Wave interventions</h2><div class="scroll"><table><thead><tr><th>Seed</th><th>Intervention</th><th>Success</th><th>Δ vs intact</th></tr></thead><tbody>' + ''.join(ablations) + '</tbody></table></div></section>'
    html += '<section class="panel wide"><h2>Promotion withheld</h2><ul>' + reasons + '</ul><h2>Evidence boundaries</h2><ul>' + limits + '</ul><details><summary>Recorded runtime environment</summary><pre>' + escape(json.dumps(result['metadata'], indent=2)) + '</pre></details></section></div><footer>SUPERMiX AETHER v0.1 · NeuroWave-X first milestone · Synthetic local evidence · 7 November 2026</footer></main></html>'
    Path(path).write_text(html, encoding='utf-8')
