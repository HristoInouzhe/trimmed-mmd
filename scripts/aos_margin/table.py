import json, numpy as np
def show(fn, keys, label, every=3):
    R = json.load(open(fn)); print(f'\n{label}  ({fn})'); print('| eps | path u | ' + ' | '.join(keys) + ' | all OK |'); print('|' + '---|'*(len(keys)+3))
    allok = True
    for i, r in enumerate(R):
        ok = all((r[k] <= r['u']) if k != 'ub' else (r[k] >= r['u']) for k in keys); allok &= ok
        if i % every == 0 or i == len(R)-1:
            print(f"| {r['e']:.2e} | {r['u']:.3e} | " + ' | '.join(f"{r[k]:.3e}" for k in keys) + f" | {ok} |")
    print('ALL ROWS OK:', allok, 'n rows', len(R))
show('lap_mu3_fine.json', ['lb_sym', 'lb_asym', 'lb_dual', 'ub'], 'Laplace, mu=3, fine grid', 2)
show('lap_mu3_unif.json', ['lb_sym', 'lb_asym', 'lb_dual', 'ub'], 'Laplace, mu=3, uniform h=.04', 3)
show('lap_mu1.5_fine.json', ['lb_sym', 'lb_asym', 'lb_dual', 'ub'], 'Laplace, mu=1.5, fine grid', 2)
show('gauss_mu3_h0.01.json', ['lb_crude', 'lb_dual'], 'Gaussian sigma=1, mu=3, h=.01', 4)
show('gauss_mu3_h0.04.json', ['lb_crude', 'lb_dual'], 'Gaussian sigma=1, mu=3, h=.04', 5)
show('gauss_mu2.json', ['lb_crude', 'lb_dual'], 'Gaussian sigma=1, mu=2, h=.02', 4)
show('gauss_mu1.5.json', ['lb_crude', 'lb_dual'], 'Gaussian sigma=1, mu=1.5, h=.02', 4)
