import argparse
from datasets.inst_dataset import InstDataset
from solvers.conga import CONGAKnapsack01Solver


def parse_args():
    parser = argparse.ArgumentParser("Test dataset")
    parser.add_argument("--solver", default="conga", type=str, help="Method for resolve task")
    parser.add_argument("--path", default="", type=str, help="Path to dataset")
    return parser.parse_args()


def test_dataset(args):
    ds = InstDataset(args.path)
    tot_err, tot_ep, max_ep = 0, 0, 0
    print("solver :", args.solver)
    print(" idx pred:  value (dv)                   weigth (dw)         [epoch] | true: value      weight            n   t_bst t_tot  |  file")
    for i, idx in enumerate(range(len(ds))):
        values, weights, capacity, value, name = ds[idx]
        solver = CONGAKnapsack01Solver(n_generations=3, n_agents=50, epochs=2000, lr=1e-1, rand=True,
                                           nu=1.0, mu1=0.0, mu2=7.0, beta_v=0.5, beta_w=0.5, minibatch=1.0, eps=1e-6,
                                           tau1=30, tau_hot=30, tau2=0.01, tau_warmap_epochs=1, tau_max_epochs=2000,
                                           std1=None, std_hot=None, std_warmap_epochs=None, std2=None,
                                           std_max_epochs=None,
                                           verbose=False)
        v_bst, w_bst, x_bst, e_bst, t_bst, t_tot, hist = solver.solve(values, weights, capacity)

        tot_err += (value - v_bst) / value
        tot_ep += e_bst
        max_ep = max(max_ep, e_bst)
        print(
            f"{idx:4.0f} {v_bst:12.2f} ({(value - v_bst):9.0f} = {(100 * (value - v_bst) / value):.1f}%) {w_bst:10.0f} ({(capacity - w_bst):10.0f}) [{e_bst:5d}] | {value:11.0f}, {capacity:11.0f}, {len(values):10d}  {t_bst:4.3f}s {t_tot:4.3f}s |  {name}")
    tot_err /= (i + 1)
    tot_ep /= (i + 1)
    print(f"avr err: {100 * tot_err:.4f}%  bst epoch: avr = {tot_ep:.0f} max = {max_ep:.0f}")


def main():
    args = parse_args()
    test_dataset(args)


if __name__ == "__main__":
    main()
