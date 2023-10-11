import argparse
from solvers.conga import CONGAKnapsack01Solver


def parse_args():
    parser = argparse.ArgumentParser("Solve knapsack01 task")
    parser.add_argument("--solver", default="conga", type=str, help="Method for resolve task")
    parser.add_argument("--values", nargs="+", type=int, required=True, help="List of item values")
    parser.add_argument("--weights", nargs="+", type=int, required=True, help="List of item weights")
    parser.add_argument("--capacity", default=0, type=int, help="Maximum capacity of knapsack")
    return parser.parse_args()


def main():
    args = parse_args()
    solver = CONGAKnapsack01Solver(n_generations=1, n_agents=100, epochs=2000, lr=1e-1, rand=True,
                               nu=1.0, mu1=1.0, mu2=7.0, beta_v=0.5, beta_w=0.5, minibatch=1.0, eps=1e-6,
                               tau1=30, tau_hot=30, tau2=0.01, tau_warmap_epochs=1, tau_max_epochs=2000,
                               std1=None, std_hot=None, std_warmap_epochs=None, std2=None, std_max_epochs=None,
                               verbose=False)
    v_bst, w_bst, x_bst, e_bst, t_bst,  t_tot,  hist = solver.solve(args.values, args.weights, args.capacity)
    print('best value:', v_bst.item())


if __name__ == "__main__":
    main()
