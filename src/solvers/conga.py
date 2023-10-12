from .base import BaseKnapsack01Solver
import torch, torch.nn as nn, torch.nn.functional as F
import math
import time


class CONGAKnapsack01Population:
    """ GA population state """
    def __init__(self) -> None:
        self.t = None  # temperature
        self.mu = None # mu distribution


class CONGAKnapsack01Solver(BaseKnapsack01Solver):
    """ CONGA algorithm for solving knapsack01 tasks """
    def __init__(self,   n_generations=1, n_agents=100, epochs=2000, lr=1e-1, rand=True,
                         nu=1.0, mu1=1.0, mu2=7.0, beta_v=0.5, beta_w=0.5, minibatch=1.0, eps=1e-6,
                         tau1=30, tau_hot=30, tau2=0.01, tau_warmap_epochs=1, tau_max_epochs=2000,
                         std1=None, std_hot=None, std_warmap_epochs=None, std2=None, std_max_epochs=None,
                         verbose=False) -> None:
        """
        Args:
            n_generations (int): number of generations
            n_agents (int): number of agents in population
            epochs (int): epochs per population
            lr (float): learning rate
            rand (bool): use stochastic in activation function
            nu (int): fraction of boundary part
            mu1 (float): minimum value of mu for initial distribution
            mu2 (float): maximum value of mu for initial distribution
            beta_v (float): EMA beta for values gradient
            beta_w (float): EMA beta for weights gradient
            minibatch (float): part of parameters for calculate gradients
            eps (float): avoid zero div in gamma calc
            tau1 (float): initial temperature for hot sigmoid
            tau_hot (float): hot temperature (cosine scheduler)
            tau2 (float): final temperature for hot sigmoid when tau_max_epochs reached
            tau_max_epochs (int): epochs where temperature reaches tau2
            std1 (float): initial std for hot sigmoid
            std_hot (float): hot std (cosine scheduler)
            std2 (float): final std for hot sigmoid when std_max_epochs reached
            std_max_epochs (int): epochs where std reaches std2
            verbose (bool): print per epoch statistics
        """
        super().__init__()
        # genetic
        self.n_generations, self.n_agents, self.epochs, self.lr, self.rand = n_generations, n_agents, epochs, lr, rand
        # optimization
        self.nu, self.mu1, self.mu2, self.beta_v, self.beta_w, self.minibatch, self.eps = nu, mu1, mu2, beta_v, beta_w, minibatch, eps
        # teperature
        self.tau1, self.tau_hot, self.tau2, self.tau_warmap_epochs, self.tau_max_epochs = tau1, tau_hot, tau2, tau_warmap_epochs, tau_max_epochs
        # std
        self.std1, self.std_hot, self.std_warmap_epochs, self.std2, self.std_max_epochs = std1, std_hot, std_warmap_epochs, std2, std_max_epochs

        self.verbose = verbose
        self.dtype = torch.float64
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

    def solve(self, values, weights, capacity):
        """ Solve knapsack01 task
        Args:
            values (list): volues of each item
            weights (list): weights of each item
            capacity (float): maximum capacity of knapsack
        """
        # meseaure performance
        tm_start = time.time()
        v_bst = -1; w_bst = -1; x_bst = None; e_bst = -1; t_bst = -1; hist = {}

        # convert to tensors
        v_data = torch.tensor(values,   dtype=self.dtype, device=self.device)
        w_data = torch.tensor(weights,  dtype=self.dtype, device=self.device)
        w_lim  = torch.tensor(capacity, dtype=self.dtype, device=self.device)
        items = len(v_data)

        # run generations
        for gen_id in range(self.n_generations):
            if gen_id == 0:
                # first population
                pop = self._init_population(items, self.mu1, self.mu2)
            else:
                # run selection for produce better population
                pop = self._selection(items, pop)
            # run generation for new population
            v_bst, w_bst, x_bst, e_bst, t_bst = self._run_generation(pop, v_data, w_data, w_lim, tm_start, gen_id, v_bst, w_bst, x_bst, e_bst, t_bst)

        return v_bst, w_bst, x_bst, e_bst, t_bst,  time.time()-tm_start,  hist

    def _run_generation(self, pop, v_data, w_data, w_lim, tm_start, gen_id, v_bst, w_bst, x_bst, e_bst, t_bst):
        """ Run cicle of generation for specific population """
        av_gv, av_gw, av_w = None, None, None
        ones  = torch.ones((self.n_agents,), device=self.device)
        drop_batch = nn.Dropout(p=1.0-self.minibatch)
        t = pop.t
        mu = pop.mu
        for epoch in range(self.epochs):
            # anneling
            tau = self._cos_scheduller(epoch, self.tau_warmap_epochs, self.tau_max_epochs, self.tau1, self.tau_hot, self.tau2)
            std = self._cos_scheduller(epoch, self.std_warmap_epochs, self.std_max_epochs, self.std1, self.std_hot, self.std2)
            # hot sigmoid
            x = self._hot_sigmoid(t, tau=tau, std=std, rand=self.rand)  # hard = True # epoch > epochs // 4
            v, w = self._value(x,v_data), self._bound(x,w_data,w_lim)
            # calc gradients
            v.backward(ones,retain_graph=True); v = v.detach(); gv = t.grad.clone();  t.grad = torch.zeros_like(t)
            w.backward(ones); w = w.detach(); gw = t.grad;
            # mask part of gradients
            msk_batch = drop_batch(torch.ones_like(gv))
            gv = gv * msk_batch
            gw = gw * msk_batch
            # EMA
            av_gv, av_gw, av_w = None, None, None
            av_gv = gv if av_gv is None else beta_v * av_gv + (1-beta_v) * gv
            av_gw = gw if av_gw is None else beta_w * av_gw + (1-beta_w) * gw
            av_w  = w #if av_w  is None else beta_w * av_w  + (1-beta_w) * w
            # calc gamma
            gvw = (av_gv * av_gw).sum(-1)
            gww = (av_gw * av_gw).sum(-1)
            gamma = F.relu((gvw + mu * av_w/self.lr)/(gww * av_w**(self.nu-1) + self.eps))
            # update t
            t.data -= self.lr * (-av_gv + gamma.unsqueeze(-1)*F.relu(av_w.unsqueeze(-1))**(self.nu-1) * av_gw)
            # clear gradients
            t.grad = torch.zeros_like(t)
            # update statistics
            v0, w0 = v.detach(), w.detach()
            v0[w0>0] = 0 # is not permitted
            # best per agent history
            pop.v_bst = torch.maximum(pop.v_bst, v0)
            # best throw all agents
            v0_max = v0.max().cpu()
            if v0_max > v_bst:
                # new best value
                v_bst = v0_max
                w_bst = w0[v0.argmax()].cpu()
                x_bst = x.detach().clone()
                e_bst = epoch + gen_id*self.epochs
                t_bst=time.time()-tm_start
        return v_bst, w_bst, x_bst, e_bst, t_bst

    def _value(self, x, v):
        """ optimization function, must be maximum"""
        return x @ v

    def _bound(self, x, w, w_lim):
        """ boundary function, must be leq 0 """
        return x @ w - w_lim

    def _hot_sigmoid(sel, t, tau=1., std=None, rand=True, eps=1e-8):
        """ activation function for decision put item into knapsack or not """
        if rand:
            # add stochastic to decision
            if std is None:
                r = torch.rand_like(t)                   # ~ U(0,1)
                r = torch.log(eps + r / (1-r+eps))       # ~ L(0,1)
            else:
                r = std * torch.empty_like(t).normal_()  # ~ N(0,std)
            t = t + r

        x_soft = torch.sigmoid(t/tau)
        x_hard = x_soft.detach().round()
        #x_hard = (x_soft.detach()-0.5).heaviside(torch.tensor([0.], device=t.device, dtype=torch.float64))
        return x_hard + (x_soft - x_soft.detach())

    def _cos_scheduller(self, step, steps_warmap, steps_max, v1, v_hot, v2):
        """ Cosine scheduller """
        if v1 is None:
            return None

        if step > steps_warmap:
            if step > steps_max:
                # constant
                return v2
            else:
                # cosine scheduller
                s = (step - steps_warmap) / (steps_max - steps_warmap)
                return v2 + (v_hot - v2) * (1 + math.cos(math.pi*s)) / 2
        else:
            # hot linear process
            if steps_warmap > 0:
                return v1 + (v_hot - v1) * step / steps_warmap

    def _init_population(self, items, mu1, mu2):
        """ Initialize population """
        # new population
        pop = CONGAKnapsack01Population()
        # random initialization of knapsack state
        pop.t = nn.Parameter(-torch.randn((self.n_agents,items), dtype=self.dtype, device=self.device).abs() / items**0.5 )
        # uniform initialization of mu
        pop.mu = torch.distributions.uniform.Uniform(mu1,mu2).sample([self.n_agents]).to(self.device)
        # best values
        pop.v_bst = torch.zeros((self.n_agents), dtype=self.dtype, device=self.device)
        return pop

    def _selection(self, items, prev_pop):
        """ Selection process """
        if prev_pop is None:
            # first population
            return self._init_population(items, self.mu1, self.mu2)
        # sort population by best value
        sorted_indexes = sorted(range(self.n_agents), key=lambda k: prev_pop.v_bst[k], reverse=True)
        # select best portion of the existing population
        select_portion = 0.2
        # selection
        best_indexes = sorted_indexes[:int(self.n_agents*select_portion)]
        # crossover of mu parameter
        pop_mu = prev_pop.mu[best_indexes]
        min_mu = pop_mu.min()
        max_mu = pop_mu.max()
        extra_frac = 2.0
        min_mu_next = min_mu / extra_frac
        max_mu_next = max_mu * extra_frac
        #print('selection:', min_mu, max_mu)
        # create new population
        next_pop = self._init_population(items, min_mu_next, max_mu_next)
        return next_pop
                                    