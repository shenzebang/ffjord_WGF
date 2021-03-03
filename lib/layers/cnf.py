import torch
import torch.nn as nn

from torchdiffeq import odeint_adjoint as odeint

from .wrappers.cnf_regularization import RegularizedODEfunc

__all__ = ["CNF", "TestFlow", "TestFlowGaussian", "TestFlowDVP"]


class CNF(nn.Module):
    def __init__(self, odefunc, T=1.0, train_T=False, regularization_fns=None, solver='dopri5', atol=1e-5, rtol=1e-5):
        super(CNF, self).__init__()
        if train_T:
            self.register_parameter("sqrt_end_time", nn.Parameter(torch.sqrt(torch.tensor(T))))
        else:
            self.register_buffer("sqrt_end_time", torch.sqrt(torch.tensor(T)))

        nreg = 0
        if regularization_fns is not None:
            odefunc = RegularizedODEfunc(odefunc, regularization_fns)
            nreg = len(regularization_fns)
        self.odefunc = odefunc
        self.nreg = nreg
        self.regularization_states = None
        self.solver = solver
        self.atol = atol
        self.rtol = rtol
        self.test_solver = solver
        self.test_atol = atol
        self.test_rtol = rtol
        self.solver_options = {}
        # TODO: Add a switch to control whether the wgf is computed

    def forward(self, z, logpz, score, wgf_reg):
        _logpz = logpz

        _score = score

        _wgf_reg = wgf_reg


        integration_times = torch.tensor([0.0, self.sqrt_end_time * self.sqrt_end_time]).to(z)

        # Refresh the odefunc statistics.
        self.odefunc.before_odeint()

        # Add regularization states.
        reg_states = tuple(torch.tensor(0).to(z) for _ in range(self.nreg))

        if self.training:
            state_t = odeint(
                self.odefunc,
                (z, _logpz, _score, _wgf_reg) + reg_states,
                integration_times.to(z),
                atol=self.atol,
                rtol=self.rtol,
                method=self.solver,
                options=self.solver_options,
            )
        else:
            state_t = odeint(
                self.odefunc,
                (z, _logpz, _score, _wgf_reg),
                integration_times.to(z),
                atol=self.test_atol,
                rtol=self.test_rtol,
                method=self.test_solver,
            )

        if len(integration_times) == 2:
            state_t = tuple(s[1] for s in state_t)

        z_t, logpz_t, score_t, wgf_reg_t = state_t[:4]

        return z_t, logpz_t, score_t, wgf_reg_t

    def get_regularization_states(self):
        reg_states = self.regularization_states
        self.regularization_states = None
        return reg_states

    def num_evals(self):
        return self.odefunc._num_evals.item()


class TestFlow(nn.Module):
    def __init__(self, odefunc, solver='dopri5', atol=1e-5, rtol=1e-5):
        super(TestFlow, self).__init__()
        self.odefunc = odefunc
        self.solver = solver
        self.atol = atol
        self.rtol = rtol
        self.solver_options = {}

    def num_evals(self):
        return self.odefunc._num_evals.item()
    # def forward(self, x, score, score_diff, integration_times):
    #     self.odefunc.before_odeint()
    #     integration_times = torch.tensor([0, integration_times])
    #     # integration_times = torch.tensor([integration_times])
    #     state_t = odeint(
    #         self.odefunc,
    #         (x, score, score_diff),
    #         integration_times.to(x),
    #         atol=self.atol,
    #         rtol=self.rtol,
    #         method=self.solver,
    #         options=self.solver_options,
    #     )
    #
    #     if len(integration_times) == 2:
    #         state_t = tuple(s[1] for s in state_t)
    #
    #     x_t, y_t, score_t, score_diff_t = state_t
    #     return x_t, y_t, score_t, score_diff_t
    #
    # def num_evals(self):
    #     return self.odefunc._num_evals.item()


class TestFlowGaussian(TestFlow):
    def __init__(self, odefunc, solver='dopri5', atol=1e-5, rtol=1e-5):
        super(TestFlowGaussian, self).__init__(odefunc, solver=solver, atol=atol, rtol=rtol)

    def forward(self, x, mu, sigma_half, diff, integration_times):
        self.odefunc.before_odeint()
        integration_times = torch.tensor([0, integration_times])
        # integration_times = torch.tensor([integration_times])
        state_t = odeint(
            self.odefunc,
            (x, mu, sigma_half, diff),
            integration_times.to(x),
            atol=self.atol,
            rtol=self.rtol,
            method=self.solver,
            options=self.solver_options,
        )

        if len(integration_times) == 2:
            state_t = tuple(s[1] for s in state_t)

        x_t, mu_t, sigma_half_t, diff_t = state_t
        return x_t, mu_t, sigma_half_t, diff_t

class TestFlowDVP(TestFlow):
    def __init__(self, odefunc, solver='dopri5', atol=1e-5, rtol=1e-5):
        super(TestFlowDVP, self).__init__(odefunc, solver=solver, atol=atol, rtol=rtol)

    def forward(self, x, diff, integration_times):
        self.odefunc.before_odeint()
        integration_times = torch.tensor([0, integration_times])
        # integration_times = torch.tensor([integration_times])
        state_t = odeint(
            self.odefunc,
            (x, diff),
            integration_times.to(x),
            atol=self.atol,
            rtol=self.rtol,
            method=self.solver,
            options=self.solver_options,
        )

        if len(integration_times) == 2:
            state_t = tuple(s[1] for s in state_t)

        x_t, diff_t = state_t
        return x_t, diff_t

def _flip(x, dim):
    indices = [slice(None)] * x.dim()
    indices[dim] = torch.arange(x.size(dim) - 1, -1, -1, dtype=torch.long, device=x.device)
    return x[tuple(indices)]
