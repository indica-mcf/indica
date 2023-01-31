from indica.bayesmodels import BayesModels, get_uniform

from indica.models.plasma import example_run
from indica.models.helike_spectroscopy import Helike_spectroscopy
from tests.integration.models.test_helike_spectroscopy import helike_LOS_example

import flatdict
import numpy as np
import emcee


class TestBayesModels():

    def setup_class(self):
        self.plasma = example_run(pulse=9229)
        self.plasma.time_to_calculate = self.plasma.t[1]
        self.los_transform = helike_LOS_example(nchannels=1)
        self.los_transform.set_equilibrium(self.plasma.equilibrium)

    def test_simple_run_bayesmodels_with_xrcs(self):
        xrcs = Helike_spectroscopy(name="xrcs", )
        xrcs.plasma = self.plasma
        xrcs.set_los_transform(self.los_transform)

        priors = {
            "Te_prof.y0": get_uniform(2e3, 5e3),
            # "Te_prof_peaking": get_uniform(1, 5),
            "Ti_prof.y0": get_uniform(2e3, 8e3),
            # "Ti_prof_peaking": get_uniform(1, 5),
        }

        bckc = {}
        bckc = dict(bckc, **{xrcs.name: {**xrcs(calc_spectra=False)}})
        flat_phantom_data = flatdict.FlatDict(bckc, delimiter=".")

        bm = BayesModels(
            plasma=self.plasma,
            data=flat_phantom_data,
            diagnostic_models=[xrcs],
            quant_to_optimise=["xrcs.ti_w", "xrcs.te_kw"],
            priors=priors,
        )

        # Setup Optimiser
        param_names = [
            "Te_prof.y0",
            # "Te_prof.peaking",
            "Ti_prof.y0",
            # "Ti_prof.peaking",
        ]

        ndim = param_names.__len__()
        nwalkers = ndim * 2
        start_points = bm.sample_from_priors(param_names, size=nwalkers)

        move = [emcee.moves.StretchMove()]
        sampler = emcee.EnsembleSampler(
            nwalkers,
            ndim,
            log_prob_fn=bm.ln_posterior,
            parameter_names=param_names,
            moves=move,
            kwargs={"minimum_lines": True}
        )
        sampler.run_mcmc(start_points, 10, progress=False)


if __name__ == "__main__":
    test = TestBayesModels()
    test.setup_class()
    test.test_simple_run_bayesmodels_with_xrcs()
