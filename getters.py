from sde import SDE, VpSdeSigmoid, VpSdeCos, GeneralizedSubVpSdeCos, SubVpSdeCos


def get_sde(sde_type: str, sde_kwargs) -> SDE:

    # mle_training = sde_kwargs['mle_training']

    if sde_type == 'vp-sigmoid':
        # return VpSdeSigmoid(mle_training)
        return VpSdeSigmoid()

    if sde_type == 'vp-cos':
        #assert ((sde_kwargs.sigma_min is not None) and (sde_kwargs.sigma_max is not None))
        # return VpSdeCos(mle_training, sigma_min, sigma_max)
        #return VpSdeCos(sde_kwargs.sigma_min, sde_kwargs.sigma_max)
        return VpSdeCos()

    if sde_type == 'subvp-cos':
        #assert ((sde_kwargs.sigma_min is not None) and (sde_kwargs.sigma_max is not None))
        ## return SubVpSdeCos(mle_training, sigma_min, sigma_max)
        #return SubVpSdeCos(sde_kwargs.sigma_min, sde_kwargs.sigma_max)
        return SubVpSdeCos()

    if sde_type == 'generalized-sub-vp-cos':
        #assert ((sde_kwargs.sigma_min is not None) and (sde_kwargs.sigma_max is not None))
        assert ((sde_kwargs.gamma is not None) and (sde_kwargs.eta is not None))
        # return GeneralizedSubVpSdeCos(mle_training, gamma, eta, sigma_min,
        #                               sigma_max)
        #return GeneralizedSubVpSdeCos(sde_kwargs.gamma, sde_kwargs.eta, sde_kwargs.sigma_min,
        #                              sde_kwargs.sigma_max)
        return GeneralizedSubVpSdeCos(sde_kwargs.gamma, sde_kwargs.eta)

    else:
        raise NotImplementedError
