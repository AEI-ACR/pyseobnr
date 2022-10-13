import numpy as np
from skopt.space import Space
from skopt.sampler import Lhs


def parameters_random(
    N: int,
    qmin: float,
    qmax: float,
    chi1min: float,
    chi1max: float,
    chi2min: float,
    chi2max: float,
    random_state: int = None,
):
    """
    Generate a Latin Hypercube sampling for q, chi1, chi2

    Parameters
    ----------
    N:
        Number of parameters to generate
    qmin, qmax:
        Bounds on the q parameter
    chi1min, chi1max, chi2min, chi2max:
        Bounds on the chi parameter

    Return parameters of the waveforms
    """
    space = Space([(qmin, qmax), (chi1min, chi1max), (chi2min, chi2max)])
    lhs = Lhs(
        lhs_type="classic", criterion="maximin", iterations=1000
    )  # Latin hypercube sampling
    q = []
    chiA = []
    chiB = []
    if N > 0:
        rand = lhs.generate(space.dimensions, N, random_state = random_state)
        for i in range(N):
            q.append(rand[i][0])
            chiA.append(rand[i][1])
            chiB.append(rand[i][2])
    return (
        np.array(q),
        np.array(chiA),
        np.array(chiB),
    )


def parameters_random_2D(
    N: int,
    qmin: float,
    qmax: float,
    chi1min: float,
    chi1max: float,
    random_state: int = None,
):
    """
    Generate a Latin Hypercube sampling for q, chi1 (for NRHybSur2dq15)

    Parameters
    ----------
    N:
        Number of parameters to generate
    qmin, qmax:
        Bounds on the q parameter
    chi1min, chi1max:
        Bounds on the chi parameter

    Return parameters of the waveforms
    """
    space = Space([(qmin, qmax), (chi1min, chi1max)])
    lhs = Lhs(
        lhs_type="classic", criterion="maximin", iterations=1000
    )  # Latin hypercube sampling
    q = []
    chiA = []
    if N > 0:
        rand = lhs.generate(space.dimensions, N, random_state = random_state)
        for i in range(N):
            q.append(rand[i][0])
            chiA.append(rand[i][1])
    return (
        np.array(q),
        np.array(chiA),
    )

def q_to_nu(q):
    return q/(1+q)**2

def nu_to_q(nu):
    delta_m = np.sqrt(np.abs(1 - 4 * nu))
    return ((1 - 2 * nu) + delta_m) / (2 * nu)


def parameters_random_fast(
    N: int,
    qmin: float,
    qmax: float,
    chi1min: float,
    chi1max: float,
    chi2min: float,
    chi2max: float,
    random_state=1
):
    """
    Generate random parameters for q, chi1, chi2

    Parameters
    ----------
    N:
        Number of parameters to generate
    qmin, qmax:
        Bounds on the q parameter
    chi1min, chi1max, chi2min, chi2max:
        Bounds on the chi parameter

    Return parameters of the waveforms
    """
    np.random.seed(random_state)
    #q = np.random.uniform(qmin, qmax, N)
    nu = np.random.uniform(q_to_nu(qmin),q_to_nu(qmax),N)
    q = nu_to_q(nu)
    chiA = np.random.uniform(chi1min, chi1max, N)
    chiB = np.random.uniform(chi2min, chi2max, N)

    return (
        np.array(q),
        np.array(chiA),
        np.array(chiB),
    )


def parameters_random_2D_fast(
    N: int,
    qmin: float,
    qmax: float,
    chi1min: float,
    chi1max: float,
    random_state=1,
):
    """
    Generate random parameters for q, chi1 (for NRHybSur2dq15)

    Parameters
    ----------
    N:
        Number of parameters to generate
    qmin, qmax:
        Bounds on the q parameter
    chi1min, chi1max:
        Bounds on the chi parameter

    Return parameters of the waveforms
    """
    np.random.seed(random_state)

    #q = np.random.uniform(qmin, qmax, N)
    nu = np.random.uniform(q_to_nu(qmin),q_to_nu(qmax),N)
    q = nu_to_q(nu)
    chiA = np.random.uniform(chi1min, chi1max, N)

    return (
        np.array(q),
        np.array(chiA),
    )