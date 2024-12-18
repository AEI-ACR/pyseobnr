# autogenerated, please report back the changes upstream.

import inspect
from pathlib import Path
from typing import Final

import numpy as np

import pytest

from pyseobnr.eob.dynamics.secular_evolution_equations_flags import edot_zdot_xdot_flags


@pytest.fixture(params=["num1", "num2"], ids=["num1", "num2"])
def extension(request):
    return request.param


params_evolution_eq_init: Final = {
    "nu",
    "chiA",
    "chiS",
    "delta",
    "flagPN1",
    "flagPN32",
    "flagPN2",
    "flagPN52",
    "flagPN3",
}

params_evolution_eq_compute: Final = {
    "e",
    "z",
    "x",
}


@pytest.fixture()
def read_values(extension):
    base_folder = Path(__file__).parent

    all_values = {}

    previous_input_params = None
    init_params = None
    input_params = None

    for filename in sorted(base_folder.rglob(f"{extension}.txt")):
        expression_name_ = filename.parent.name.lower()
        with filename.open() as file:

            def _transform(value_str: str):
                value_str = value_str.replace("*^-", "e-")
                return float(value_str.split(":")[-1])

            num_vals = [_transform(line) for line in file]
            input_vals = num_vals[:-2] + [
                1,  # flagPN1
                1,  # flagPN12
                1,  # flagPN32
                1,  # flagPN2
                1,  # flagPN52
                1,  # flagPN3
                1,  # flagPA
                1,  # flagPA_modes
                1,  # flagTail
                1,  # flagMemory
            ]

            # We add the flags for the PN orders
            expected_vals = num_vals[-2:]

        (
            nu,
            delta,
            chiA,
            chiS,
            e,
            z,
            omega_avg,
            omega_inst,
            x,
            L,
            flagPN1,
            flagPN12,
            flagPN32,
            flagPN2,
            flagPN52,
            flagPN3,
            flagPA,
            flagPA_modes,
            flagTail,
            flagMemory,
        ) = input_vals

        init_params = {
            "nu": nu,
            "chiA": chiA,
            "chiS": chiS,
            "delta": delta,
            "flagPN1": flagPN1,
            "flagPN12": flagPN12,
            "flagPN32": flagPN32,
            "flagPN2": flagPN2,
            "flagPN52": flagPN52,
            "flagPN3": flagPN3,
            # "flagPA": flagPA,
            "flagPA": flagPA_modes,
            "flagTail": flagTail,
            "flagMemory": flagMemory,
        }

        input_params = {
            "omega": omega_inst,
            "omega_avg": omega_avg,
            "L": L,
            "e": e,
            "x": x,
            "z": z,
        }

        if previous_input_params is None:
            previous_input_params = init_params, input_params
        else:
            assert previous_input_params == (init_params, input_params)

        all_values[expression_name_] = expected_vals[0] + expected_vals[1] * 1j

    init_params = {k: init_params[k] for k in params_evolution_eq_init}
    input_params = {k: input_params[k] for k in params_evolution_eq_compute}

    return init_params, input_params, all_values


@pytest.fixture()
def new_instance():
    """Abstracts the instantiation of the modes class"""
    # name of the package and class correspond to the ones used in tox
    return edot_zdot_xdot_flags()


class TestComputationsForces:
    def test_values(self, new_instance, read_values):
        """Equivalence check between Mathematica and cython numerical values: direct access"""

        init_params, input_values, all_expected = read_values

        instance = new_instance
        instance.initialize(**init_params)
        instance.compute(**input_values)

        mapping_name_internal_store = {
            "edot": "edotExpResumParser",
            "xdot": "xdotExpResumParser",
            "zdot": "zdotParser",
        }

        for expression_name_ in sorted(all_expected):
            name = mapping_name_internal_store[expression_name_]

            assert hasattr(instance, name)
            ret = getattr(instance, name)

            assert np.allclose(
                [ret],
                [all_expected[expression_name_]],
                rtol=1e-6,
                atol=1e-10,
            )

    def test_values_through_accessor(self, new_instance, read_values):
        """Equivalence check between Mathematica and cython numerical values: access through get"""
        init_params, input_values, all_expected = read_values

        instance = new_instance
        instance.initialize(**init_params)
        instance.compute(**input_values)

        for expression_name_ in sorted(all_expected):
            ret = instance.get(expression_name_)
            assert np.allclose(
                [ret],
                [all_expected[expression_name_]],
                rtol=1e-6,
                atol=1e-10,
            )

    def test_accessor_incorrect_expressions(self, new_instance, read_values):
        """Equivalence check between Mathematica and cython numerical values: access through get"""
        init_params, input_values, all_expected = read_values

        instance = new_instance
        instance.initialize(**init_params)
        instance.compute(**input_values)

        for expression_name_ in (
            "some",
            "things",
            "xdotExpResum__Parser",
        ):  # lower case only
            with pytest.raises(RuntimeError) as exc_info:
                instance.get(expression_name_)

            assert (
                exc_info.value.args[0] == f"Unsupported expression '{expression_name_}'"
            )

    def test_raises_exception_when_incorrectly_called(self, new_instance):
        instance = new_instance

        # initialize has not been called yet
        with pytest.raises(RuntimeError) as exc_info:
            instance.compute(e=0, z=0, x=1)

        assert exc_info.value.args == ("Instance has not been initialized yet",)

    def test_doc(self, new_instance):
        """Checks the documentation exists"""
        class_instance = type(new_instance)

        assert inspect.getdoc(class_instance) is not None
        assert inspect.getdoc(class_instance) != ""