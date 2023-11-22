import math

import numpy as np
import pandas as pd


def compare_frames(test_frame, reference_frame, known_differences, time_tolerance=5e-4):
    # sanity checks
    # all returned values are numbers
    assert test_frame.isna().any(axis=None).item() is False
    # all time steps are unique
    assert test_frame.t.is_unique and reference_frame.t.is_unique
    # time starts with 0
    assert test_frame.t[0] == 0
    # time is all positive
    assert (test_frame.t[1:] > 0).all()
    # time is ordered and strictly increasing
    assert (test_frame.t.diff()[1:] > 0).all()
    # time steps are decreasing non monotonically
    # this is not true
    # assert (frame_ehm.t.diff()[1:].diff().diff() < 0).all()

    # we generate the same number of elements in the dynamic
    assert len(reference_frame.columns) == reference_frame.shape[1]

    # dynamics stop at more or less the same time
    assert (
        test_frame[["t"]].max() - reference_frame[["t"]].max()
    ).abs().item() < time_tolerance

    # we create a new index composite of both index, keep unique values
    new_index_t = np.array(
        sorted(pd.concat((reference_frame.t, test_frame.t)).unique())
    )

    # we now inject the time steps of the reference frame into the current frame
    test_frame_missing_values = test_frame.set_index("t").reindex(new_index_t)

    # we interpolate the missing values (method is important, index takes the time
    # point for performing the interpolation but is not precise enough)
    test_frame_missing_values_interpolated = test_frame_missing_values.interpolate(
        # method="index"
        method="cubic",
        order=3,
        limit_area="inside",
    )

    # all timesteps of the reference frame have now a proper value
    if test_frame.t.max() > reference_frame.t.max():
        assert (
            test_frame_missing_values_interpolated.loc[reference_frame.t]
            .isna()
            .any(axis=None)
            .item()
            is False
        )
    else:
        # drop the last element in this case
        assert (
            test_frame_missing_values_interpolated.loc[reference_frame.t]
            .isna()
            .iloc[:-1]
            .any(axis=None)
            .item()
            is False
        )

    test_frame_projected = test_frame_missing_values_interpolated.loc[reference_frame.t]

    # dynamics do not necessarily have the same subdivision
    reference_frame_t = reference_frame.set_index("t")
    for idx_column, column in enumerate(reference_frame_t.columns):
        # transforms 5.755399e-05 to 1e-4
        abs_diff = known_differences[column]
        abs_diff = 10 ** math.ceil(math.log10(abs_diff))

        assert (
            reference_frame_t[[column]] - test_frame_projected[[column]]
        ).abs().max().item() < abs_diff, f"column {column}"
