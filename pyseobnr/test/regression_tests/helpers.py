import numpy as np
import pandas as pd


def compare_frames(
    test_frame, reference_frame, known_differences_percentage, time_tolerance_percent=1
):
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
    # this is a very rough check that values are not completely crazy
    t1_end = test_frame[["t"]].max().item()
    t2_end = reference_frame[["t"]].max().item()

    assert abs(t1_end - t2_end) / max(t1_end, t2_end) < (time_tolerance_percent / 100)

    # we create a new index composite of both index, keep unique values
    new_index_t = np.array(
        sorted(pd.concat((reference_frame.t, test_frame.t)).unique())
    )

    # we now inject the time steps of the reference frame into the current frame
    test_frame_missing_values = test_frame.set_index("t").reindex(new_index_t)
    reference_frame_missing_values = reference_frame.set_index("t").reindex(new_index_t)

    # we interpolate the missing values (method is important, index takes the time
    # point for performing the interpolation but is not precise enough)
    test_frame_missing_values_interpolated = test_frame_missing_values.interpolate(
        # method="index"
        method="cubic",
        order=3,
        limit_area="inside",
    )

    reference_frame_missing_values_interpolated = reference_frame_missing_values.interpolate(
        # method="index"
        method="cubic",
        order=3,
        limit_area="inside",
    )

    # we keep the section common to both
    test_frame_projected = test_frame_missing_values_interpolated[
        test_frame_missing_values_interpolated.index <= min(t1_end, t2_end)
    ]
    reference_frame_projected = reference_frame_missing_values_interpolated[
        reference_frame_missing_values_interpolated.index <= min(t1_end, t2_end)
    ]

    # all timesteps of both reference and test frames have now a proper value and they align
    assert test_frame_projected.isna().any(axis=None).item() is False
    assert reference_frame_projected.isna().any(axis=None).item() is False
    assert (test_frame_projected.index == reference_frame_projected.index).all()

    for idx_column, column in enumerate(reference_frame_projected.columns):
        percentage_tolerance = known_differences_percentage[column]

        diffs_column = (
            reference_frame_projected[[column]] - test_frame_projected[[column]]
        ).abs()

        fractional_diffs_column = (
            100
            * diffs_column[column]
            / pd.concat(
                (reference_frame_projected[column], test_frame_projected[column]),
                axis=1,
            ).max(axis=1)
        )

        assert fractional_diffs_column.max().item() < percentage_tolerance, (
            f"column {column} exceeds fractional tolerance: "
            f"%g > {percentage_tolerance} at index %d and time %g "
            "(reference=%g, test=%g)"
        ) % (
            fractional_diffs_column.max().item(),
            fractional_diffs_column.argmax(),
            fractional_diffs_column.idxmax().item(),
            reference_frame_projected.iloc[fractional_diffs_column.argmax()][
                [column]
            ].item(),
            test_frame_projected.iloc[fractional_diffs_column.argmax()][
                [column]
            ].item(),
        )
