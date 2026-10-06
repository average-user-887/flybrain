import pytest

from neurofly_body.decoder import DNa02CPGDecoder


def test_zero_rates_produce_exact_zero_drive():
    decoder = DNa02CPGDecoder()
    output = decoder.decode(0.0, 0.0, 2.0)
    assert output["left_cpg_drive"] == 0.0
    assert output["right_cpg_drive"] == 0.0
    assert decoder.describe()["tonic_drive"] == 0.0


def test_dna02_mapping_is_crossed_and_symmetric():
    left_dn = DNa02CPGDecoder(gain_per_hz=0.01, tau_ms=2.0, max_drive=10.0)
    right_dn = DNa02CPGDecoder(gain_per_hz=0.01, tau_ms=2.0, max_drive=10.0)
    left = left_dn.decode(20.0, 0.0, 2.0)
    right = right_dn.decode(0.0, 20.0, 2.0)
    assert left["left_cpg_drive"] == 0.0
    assert left["right_cpg_drive"] > 0.0
    assert right["right_cpg_drive"] == 0.0
    assert right["left_cpg_drive"] == pytest.approx(left["right_cpg_drive"])


def test_decoder_refuses_nonphysical_input():
    decoder = DNa02CPGDecoder()
    with pytest.raises(ValueError):
        decoder.decode(-1.0, 0.0, 2.0)
    with pytest.raises(ValueError):
        decoder.decode(float("nan"), 0.0, 2.0)


def test_rate_fields_are_named_for_what_they_measure():
    """D2: no `raw_rate_*_hz`; integer counts and bin rates are distinct."""
    decoder = DNa02CPGDecoder()
    out = decoder.decode(500.0, 0.0, 2.0)
    assert "raw_rate_l_hz" not in out and "raw_rate_r_hz" not in out
    # One spike of one neuron in a 2 ms bin is 500 Hz as a bin rate.
    assert out["dna02_spikes_l"] == 1
    assert out["dna02_spikes_r"] == 0
    assert out["dna02_bin_rate_l_hz"] == 500.0
    assert out["dna02_bin_rate_r_hz"] == 0.0
    # The filtered rate is the only WP5-comparable quantity, and is far below
    # the bin rate it came from.
    assert 0.0 < out["filtered_rate_l_hz"] < 20.0
    conventions = decoder.describe()["rate_conventions"]
    assert "NOT comparable" in conventions["dna02_bin_rate_l_and_r_hz"]
    assert "WP5" in conventions["filtered_rate_l_and_r_hz"]


def test_dn_v2_states_which_rate_is_wp5_comparable():
    """D2 under dn-v2: the per-bin rates keep master's names but say what they are."""
    from neurofly_body.decoder import DNCommandDecoder

    conventions = DNCommandDecoder().describe()["rate_conventions"]
    assert "NOT comparable" in conventions["raw_rates_hz"]
    assert "WP5" in conventions["filtered_rates_hz"]
