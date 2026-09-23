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
