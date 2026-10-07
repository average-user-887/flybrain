"""The Science Guide names only real connected controls, and every planned knob is in the backlog.

Part C of the guide in web/index.html lists the connected assay controls. They must
match assay_controls.describe() exactly, in both directions, so the guide can never
advertise a control the daemon refuses or hide one it offers. Planned knobs must name
existing POST_V04 backlog IDs.
"""
import html
import re
from pathlib import Path

import pytest

from arena import Arena
from assay_controls import describe
from experiment_brains import PARADIGMS

ROOT = Path(__file__).resolve().parents[1]
INDEX = (ROOT / 'web/index.html').read_text(encoding='utf-8')
BACKLOG = (ROOT / 'docs/POST_V04_FEATURES.md').read_text(encoding='utf-8')
GUIDE = INDEX[INDEX.index('id="scienceGuideModal"'):INDEX.index('</body>')]
CONTROLS = [(pid, name, html.unescape(label)) for pid, name, label in
            re.findall(r'<span data-guide-control="([a-z-]+):([A-Za-z_]+)">([^<]+)</span>', GUIDE)]


def arena(pid):
    return Arena(paradigm=None if pid == 'open-arena' else pid, seed=4, num_flies=1, num_predators=0)


def offered(pid):
    info = describe(arena(pid))
    return {item['name']: item['label'] for item in info['parameters'] + info['actions']}


def test_guide_lists_connected_controls():
    assert len(CONTROLS) >= 10
    assert len({(pid, name) for pid, name, _ in CONTROLS}) == len(CONTROLS)


@pytest.mark.parametrize('pid,name,label', CONTROLS)
def test_every_named_control_is_a_real_connected_control(pid, name, label):
    controls = offered(pid)
    assert name in controls, f'{pid}:{name} is named in the guide but is not a connected control'
    assert label == controls[name], f'{pid}:{name} guide label {label!r} != daemon label {controls[name]!r}'


@pytest.mark.parametrize('pid', PARADIGMS)
def test_every_connected_control_appears_in_the_guide(pid):
    named = {name for p, name, _ in CONTROLS if p == pid}
    assert set(offered(pid)) == named, f'{pid}: guide and assay_controls.describe() disagree'


def test_planned_knobs_name_existing_backlog_ids():
    planned = re.findall(r'<td class="guide-planned" data-planned="([^"]+)">([^<]+)</td>', GUIDE)
    assert len(planned) >= 13
    for ids, text in planned:
        for item in ids.split():
            assert re.fullmatch(r'NEXT-\d\d', item)
            assert f'| {item} |' in BACKLOG, f'{item} is not in docs/POST_V04_FEATURES.md'
            assert item in text
        assert html.unescape(text).startswith('Planned — not in v0.4')
