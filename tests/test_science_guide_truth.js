'use strict';
// The Science Guide keeps published fly results, the engineered preview and v0.4 support apart.
// Performance numbers may appear only in Part A (published research), every Part A row carries a
// source or "citation needed", and every legacy knob is either a connected control (pinned against
// assay_controls.describe() by tests/test_science_guide_truth.py) or marked planned with a backlog ID.

const test = require('node:test');
const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');
const vm = require('node:vm');

const root = path.resolve(__dirname, '..');
const html = fs.readFileSync(path.join(root, 'web/index.html'), 'utf8');
const app = fs.readFileSync(path.join(root, 'web/app.js'), 'utf8');
const backlog = fs.readFileSync(path.join(root, 'docs/POST_V04_FEATURES.md'), 'utf8');

const guide = html.slice(html.indexOf('id="scienceGuideModal"'), html.indexOf('</body>'));
const ENTITIES = {amp: '&', lt: '<', gt: '>', le: '≤', ge: '≥', mdash: '—', ndash: '–', minus: '−', deg: '°',
    plusmn: '±', asymp: '≈', times: '×', rarr: '→', middot: '·', gamma: 'γ', Delta: 'Δ', sup2: '²', Aacute: 'Á'};
const text = (s) => s.replace(/<[^>]+>/g, ' ').replace(/&([a-zA-Z0-9]+);/g, (m, e) => ENTITIES[e] ?? m)
    .replace(/\s+/g, ' ').trim();

/** The outer HTML of the element with this id (balanced on <div>). */
function element(id) {
    const start = guide.indexOf(`id="${id}"`);
    assert.notEqual(start, -1, id);
    const open = guide.lastIndexOf('<div', start);
    const re = /<\/?div\b/g;
    re.lastIndex = open;
    let depth = 0, m;
    while ((m = re.exec(guide))) {
        depth += m[0] === '<div' ? 1 : -1;
        if (depth === 0) return guide.slice(open, guide.indexOf('>', m.index) + 1);
    }
    throw new Error('unbalanced ' + id);
}
const rows = (s) => s.match(/<tr\b[\s\S]*?<\/tr>/g) || [];

const research = element('guideResearch');
const preview = element('guidePreview');
const supported = element('guideSupported');
const outsideResearch = guide.replace(research, '');

// A score, rate, percentage, threshold or outcome claim.
const PERFORMANCE = [
    /\bPI\b/, /performance index/i, /\d\s*%/, /[<>≤≥]\s*\d/, /\b\d+(\.\d+)?\s*ms\b/,
    /\b(high|complete|total|robust|rapid|fails?|loss of|blind to|deaf to|amnesia|captured)\b[^.]*\b(learning|memory|fixation|escape|orientation|tracking|predators?|takeoff|anemotaxis)\b/i,
    /expected (phenotype|deficit|performance)/i,
];

test('the guide has three labelled parts in order', () => {
    const order = ['guideResearch', 'guidePreview', 'guideSupported'].map((id) => guide.indexOf(`id="${id}"`));
    assert.ok(order.every((i, k) => i > 0 && (k === 0 || i > order[k - 1])), String(order));
    assert.match(text(research), /^Part A Published fly research: context, not model output/);
    assert.match(text(preview), /^Part B Engineered modular preview: illustrative, not connectome results/);
    assert.match(text(supported), /^Part C What v0\.4 actually supports/);
    assert.match(text(supported), /neurofly capability/);
});

test('removed product promises stay removed', () => {
    for (const phrase of ['PI > 0.70', 'Intact connectome', 'captured by predators', 'NEUROETHOLOGY HANDBOOK',
        'Critical Knobs', 'Expected Phenotype', 'instantly showing you', 'cross if', 'abort if']) {
        assert.equal(guide.includes(phrase), false, phrase);
    }
});

test('no performance claim appears outside the published-research part', () => {
    const outside = text(outsideResearch);
    for (const pattern of PERFORMANCE) assert.doesNotMatch(outside, pattern);
});

test('every published-research row carries a source or says citation needed', () => {
    const body = rows(research).filter((r) => !/<th\b/.test(r));
    assert.ok(body.length >= 8, String(body.length));
    for (const row of body) {
        const t = text(row);
        const cited = /\b(19|20)\d\d\b/.test(t) && /(doi:10\.\d{4,}\/\S+|J Comp Physiol A|Nature|Neuron|eLife|J Neurosci|Nat Neurosci|J Exp Biol)/.test(t);
        assert.ok(cited || /citation needed/.test(t), t.slice(0, 80));
    }
});

test('no sentence claims a demonstrated circuit computation', () => {
    for (const sentence of text(guide).split(/(?<=[.;])\s+/)) {
        if (/demonstrat/i.test(sentence)) assert.match(sentence, /\b(no|not|never)\b/i, sentence);
    }
});

test('every legacy knob is either connected or planned with backlog IDs', () => {
    const KNOB = /shock voltage|dopamine|turn bias|refuge radius|stripe width|moat|laser|drum friction|plume|r\/v ratio|GF threshold|saccad|gap estimation|light phase|clock anticipation|courtship conditioning|wall friction|genotype lesion/i;
    const knobRows = rows(supported).concat(rows(preview)).filter((r) => KNOB.test(text(r)));
    assert.ok(knobRows.length >= 12, String(knobRows.length));
    for (const row of knobRows) {
        if (/data-guide-control=/.test(row)) continue; // pinned by tests/test_science_guide_truth.py
        const m = row.match(/data-planned="([^"]+)"/);
        assert.ok(m, 'knob without planned status: ' + text(row));
        for (const id of m[1].split(/\s+/)) {
            assert.match(id, /^NEXT-\d\d$/);
            assert.ok(backlog.includes(`| ${id} |`), id + ' missing from POST_V04_FEATURES.md');
        }
    }
    for (const cell of guide.match(/<td class="guide-planned"[^>]*>[^<]*<\/td>/g)) {
        assert.match(text(cell), /^Planned — not in v0\.4 \(NEXT-\d\d(, NEXT-\d\d)*\)$/);
    }
});

test('genotype preview card makes no performance or connectome claim', () => {
    const start = app.indexOf('const LESION_INFO =');
    const end = app.indexOf('class ScientificHUD', start);
    const context = {};
    vm.createContext(context);
    vm.runInContext(app.slice(start, end) + ';this.INFO = LESION_INFO;', context);
    for (const [type, info] of Object.entries(context.INFO)) {
        const all = [info.name, info.driver, info.mechanism, info.expectedDeficit].join(' ');
        for (const pattern of PERFORMANCE) assert.doesNotMatch(all, pattern, type);
        assert.match(info.name, /preview/i, type);
        if (type !== 'WT') assert.match(info.expectedDeficit, /^Preview only; no deficit size is predicted\..*\(NEXT-02\)\.$/, type);
    }
    const card = html.slice(html.indexOf('id="lesionInfoCard"'), html.indexOf('id="btnOpenScienceGuide"'));
    assert.doesNotMatch(card, /Expected Deficit|Canton-S|Intact Baseline/);
    for (const pattern of PERFORMANCE) assert.doesNotMatch(text(card), pattern);
});
