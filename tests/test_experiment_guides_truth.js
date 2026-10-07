'use strict';
// The per-assay Experiment Director's Guide describes the engineered browser preview as it is:
// no learned or circuit behaviour the code does not have, no target scores, unverified citations
// labelled, and unsupported biology marked as planned post-v0.4 work (CARD64).

const test = require('node:test');
const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');
const vm = require('node:vm');

const root = path.resolve(__dirname, '..');
const app = fs.readFileSync(path.join(root, 'web/app.js'), 'utf8');
const html = fs.readFileSync(path.join(root, 'web/index.html'), 'utf8');
const backlog = fs.readFileSync(path.join(root, 'docs/POST_V04_FEATURES.md'), 'utf8');

function slice(a, b) {
    const start = app.indexOf(a), end = app.indexOf(b, start);
    assert.ok(start >= 0 && end > start, a);
    return app.slice(start, end);
}
const context = {window: {}, document: {getElementById: () => null}, Math};
vm.createContext(context);
vm.runInContext(slice('const EXPERIMENT_GUIDES', 'class DaemonBridgeClient') + ';this.G = EXPERIMENT_GUIDES;', context);
const GUIDES = context.G;
const texts = (g) => [...g.whatToWatch, ...g.params.map((p) => p.desc)];
const sentences = (t) => t.split(/(?<=[.!;])(?<!\bal\.)\s+/);

const VERIFIED_REFS = [
    'Tully & Quinn (1985) J Comp Physiol A 157:263–277 (abstract read)',
    'Ofstad, Zuker & Reiser (2011) Nature 474:204–207 (cited only for the existence of visual place learning)',
    'von Reyn et al. (2014) Nat Neurosci 17:962–970',
    'Álvarez-Salvado et al. (2018) eLife (recorded, not re-read)',
];

test('all 14 assays have a guide', () => {
    assert.equal(Object.keys(GUIDES).length, 14);
});

test('titles name the assay, not an unimplemented capability', () => {
    for (const [id, g] of Object.entries(GUIDES)) {
        assert.doesNotMatch(g.title, /conditioning|learning|memory|planning|stabili[sz]ation|efference|alternation/i, id);
    }
});

test('removed false mechanism claims stay removed', () => {
    const all = JSON.stringify(Object.values(GUIDES).map((g) => [g.title, g.ref, texts(g)]));
    for (const phrase of ['bilateral antennal gradient', 'Delta7 interneurons promote', 'SAR > 0.60', 'CI > 0.70',
        'shunts >80%', 'Central Complex triggers', 'anticipation peaks followed', 'dopaminergic conditioning',
        'organ wall deflection', 'Giant Fiber fires', 'PAM pain-relief', 'E-PG heading compass', '4.2 mm', '4.2mm',
        'intermittent odor puffs', 'Nature 2014', 'Nature 551', 'Nature 2015', 'Biomimetic']) {
        assert.equal(all.includes(phrase), false, phrase);
    }
});

test('no target score or threshold claim is presented as behaviour', () => {
    for (const [id, g] of Object.entries(GUIDES)) {
        for (const t of texts(g)) {
            assert.doesNotMatch(t, /(^|[^±])[<>≤≥]=?\s*\d/, id + ': ' + t);
            assert.doesNotMatch(t, /\b(SAR|PI|CI)\b/, id + ': ' + t);
        }
    }
});

test('every citation is repo-verified or labelled unverified', () => {
    for (const [id, g] of Object.entries(GUIDES)) {
        if (/no biological reference|Project NeuroFly compact modular/.test(g.ref)) continue;
        assert.match(g.ref, /^Background: /, id);
        let rest = g.ref.slice('Background: '.length);
        for (const v of VERIFIED_REFS) rest = rest.replace(v, '');
        const unverified = /\d{4}\)/.test(rest.replace(/is a handedness study, not an alternation study/, ''));
        if (unverified) assert.match(rest, /\(citation not verified in this repository\)$/, id);
    }
});

test('named neurons and mechanisms appear only negated, as connectome facts or as planned work', () => {
    const MECHANISM = /T4\/T5|HS\/VS|Delta7|DNa02|DNp09|\bP1\b|cVA|Johnston|LPLC2|Giant Fiber|\bGF\b|efference|central.complex|E-PG|\bPAM\b|PPL1|Kenyon|dopamin/i;
    for (const [id, g] of Object.entries(GUIDES)) {
        for (const t of texts(g)) {
            for (const s of sentences(t)) {
                if (!MECHANISM.test(s)) continue;
                assert.match(s, /\b(no|not|planned|proposal)\b|^Connectome:|^von Reyn|are computed|local preview/i, id + ': ' + s);
            }
        }
    }
});

test('planned work names existing backlog IDs', () => {
    for (const [id, g] of Object.entries(GUIDES)) {
        for (const t of texts(g)) {
            if (!/planned|proposal/i.test(t)) continue;
            const ids = t.match(/NEXT-\d\d/g);
            assert.ok(ids, id + ': ' + t);
            for (const n of ids) assert.ok(backlog.includes(`| ${n} |`), n);
        }
    }
});

test('owner rules: negative motion finding, missing lamina input and the imposed encoder are stated', () => {
    const opto = GUIDES.optomotor.whatToWatch.join(' ');
    assert.match(opto, /no T4\/T5 direction selectivity appears/);
    assert.match(opto, /55% of lamina cells have no photoreceptor input/);
    assert.match(opto, /encoder that imposes direction selectivity/);
    assert.match(opto, /hand-set 85%/);
});

test('efference copy stays a post-v0.4 proposal in the backlog', () => {
    const row = backlog.split('\n').find((l) => l.startsWith('| NEXT-06 |'));
    assert.match(row, /saccadic efference copy/);
    assert.match(row, /Efference copy is a proposal/);
    assert.match(html, /saccadic efference copy \(proposal\)/);
});

test('the rendered guide carries the engineered-preview label', () => {
    const elements = {guideTitle: {textContent: ''}, guideRef: {textContent: ''}, guideWhatToWatch: {innerHTML: ''},
        dynamicSlidersContainer: {innerHTML: '', appendChild() {}}};
    const c = {window: {}, document: {getElementById: (id) => elements[id] || null, createElement: () => ({
        className: '', innerHTML: '', querySelector: () => ({addEventListener() {}})})}, Math};
    vm.createContext(c);
    vm.runInContext(slice('const EXPERIMENT_GUIDES', 'class DaemonBridgeClient')
        + ';class H {\n' + slice('    renderExperimentGuide(pid) {', '    setupLesionEvents() {') + '}\nthis.H = H;', c);
    for (const id of Object.keys(GUIDES)) {
        c.H.prototype.renderExperimentGuide.call({arena: {isStandalonePreview: () => true}}, id);
        assert.match(elements.guideWhatToWatch.innerHTML, /^<div><b>Engineered preview, not connectome results\.<\/b><\/div><ul>/, id);
        assert.equal(elements.guideRef.textContent, GUIDES[id].ref);
    }
    assert.match(html, /id="guideWhatToWatch"[^>]*>\s*Engineered preview, not connectome results\./);
});
