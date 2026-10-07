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

// ---- CARD67: catalog badges, preview tool panel, exported metadata, lesion text ----

const CAPABILITY = /conditioning|learning|memory|\bMEM\b|planning|\bPLAN\b|efference|stabili[sz]ation|benchmark|surge-cast|alternation|foreleg|sleep\/wake/i;

test('catalog cards name implemented assays with neutral badges and labelled references', () => {
    const cards = [...html.matchAll(/<div class="experiment-card[^"]*" data-paradigm="([a-z-]+)"[\s\S]*?<span class="exp-card-title">([^<]+)<\/span>\s*<span class="badge[^"]*">([^<]+)<\/span>[\s\S]*?<div class="exp-card-ref">([^<]+)<\/div>/g)]
        .map(([, id, title, badge, ref]) => ({id, title, badge, ref}));
    assert.equal(cards.length, 14);
    assert.deepEqual(cards.map((c) => c.badge), ['BASELINE', 'OLFACTION', 'EXPLORATION', 'THERMAL', 'VISION', 'CLOSED LOOP',
        'ODOUR PLUME', 'LOOMING', 'MOTION', 'GAP', 'ACTIVITY', 'PHEROMONE', 'SLIDING OBSTACLE', 'BODY PROXY']);
    const verified = ['Tully &amp; Quinn (1985) J Comp Physiol A', 'Tully & Quinn (1985) J Comp Physiol A', 'Ofstad, Zuker & Reiser (Nature 2011)',
        'Álvarez-Salvado et al. (2018) eLife', 'von Reyn et al. (2014) Nat Neurosci'];
    for (const c of cards) {
        assert.doesNotMatch(c.title + ' ' + c.badge, CAPABILITY, c.id);
        assert.ok(verified.includes(c.ref) || /unverified$|^Engineered /.test(c.ref), c.id + ': ' + c.ref);
    }
});

function previewContext() {
    const ctx = new Proxy({}, {get: (o, k) => o[k] || (() => {})});
    const canvas = {getContext: () => ctx, getBoundingClientRect: () => ({width: 500, height: 400}), addEventListener() {}};
    const captured = [];
    const c = {window: {addEventListener() {}, devicePixelRatio: 1, hud: {daemonBridge: {connected: false}}},
        document: {getElementById: (id) => id === 'arenaCanvas' ? canvas : null,
            createElement: () => ({click() {}}), body: {appendChild() {}, removeChild() {}}},
        Blob: class { constructor(parts) { captured.push(JSON.parse(parts.join(''))); } },
        URL: {createObjectURL: () => 'blob:x', revokeObjectURL() {}}, setTimeout() { return 1; }, clearTimeout() {}, console};
    vm.createContext(c);
    for (const [a, b] of [['class MushroomBodyCircuit', 'const EXPERIMENT_GUIDES'], ['const EXPERIMENT_GUIDES', 'class DaemonBridgeClient'],
        ['const ASSAY_CONFIGS =', 'const LESION_INFO ='], ['const LESION_INFO =', 'class ScientificHUD'],
        ['class ScientificHUD', '// 9. APPLICATION INITIALIZATION']]) vm.runInContext(slice(a, b), c);
    vm.runInContext('this.A = ScientificBioArena; this.H = ScientificHUD; this.tools = ASSAY_CONFIGS; this.LESION_INFO = LESION_INFO;', c);
    return {c, captured};
}

test('preview tool panel: neutral titles and badges, guide references, no foreleg or JO claims', () => {
    const {c} = previewContext();
    assert.equal(Object.keys(c.tools).length, 14);
    for (const [id, spec] of Object.entries(c.tools)) {
        assert.doesNotMatch(spec.title + ' ' + spec.badge, CAPABILITY, id);
        if (id !== 'multisensory-sandbox') assert.equal(spec.ref, GUIDES[id].ref, id);
        else assert.match(spec.ref, /^Engineered preview: .*not a biological nerve cord\)$/);
        const words = [...(spec.sliders || []).map((s) => s.label + ' ' + s.desc), ...(spec.metrics || []).map((m) => m.label)].join(' | ');
        assert.doesNotMatch(words, /foreleg|Johnston’s organ mechanoreceptors|triangulation by|intermittent|toward cool refuge/i, id);
    }
    assert.ok(c.tools['gap-crossing'].metrics.some((m) => m.label === 'Preview crossing threshold'));
});

// Exact strings exported in the JSON download's metadata.paradigmTitle / metadata.reference.
const NV = ' (citation not verified in this repository)';
const EXPORTED = {
    'open-arena': ['Open Arena Multi-Modal Foraging', 'Background: Budick & Dickinson (2006); Maimon et al. (2010)' + NV],
    't-maze': ['T-Maze Odour Choice', 'Background: Tully & Quinn (1985) J Comp Physiol A 157:263–277 (abstract read); Dudai (1976)' + NV],
    'y-maze': ['Y-Maze Exploration', 'Background: Buchanan et al. (2015) is a handedness study, not an alternation study; Churgin (2017)' + NV],
    'heat-maze': ['Thermal Heat-Maze', 'Background: Ofstad, Zuker & Reiser (2011) Nature 474:204–207 (cited only for the existence of visual place learning)'],
    'buridan': ["Buridan's Paradigm", 'Background: Götz (1980); Colomb et al. (2012)' + NV],
    'visual-operant': ['Visual Operant Flight Simulator', 'Background: Wolf & Heisenberg (1991); Liu et al. (2006)' + NV],
    'wind-tunnel': ['Wind Tunnel Plume', 'Background: Álvarez-Salvado et al. (2018) eLife (recorded, not re-read); Demir et al. (2020)' + NV],
    'looming-escape': ['Looming Escape', 'Background: von Reyn et al. (2014) Nat Neurosci 17:962–970; Card & Dickinson (2008)' + NV],
    'optomotor': ['Optomotor Drum', 'Background: Götz (1964); Kim et al. (2017) on efference copy' + NV],
    'gap-crossing': ['Gap Crossing', 'Background: Pick & Strauss (2005); Triphan et al. (2010)' + NV],
    'circadian-dam': ['Circadian DAM Monitor', 'Background: Konopka & Benzer (1971); Allada & Chung (2010)' + NV],
    'courtship': ['Courtship Chamber', 'Background: Siegel & Hall (1979); Keleman et al. (2007)' + NV],
    'labyrinth': ['Corridor Obstacle Labyrinth', 'Engineered maze with Coulomb sliding contacts (no biological reference)'],
    'multisensory-sandbox': ['Multisensory Sandbox', 'Project NeuroFly compact modular sensorimotor model'],
};

test('JSON download metadata exports the exact neutral title and reference for every assay', () => {
    const {c, captured} = previewContext();
    const arena = new c.A('arenaCanvas');
    for (const [id, [title, ref]] of Object.entries(EXPORTED)) {
        arena.initParadigm(id);
        c.H.prototype.downloadJson.call({arena, daemonBridge: null});
        const meta = captured.at(-1).metadata;
        assert.equal(meta.activeParadigm, id);
        assert.equal(meta.dataSource, 'local_preview');
        assert.equal(meta.paradigmTitle, title, id);
        assert.equal(meta.reference, ref, id);
        assert.equal(GUIDES[id].title, title, id);
        assert.equal(GUIDES[id].ref, ref, id);
    }
    // The constructor's initial Open Arena metadata matches too.
    const fresh = new c.A('arenaCanvas');
    assert.deepEqual([fresh.activeParadigmTitle, fresh.activeParadigmRef], EXPORTED['open-arena']);
});

test('per-frame lesion description stays the selected preview toggle text', () => {
    const {c} = previewContext();
    const block = slice("        const lesionDescription = document.getElementById('lesionCardDesc');", "        const guideRef = document.getElementById('guideRef');");
    const el = {textContent: ''};
    const frame = new vm.Script('(function(panelCaps){' + block + '})');
    for (const type of ['WT', 'DELTA_MB', 'DELTA_CX', 'DELTA_GF', 'DELTA_JO']) {
        const ctx = {document: {getElementById: (id) => id === 'lesionCardDesc' ? el : null}, LESION_INFO: c.LESION_INFO};
        vm.createContext(ctx);
        const fn = frame.runInContext(ctx);
        for (let i = 0; i < 3; i++) {
            fn.call({arena: {lesion: type}}, {graph: false, label: 'Modular'});
            assert.equal(el.textContent, c.LESION_INFO[type].mechanism, type);
        }
    }
    const ctx = {document: {getElementById: () => el}, LESION_INFO: c.LESION_INFO};
    vm.createContext(ctx);
    frame.runInContext(ctx).call({arena: {lesion: 'DELTA_MB'}}, {graph: true, label: 'Connectome (fixed)'});
    assert.match(el.textContent, /cannot alter this graph run/);
});

test('preview optomotor and looming readouts call their mechanisms hand-set or proxies', () => {
    assert.match(app, /'PREVIEW SACCADE \(HAND-SET SLIP CUT\)'/);
    assert.match(app, /Preview HS proxy: \$\{p\.hsFiringRate\.toFixed\(0\)\} Hz \| Hand-set saccade slip cut: 85%/);
    assert.match(app, /Preview Vm proxy: /);
    assert.doesNotMatch(app, /SACCADIC EFFERENCE SHUNT|Efference: 85%/);
});
