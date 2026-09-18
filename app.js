/**
 * FlyBrain Connectome — In-Silico Neuroethology Scientific Instrument
 * ===================================================================
 * Complete biophysical simulation and electrophysiology HUD:
 * 1. Mushroom Body (120 KCs, Huang/Luo Nature 2024 anti-Hebbian rate rule)
 * 2. Central Complex (16-wedge E-PG ring attractor compass + PFL3/PFL2 steering)
 * 3. Descending Locomotion Decoders (DNa02, DNa01, DNp09, BPN, MDN, DNp01/GF)
 * 4. Biomechanical Kuramoto-Hopf Tripod Gait CPG (Cruse Walknet Rule 1)
 * 5. WallSegment 2D Continuous Sliding Collision Physics (Coulomb friction & restitution)
 * 6. 12 Canonical Neuroethological Paradigms + Open Arena:
 *    - open-arena: Free multi-modal foraging
 *    - t-maze: Tully & Quinn (1985) Olfactory Conditioning (CS+/CS-, vacuum airflow, shock grid)
 *    - y-maze: Buchanan et al. (Nature 2015) Spontaneous Alternation & Handedness
 *    - heat-maze: Ofstad, Zuker & Reiser (Nature 2011) Thermal Place Learning (cool refuge at (22, 18), 4 distal landmarks)
 *    - buridan: Götz (1980) Visual Landmark Fixation & Centrophobism (water moat, 2 opposing black stripes)
 *    - visual-operant: Wolf & Heisenberg (1991) Operant Flight Simulator (360° drum, yaw torque, laser heat beam)
 *    - wind-tunnel: Alvarez-Salvado (2018) / Demir (2020) Plume Navigation (laminar flow, surge-and-cast)
 *    - looming-escape: Card & Dickinson (2008) Looming Predator Escape (optical expansion, GF spike threshold)
 *    - optomotor: Götz (1964) / Kim et al. (Cell 2017) Gaze Stabilization (rotating grating drum, saccadic efference copy)
 *    - gap-crossing: Pick & Strauss (Nature 2005) / Triphan (2010) Gap Crossing & Spatial Motor Planning
 *    - circadian-dam: Konopka (1971) / Allada (2010) DAM Sleep/Wake Monitor (16 tubes, mid-tube IR beam break)
 *    - courtship: Siegel & Hall (1979) / Keleman (Nature 2007) Courtship Conditioning & Wing Extension Song
 *    - labyrinth: Multi-junction Obstacle Labyrinth (16 walls, 4 junctions, dead ends, food goal, sliding physics)
 * 7. Real-Time Electrophysiology HUD & Telemetry Blob Exporters (CSV & JSON)
 */

// =============================================================================
// 1. BIOPHYSICAL CONNECTOME MODELS
// =============================================================================

class MushroomBodyCircuit {
    constructor(nKc = 120, nPn = 40, pnPerKc = 5, seed = 42) {
        this.nKc = nKc;
        this.nPn = nPn;
        this.pnPerKc = pnPerKc;
        this.kcThreshold = 0.20;
        this.eta = 0.05;

        let s = seed;
        const rng = () => {
            s = (s * 9301 + 49297) % 233280;
            return s / 233280;
        };

        this.wPnKc = Array.from({ length: this.nKc }, () => {
            const indices = [];
            while (indices.length < this.pnPerKc) {
                const idx = Math.floor(rng() * this.nPn);
                if (!indices.includes(idx)) indices.push(idx);
            }
            return indices;
        });

        this.wBaseline = 1.0;
        this.w = Array.from({ length: this.nKc }, () => [0.0, 0.0]);
        this.u = Array.from({ length: this.nKc }, () => [0.0, 0.0]);
        this.kcFiring = new Float32Array(this.nKc);
        this.pamRate = 0.0;
        this.ppl1Rate = 0.0;
        this.netValence = 0.0;
        this.approachBias = 0.0;
        this.avoidanceBias = 0.0;
        this.plasticityEnabled = true;
    }

    reset(keepMemory = false) {
        this.kcFiring.fill(0);
        this.pamRate = 0.0;
        this.ppl1Rate = 0.0;
        this.netValence = 0.0;
        this.approachBias = 0.0;
        this.avoidanceBias = 0.0;
        if (!keepMemory) {
            for (let i = 0; i < this.nKc; i++) {
                this.w[i][0] = 0.0;
                this.w[i][1] = 0.0;
                this.u[i][0] = 0.0;
                this.u[i][0] = 0.0;
            }
        }
    }

    encodeOdor(odorA, odorB) {
        const pnRates = new Float32Array(this.nPn);
        const halfPn = Math.floor(this.nPn / 2);
        for (let i = 0; i < halfPn; i++) {
            const d = (i - halfPn / 2) / 3.0;
            pnRates[i] = odorA * Math.exp(-d * d) * 50.0;
        }
        for (let i = halfPn; i < this.nPn; i++) {
            const d = (i - 3 * halfPn / 2) / 3.0;
            pnRates[i] = odorB * Math.exp(-d * d) * 50.0;
        }

        for (let k = 0; k < this.nKc; k++) {
            let current = 0;
            const pns = this.wPnKc[k];
            for (let j = 0; j < pns.length; j++) {
                current += (pnRates[pns[j]] / 50.0) / this.pnPerKc;
            }
            if (current > this.kcThreshold) {
                this.kcFiring[k] = 10.0 + 25.0 * (current - this.kcThreshold) / (1.0 - this.kcThreshold);
            } else {
                this.kcFiring[k] = 0.0;
            }
        }
    }

    forward() {
        let sumApp = 0, sumAvo = 0, totalKc = 0;
        for (let i = 0; i < this.nKc; i++) {
            const r = this.kcFiring[i];
            if (r > 0) {
                totalKc += r;
                sumApp += r * Math.max(0.1, this.wBaseline + this.w[i][0]);
                sumAvo += r * Math.max(0.1, this.wBaseline + this.w[i][1]);
            }
        }
        const denom = Math.max(totalKc, 1.0);
        this.approachBias = sumApp / denom;
        this.avoidanceBias = sumAvo / denom;
        this.netValence = Math.max(-1.0, Math.min(1.0, this.approachBias - this.avoidanceBias));
        return this.netValence;
    }

    stepPlasticity(reward, punishment, driveMod = 1.0, dtSec = 0.02) {
        this.pamRate = Math.min(40.0, reward * 40.0);
        this.ppl1Rate = Math.min(40.0, punishment * 40.0);

        if (!this.plasticityEnabled) return;

        for (let i = 0; i < this.nKc; i++) {
            const rKc = this.kcFiring[i];
            if (rKc > 0) {
                if (this.pamRate > 2.0) {
                    const drive = this.eta * (rKc / 35.0) * (this.pamRate / 40.0) * dtSec * 4.0 * driveMod;
                    this.w[i][1] = Math.max(-0.9, this.w[i][1] - drive);
                }
                if (this.ppl1Rate > 2.0) {
                    const drive = this.eta * (rKc / 35.0) * (this.ppl1Rate / 40.0) * dtSec * 4.0 * driveMod;
                    this.w[i][0] = Math.max(-0.9, this.w[i][0] - drive);
                }
            }
        }
    }
}

class CentralComplexCompass {
    constructor(numWedges = 16) {
        this.numWedges = numWedges;
        this.wedgeAngles = Array.from({ length: numWedges }, (_, i) => -Math.PI + (i * 2 * Math.PI) / numWedges);
        this.headingBump = 0.0;
        this.bumpProfile = new Float32Array(numWedges);
        this.pfl3ErrorL = 0.0;
        this.pfl3ErrorR = 0.0;
        this.isLesioned = false;
        this.updateBump(0.0);
    }

    updateBump(heading) {
        this.headingBump = ((heading + Math.PI) % (2 * Math.PI) + 2 * Math.PI) % (2 * Math.PI) - Math.PI;
        for (let i = 0; i < this.numWedges; i++) {
            this.bumpProfile[i] = Math.exp(3.5 * Math.cos(this.wedgeAngles[i] - this.headingBump));
        }
        let maxVal = 0.0001;
        for (let i = 0; i < this.numWedges; i++) if (this.bumpProfile[i] > maxVal) maxVal = this.bumpProfile[i];
        for (let i = 0; i < this.numWedges; i++) this.bumpProfile[i] /= maxVal;
    }

    step(flyYawRate, egocentricWind, dt = 0.02) {
        if (this.isLesioned) {
            this.updateBump(this.headingBump + (Math.random() - 0.5) * 0.8);
            this.pfl3ErrorL = Math.random() * 0.5;
            this.pfl3ErrorR = Math.random() * 0.5;
            return (Math.random() - 0.5) * 2.0;
        }

        const compassDrift = flyYawRate * dt;
        const windBias = 0.15 * Math.sin(egocentricWind) * dt;
        this.updateBump(this.headingBump + compassDrift + windBias);

        const goalError = 0.60 * (0.45 * egocentricWind);
        this.pfl3ErrorL = Math.max(0, -goalError);
        this.pfl3ErrorR = Math.max(0, goalError);
        return goalError;
    }
}

class KuramotoGaitCPG {
    constructor(baseFreq = 8.0) {
        this.baseFreq = baseFreq;
        this.phaseA = 0.0;
        this.phaseB = Math.PI;
        this.steppingFreq = baseFreq;
        this.legStates = { L1: true, L2: false, L3: true, R1: false, R2: true, R3: false };
    }

    step(forwardDrive, isReversing = false, dt = 0.02) {
        this.steppingFreq = Math.min(14.0, Math.max(3.0, this.baseFreq * Math.max(0.2, forwardDrive / 40.0)));
        const phaseSign = isReversing ? -1.0 : 1.0;
        const dphi = phaseSign * 2 * Math.PI * this.steppingFreq * dt;
        this.phaseA = (this.phaseA + dphi) % (2 * Math.PI);
        this.phaseB = (this.phaseA + Math.PI) % (2 * Math.PI);

        const stanceA = Math.sin(this.phaseA) > 0;
        const stanceB = Math.sin(this.phaseB) > 0;
        this.legStates = {
            L1: stanceA, R2: stanceA, L3: stanceA,
            R1: stanceB, L2: stanceB, R3: stanceB
        };
        return this.legStates;
    }
}

// =============================================================================
// 2. GEOMETRIC & CONTINUOUS SLIDING COLLISION PHYSICS (WallSegment)
// =============================================================================

class WallSegment {
    constructor(p1, p2, friction = 0.5, restitution = 0.1) {
        this.p1 = [Number(p1[0]), Number(p1[1])];
        this.p2 = [Number(p2[0]), Number(p2[1])];
        this.friction = Number(friction);
        this.restitution = Number(restitution);

        this.dx = this.p2[0] - this.p1[0];
        this.dy = this.p2[1] - this.p1[1];
        this.lengthSq = this.dx * this.dx + this.dy * this.dy;
        this.length = Math.sqrt(this.lengthSq);

        if (this.length > 1e-12) {
            this.ux = this.dx / this.length;
            this.uy = this.dy / this.length;
            this.nx = -this.uy;
            this.ny = this.ux;
        } else {
            this.ux = 1.0;
            this.uy = 0.0;
            this.nx = 0.0;
            this.ny = 1.0;
        }
    }

    getNormal() {
        return [this.nx, this.ny];
    }

    projectPoint(x, y) {
        if (this.lengthSq < 1e-12) {
            return [this.p1[0], this.p1[1], 0.0];
        }
        const vx = x - this.p1[0];
        const vy = y - this.p1[1];
        const t = (vx * this.dx + vy * this.dy) / this.lengthSq;
        const tClamped = Math.max(0.0, Math.min(1.0, t));
        const cx = this.p1[0] + tClamped * this.dx;
        const cy = this.p1[1] + tClamped * this.dy;
        return [cx, cy, tClamped];
    }

    distanceToPoint(x, y) {
        const [cx, cy] = this.projectPoint(x, y);
        const dx = x - cx;
        const dy = y - cy;
        return Math.sqrt(dx * dx + dy * dy);
    }

    resolveCircleCollision(x, y, vx, vy, radius) {
        const [cx, cy, t] = this.projectPoint(x, y);
        const dx = x - cx;
        const dy = y - cy;
        const dist = Math.sqrt(dx * dx + dy * dy);

        if (dist >= radius) {
            return { x, y, vx, vy, collided: false, normal: [0.0, 0.0], t };
        }

        const penetration = radius - dist;
        let cnx, cny;
        if (dist > 1e-8) {
            cnx = dx / dist;
            cny = dy / dist;
        } else {
            cnx = this.nx;
            cny = this.ny;
        }

        const resolvedX = x + cnx * penetration;
        const resolvedY = y + cny * penetration;

        const vDotN = vx * cnx + vy * cny;
        let newVx = vx;
        let newVy = vy;

        if (vDotN < 0.0) {
            const vNormalMag = -this.restitution * vDotN;
            const vtx = vx - vDotN * cnx;
            const vty = vy - vDotN * cny;
            const frictionFactor = Math.max(0.0, 1.0 - this.friction);
            const vtxNew = vtx * frictionFactor;
            const vtyNew = vty * frictionFactor;
            newVx = vNormalMag * cnx + vtxNew;
            newVy = vNormalMag * cny + vtyNew;
        }

        return { x: resolvedX, y: resolvedY, vx: newVx, vy: newVy, collided: true, normal: [cnx, cny], t };
    }
}

// =============================================================================
// 3. SCIENTIFIC BIO-ARENA & PARADIGM BATTERY CONTROLLER
// =============================================================================

class ScientificBioArena {
    constructor(canvasId) {
        this.canvas = document.getElementById(canvasId);
        this.ctx = this.canvas.getContext('2d');

        this.resize();
        window.addEventListener('resize', () => this.resize());

        this.activeParadigmId = 'open-arena';
        this.activeParadigmTitle = 'Open Arena Multi-Modal Assay';
        this.activeParadigmRef = 'General Neuroethology Open Arena with multi-sensory foraging';
        this.currentTrial = 1;
        this.paradigmElapsedSec = 0.0;
        this.paradigmStatus = 'FORAGING';

        this.worldBounds = { minX: -150, maxX: 150, minY: -110, maxY: 110 };

        this.fly = {
            x: 0.0,
            y: 0.0,
            heading: 0.0,
            speed: 0.0,
            yawRate: 0.0,
            energy: 1.0,
            radius: 3.5,
            trail: []
        };

        this.mb = new MushroomBodyCircuit(120, 40, 5, 42);
        this.cx = new CentralComplexCompass(16);
        this.cpg = new KuramotoGaitCPG(8.0);

        this.dn = {
            dna02L: 0.0, dna02R: 0.0, dna02Diff: 0.0,
            dnp09: 0.0, bpn: 22.0, mdn: 0.0,
            dnp01Gf: 0, escapeActive: false, escapeTimer: 0.0
        };

        this.lesion = 'WT';
        this.gfLesioned = false;
        this.joLesioned = false;

        this.windVector = [-15.0, 0.0];
        this.foodItems = [{ x: 90.0, y: 0.0, radius: 14.0, odorStrength: 1.0 }];
        this.alarms = [];
        this.predators = [{ x: 110.0, y: 60.0, vx: -18.0, vy: -12.0, radius: 8.0, active: true }];
        this.particles = Array.from({ length: 55 }, () => ({
            x: (Math.random() - 0.5) * 280,
            y: (Math.random() - 0.5) * 200,
            speed: 0.7 + Math.random() * 0.6,
            life: Math.random() * 100
        }));

        this.currentWalls = [];
        this.paradigmState = {};
        this.collisionNormals = [];

        this.isRecording = false;
        this.telemetryBuffer = [];
        this.simTime = 0.0;
        this.stepCount = 0;

        this.toolMode = 'select';
        this.setupMouseEvents();
        this.initParadigm('open-arena');
    }

    resize() {
        const rect = this.canvas.getBoundingClientRect();
        this.canvas.width = rect.width * (window.devicePixelRatio || 1);
        this.canvas.height = rect.height * (window.devicePixelRatio || 1);
        this.ctx.setTransform(1, 0, 0, 1, 0, 0);
        this.ctx.scale(window.devicePixelRatio || 1, window.devicePixelRatio || 1);
        this.viewWidth = rect.width;
        this.viewHeight = rect.height;
    }

    worldToScreen(wx, wy) {
        const spanX = this.worldBounds.maxX - this.worldBounds.minX;
        const spanY = this.worldBounds.maxY - this.worldBounds.minY;
        const midX = (this.worldBounds.minX + this.worldBounds.maxX) / 2;
        const midY = (this.worldBounds.minY + this.worldBounds.maxY) / 2;
        const margin = 32;
        const scale = Math.min((this.viewWidth - margin * 2) / spanX, (this.viewHeight - margin * 2) / spanY);
        const sx = this.viewWidth / 2 + (wx - midX) * scale;
        const sy = this.viewHeight / 2 - (wy - midY) * scale;
        return { x: sx, y: sy, scale };
    }

    screenToWorld(sx, sy) {
        const spanX = this.worldBounds.maxX - this.worldBounds.minX;
        const spanY = this.worldBounds.maxY - this.worldBounds.minY;
        const midX = (this.worldBounds.minX + this.worldBounds.maxX) / 2;
        const midY = (this.worldBounds.minY + this.worldBounds.maxY) / 2;
        const margin = 32;
        const scale = Math.min((this.viewWidth - margin * 2) / spanX, (this.viewHeight - margin * 2) / spanY);
        const wx = midX + (sx - this.viewWidth / 2) / scale;
        const wy = midY - (sy - this.viewHeight / 2) / scale;
        return { x: wx, y: wy };
    }

    setupMouseEvents() {
        this.canvas.addEventListener('mousedown', (e) => {
            const rect = this.canvas.getBoundingClientRect();
            const mouseX = e.clientX - rect.left;
            const mouseY = e.clientY - rect.top;
            const worldPos = this.screenToWorld(mouseX, mouseY);

            if (this.toolMode === 'food') {
                this.foodItems.push({ x: worldPos.x, y: worldPos.y, radius: 12.0, odorStrength: 1.0 });
            } else if (this.toolMode === 'alarm') {
                this.alarms.push({ x: worldPos.x, y: worldPos.y, radius: 25.0, strength: 1.0, life: 1.0 });
            } else if (this.toolMode === 'predator') {
                const angle = Math.atan2(this.fly.y - worldPos.y, this.fly.x - worldPos.x);
                this.predators.push({
                    x: worldPos.x, y: worldPos.y,
                    vx: Math.cos(angle) * 22.0, vy: Math.sin(angle) * 22.0,
                    radius: 8.0, active: true
                });
            } else if (this.toolMode === 'wind') {
                const angle = Math.atan2(worldPos.y, worldPos.x);
                this.windVector = [-Math.cos(angle) * 15.0, -Math.sin(angle) * 15.0];
            }
        });
    }

    setLesion(type) {
        this.lesion = type;
        this.cx.isLesioned = (type === 'DELTA_CX');
        this.mb.plasticityEnabled = (type !== 'DELTA_MB');
        this.gfLesioned = (type === 'DELTA_GF');
        this.joLesioned = (type === 'DELTA_JO');
    }

    initParadigm(paradigmId) {
        this.activeParadigmId = paradigmId;
        this.paradigmElapsedSec = 0.0;
        this.fly.trail = [];
        this.collisionNormals = [];
        this.currentWalls = [];

        switch (paradigmId) {
            case 'open-arena':
                this.activeParadigmTitle = 'Open Arena Multi-Modal Assay';
                this.activeParadigmRef = 'General Neuroethology Open Arena with multi-sensory foraging';
                this.worldBounds = { minX: -150, maxX: 150, minY: -110, maxY: 110 };
                this.fly.x = 0; this.fly.y = 0; this.fly.heading = 0; this.fly.speed = 0;
                this.windVector = [-15.0, 0.0];
                this.paradigmStatus = 'FORAGING';
                this.paradigmState = {};
                break;

            case 't-maze':
                this.activeParadigmTitle = 'T-Maze Olfactory Associative Conditioning';
                this.activeParadigmRef = 'Tully & Quinn (1985) Science / Cell Pavlovian Conditioning';
                this.worldBounds = { minX: 0, maxX: 140, minY: 0, maxY: 80 };
                this.fly.x = 70.0; this.fly.y = 15.0; this.fly.heading = Math.PI / 2; this.fly.speed = 10.0;
                this.paradigmStatus = 'ASCENDING STEM';
                this.currentWalls = [
                    new WallSegment([63.0, 10.0], [77.0, 10.0]),
                    new WallSegment([63.0, 10.0], [63.0, 43.0]),
                    new WallSegment([77.0, 10.0], [77.0, 43.0]),
                    new WallSegment([10.0, 43.0], [63.0, 43.0]),
                    new WallSegment([10.0, 43.0], [10.0, 57.0]),
                    new WallSegment([10.0, 57.0], [130.0, 57.0]),
                    new WallSegment([130.0, 43.0], [130.0, 57.0]),
                    new WallSegment([77.0, 43.0], [130.0, 43.0]),
                ];
                this.paradigmState = {
                    csPlusArm: 'arm_a',
                    choiceCounts: { arm_a: 0, arm_b: 0 },
                    firstChoice: null,
                    latencyMs: null,
                    performanceIndex: 0.0,
                    shockPulse: 0.0
                };
                break;

            case 'y-maze':
                this.activeParadigmTitle = 'Y-Maze Spontaneous Alternation & Handedness';
                this.activeParadigmRef = 'Buchanan, Kain & de Bivort (Nature 2015)';
                this.worldBounds = { minX: 0, maxX: 120, minY: 0, maxY: 120 };
                this.fly.x = 60.0; this.fly.y = 60.0; this.fly.heading = Math.PI / 2; this.fly.speed = 10.0;
                this.paradigmStatus = 'IN HUB';
                this.currentWalls = [
                    new WallSegment([66.0, 100.0], [54.0, 100.0]),
                    new WallSegment([22.4, 45.2], [28.4, 34.8]),
                    new WallSegment([91.6, 34.8], [97.6, 45.2]),
                    new WallSegment([66.0, 100.0], [51.3, 65.0]),
                    new WallSegment([51.3, 65.0], [28.4, 34.8]),
                    new WallSegment([22.4, 45.2], [60.0, 50.0]),
                    new WallSegment([60.0, 50.0], [97.6, 45.2]),
                    new WallSegment([91.6, 34.8], [68.7, 65.0]),
                    new WallSegment([68.7, 65.0], [54.0, 100.0]),
                ];
                this.paradigmState = {
                    armCounts: [0, 0, 0],
                    armSequence: [],
                    turnDirections: [],
                    lastZone: 'hub',
                    sar: 0.0,
                    handedness: 0.0
                };
                break;

            case 'heat-maze':
                this.activeParadigmTitle = 'Thermal Heat-Maze Place Learning';
                this.activeParadigmRef = 'Ofstad, Zuker & Reiser (Nature 2011)';
                this.worldBounds = { minX: 0, maxX: 120, minY: 0, maxY: 120 };
                this.fly.x = 50.0; this.fly.y = 50.0; this.fly.heading = 0.8; this.fly.speed = 10.0;
                this.paradigmStatus = 'HOT FLOOR (36.5°C)';
                const wallsHm = [];
                const cx = 60.0, cy = 60.0, rHm = 55.0;
                for (let i = 0; i < 32; i++) {
                    const a1 = (2 * Math.PI * i) / 32;
                    const a2 = (2 * Math.PI * (i + 1)) / 32;
                    wallsHm.push(new WallSegment(
                        [cx + rHm * Math.cos(a1), cy + rHm * Math.sin(a1)],
                        [cx + rHm * Math.cos(a2), cy + rHm * Math.sin(a2)]
                    ));
                }
                this.currentWalls = wallsHm;
                this.paradigmState = {
                    refugePos: [82.0, 78.0],
                    refugeRadius: 9.0,
                    refugeReached: false,
                    escapeLatencyMs: null,
                    cumulativeDose: 0.0,
                    temp: 36.5
                };
                break;

            case 'buridan':
                this.activeParadigmTitle = "Buridan's Visual Landmark Fixation & Centrophobism";
                this.activeParadigmRef = "Götz (1980); Colomb & Brembs (2012)";
                this.worldBounds = { minX: 0, maxX: 120, minY: 0, maxY: 120 };
                this.fly.x = 60.0; this.fly.y = 60.0; this.fly.heading = 0.0; this.fly.speed = 10.0;
                this.paradigmStatus = 'STRIPE FIXATION';
                this.paradigmState = {
                    center: [60.0, 60.0],
                    platformRadius: 50.0,
                    stripes: [{ x: 110.0, y: 60.0, deg: 0 }, { x: 10.0, y: 60.0, deg: 180 }],
                    timeCenterMs: 0.0,
                    timePerimeterMs: 0.0,
                    fixationScores: [],
                    stripeCrossings: 0,
                    lastSide: null,
                    centrophobism: 1.0,
                    meanFixation: 0.0
                };
                break;

            case 'visual-operant':
                this.activeParadigmTitle = 'Visual Operant Flight Simulator (Yaw Conditioning)';
                this.activeParadigmRef = 'Wolf & Heisenberg (1991) J. Comp. Physiol. A';
                this.worldBounds = { minX: 0, maxX: 80, minY: 0, maxY: 80 };
                this.fly.x = 40.0; this.fly.y = 40.0; this.fly.heading = 0.0; this.fly.speed = 0.0;
                this.paradigmStatus = 'SAFE QUADRANT (T)';
                this.paradigmState = {
                    drumAngleDeg: 0.0,
                    couplingGain: 120.0,
                    timeSafeMs: 0.0,
                    timePunishedMs: 0.0,
                    laserActive: false,
                    learningIndex: 0.0,
                    yawTorque: 0.0
                };
                break;

            case 'wind-tunnel':
                this.activeParadigmTitle = 'Wind Tunnel Odor Plume Tracking (Surge-and-Cast)';
                this.activeParadigmRef = 'Alvarez-Salvado et al. (2018); Demir et al. (2020)';
                this.worldBounds = { minX: 0, maxX: 200, minY: 0, maxY: 60 };
                this.fly.x = 25.0; this.fly.y = 30.0; this.fly.heading = 0.0; this.fly.speed = 10.0;
                this.paradigmStatus = 'SEARCHING (CAST)';
                this.currentWalls = [
                    new WallSegment([0.0, 0.0], [200.0, 0.0]),
                    new WallSegment([0.0, 60.0], [200.0, 60.0]),
                    new WallSegment([0.0, 0.0], [0.0, 60.0]),
                    new WallSegment([200.0, 0.0], [200.0, 60.0])
                ];
                this.paradigmState = {
                    nozzlePos: [180.0, 30.0],
                    windFlow: [-25.0, 0.0],
                    filamentSigma: 3.5,
                    surgeSteps: 0,
                    castSteps: 0,
                    sourceReached: false,
                    timeToSourceMs: null,
                    upwindProgress: 0.0,
                    behavioralState: 'CAST'
                };
                break;

            case 'looming-escape':
                this.activeParadigmTitle = 'Visual Looming Predator Escape & Takeoff Assay';
                this.activeParadigmRef = 'Card & Dickinson (PNAS 2008) Giant Fiber Looming';
                this.worldBounds = { minX: 0, maxX: 80, minY: 0, maxY: 80 };
                this.fly.x = 40.0; this.fly.y = 40.0; this.fly.heading = 0.0; this.fly.speed = 0.0;
                this.paradigmStatus = 'APPROACHING THREAT';
                this.paradigmState = {
                    tCollisionS: 1.2,
                    rOverVS: 0.025,
                    gfThresholdRad: (65.0 * Math.PI) / 180.0,
                    escapeInitiated: false,
                    timeToCollisionJumpMs: null,
                    loomingSizeJumpDeg: null,
                    gfSpike: false,
                    thetaDeg: 4.0,
                    vm: -70.0
                };
                break;

            case 'optomotor':
                this.activeParadigmTitle = 'Optomotor Gaze Stabilization & Saccadic Efference Copy';
                this.activeParadigmRef = 'Götz (1964); Kim, Fenk, Lyu & Maimon (Cell 2017)';
                this.worldBounds = { minX: 0, maxX: 90, minY: 0, maxY: 90 };
                this.fly.x = 45.0; this.fly.y = 45.0; this.fly.heading = 0.0; this.fly.speed = 0.0;
                this.paradigmStatus = 'GAZE STABILIZING';
                this.paradigmState = {
                    drumVelocityDegS: 30.0,
                    drumAngleDeg: 0.0,
                    hsFiringRate: 40.0,
                    retinalSlip: 0.0,
                    effectiveSlip: 0.0,
                    efferenceCopyActive: false,
                    gain: 0.88,
                    saccadeTimer: 0.0
                };
                break;

            case 'gap-crossing':
                this.activeParadigmTitle = 'Gap Crossing & Spatial Motor Planning';
                this.activeParadigmRef = 'Pick & Strauss (Nature 2005); Triphan (2010)';
                this.worldBounds = { minX: 0, maxX: 100, minY: 0, maxY: 20 };
                this.fly.x = 12.0; this.fly.y = 10.0; this.fly.heading = 0.0; this.fly.speed = 8.0;
                this.paradigmStatus = 'APPROACHING CHASM';
                this.paradigmState = {
                    gapWidthMm: 3.5,
                    reachabilityThreshMm: 3.8,
                    probingDurationMs: 0.0,
                    decisionOutcome: null,
                    crossingSuccess: false,
                    isProbing: false
                };
                break;

            case 'circadian-dam':
                this.activeParadigmTitle = 'Circadian Locomotor Sleep/Wake DAM Monitor';
                this.activeParadigmRef = 'Konopka & Benzer (1971); Allada & Siegel (2010)';
                this.worldBounds = { minX: 0, maxX: 65, minY: 0, maxY: 160 };
                this.fly.x = 10.0; this.fly.y = 75.0; this.fly.heading = 0.0; this.fly.speed = 6.0;
                this.paradigmStatus = 'LOCOMOTING [AWAKE]';
                this.paradigmState = {
                    activeTube: 7,
                    beamCrossings: 0,
                    consecutiveImmobileMin: 0.0,
                    totalSleepMin: 0.0,
                    sleepBouts: 0,
                    inSleepBout: false,
                    lastX: 10.0,
                    actogramHistory: []
                };
                break;

            case 'courtship':
                this.activeParadigmTitle = 'Courtship Conditioning, Wing Song & cVA Suppression';
                this.activeParadigmRef = 'Siegel & Hall (1979); Keleman et al. (Nature 2007)';
                this.worldBounds = { minX: 0, maxX: 20, minY: 0, maxY: 20 };
                this.fly.x = 7.0; this.fly.y = 9.0; this.fly.heading = 0.2; this.fly.speed = 6.0;
                this.paradigmStatus = 'SEARCHING FOR FEMALE';
                this.paradigmState = {
                    chamberCenter: [10.0, 10.0],
                    chamberRadius: 5.0,
                    femalePos: [11.5, 11.0],
                    femaleType: 'mated',
                    courtshipActiveSteps: 0,
                    totalSteps: 0,
                    rejectionKicks: 0,
                    wingAngleDeg: 0.0,
                    courtshipIndex: 0.0
                };
                break;

            case 'labyrinth':
                this.activeParadigmTitle = 'Corridor Obstacle Labyrinth (Sliding Physics & Goal)';
                this.activeParadigmRef = 'Continuous Sliding Physics (Coulomb mu=0.5, eps=0.1)';
                this.worldBounds = { minX: 0, maxX: 140, minY: 0, maxY: 100 };
                this.fly.x = 12.0; this.fly.y = 15.0; this.fly.heading = Math.PI / 2; this.fly.speed = 8.0;
                this.paradigmStatus = 'NAVIGATING MAZE';
                this.currentWalls = [
                    new WallSegment([0.0, 0.0], [140.0, 0.0]),
                    new WallSegment([140.0, 0.0], [140.0, 100.0]),
                    new WallSegment([140.0, 100.0], [0.0, 100.0]),
                    new WallSegment([0.0, 100.0], [0.0, 0.0]),
                    new WallSegment([25.0, 0.0], [25.0, 70.0]),
                    new WallSegment([25.0, 70.0], [50.0, 70.0]),
                    new WallSegment([50.0, 30.0], [50.0, 70.0]),
                    new WallSegment([50.0, 30.0], [75.0, 30.0]),
                    new WallSegment([75.0, 0.0], [75.0, 55.0]),
                    new WallSegment([75.0, 55.0], [100.0, 55.0]),
                    new WallSegment([100.0, 20.0], [100.0, 55.0]),
                    new WallSegment([100.0, 20.0], [125.0, 20.0]),
                    new WallSegment([125.0, 20.0], [125.0, 70.0]),
                    new WallSegment([100.0, 80.0], [140.0, 80.0]),
                    new WallSegment([50.0, 85.0], [100.0, 85.0]),
                    new WallSegment([100.0, 70.0], [100.0, 80.0])
                ];
                this.paradigmState = {
                    goalPos: [130.0, 85.0],
                    goalReached: false,
                    timeToGoalMs: null,
                    wallCollisions: 0,
                    deadEndEntries: 0,
                    pathLength: 0.0,
                    tortuosity: 1.0,
                    lastPathX: 12.0,
                    lastPathY: 15.0
                };
                break;
        }
    }

    resetTrial() {
        this.currentTrial += 1;
        this.initParadigm(this.activeParadigmId);
    }

    step(dt = 0.02) {
        this.simTime += dt;
        this.stepCount += 1;
        this.paradigmElapsedSec += dt;

        let rewardSignal = 0.0;
        let punishmentSignal = 0.0;
        let odorA = 0.0, odorB = 0.0;
        let egocentricWind = 0.0;
        let loomingTrigger = false;
        let minThreatDist = 999.0;

        this.collisionNormals = [];
        const p = this.paradigmState;

        switch (this.activeParadigmId) {
            case 'open-arena': {
                const antDist = 1.8;
                const leftAnt = { x: this.fly.x + antDist * Math.cos(this.fly.heading + 0.35), y: this.fly.y + antDist * Math.sin(this.fly.heading + 0.35) };
                const rightAnt = { x: this.fly.x + antDist * Math.cos(this.fly.heading - 0.35), y: this.fly.y + antDist * Math.sin(this.fly.heading - 0.35) };

                for (const f of this.foodItems) {
                    const dl = Math.hypot(f.x - leftAnt.x, f.y - leftAnt.y);
                    const dr = Math.hypot(f.x - rightAnt.x, f.y - rightAnt.y);
                    odorA += f.odorStrength * (Math.exp(-(dl * dl) / 2500) + Math.exp(-(dr * dr) / 2500)) * 0.5;
                    if (Math.hypot(f.x - this.fly.x, f.y - this.fly.y) < f.radius + this.fly.radius) {
                        rewardSignal = 1.0;
                        this.fly.energy = Math.min(1.0, this.fly.energy + 0.15);
                    }
                }

                for (let i = this.alarms.length - 1; i >= 0; i--) {
                    const a = this.alarms[i];
                    const dl = Math.hypot(a.x - leftAnt.x, a.y - leftAnt.y);
                    const dr = Math.hypot(a.x - rightAnt.x, a.y - rightAnt.y);
                    odorB += a.strength * (Math.exp(-(dl * dl) / 1600) + Math.exp(-(dr * dr) / 1600)) * 0.5;
                    a.life -= dt * 0.15;
                    if (a.life <= 0) this.alarms.splice(i, 1);
                }

                const upwindAngle = Math.atan2(-this.windVector[1], -this.windVector[0]);
                egocentricWind = this.joLesioned ? 0 : (((upwindAngle - this.fly.heading + Math.PI) % (2 * Math.PI) + 2 * Math.PI) % (2 * Math.PI) - Math.PI);

                for (const pred of this.predators) {
                    if (!pred.active) continue;
                    pred.x += pred.vx * dt; pred.y += pred.vy * dt;
                    if (Math.abs(pred.x) > 140) pred.vx *= -1;
                    if (Math.abs(pred.y) > 100) pred.vy *= -1;
                    const d = Math.hypot(pred.x - this.fly.x, pred.y - this.fly.y);
                    minThreatDist = Math.min(minThreatDist, d);
                    const relVel = -((pred.x - this.fly.x) * pred.vx + (pred.y - this.fly.y) * pred.vy) / (d + 1e-5);
                    if (d < 75 && relVel > 10) loomingTrigger = true;
                    if (d < pred.radius + this.fly.radius) {
                        punishmentSignal = 1.0;
                        this.alarms.push({ x: this.fly.x, y: this.fly.y, radius: 25.0, strength: 1.0, life: 1.0 });
                    }
                }
                break;
            }

            case 't-maze': {
                const da = Math.hypot(this.fly.x - 15.0, this.fly.y - 50.0);
                const db = Math.hypot(this.fly.x - 125.0, this.fly.y - 50.0);
                const concA = Math.exp(-(da * da) / (2 * 22 * 22));
                const concB = Math.exp(-(db * db) / (2 * 22 * 22));
                odorA = concA;
                odorB = concB;

                const upwindAngle = (this.fly.y < 45) ? (Math.PI / 2) : (this.fly.x < 70 ? 0 : Math.PI);
                egocentricWind = (((upwindAngle - this.fly.heading + Math.PI) % (2 * Math.PI) + 2 * Math.PI) % (2 * Math.PI) - Math.PI);

                if (this.fly.y >= 43.0 && this.fly.y <= 57.0) {
                    if (this.fly.x < 45.0) {
                        if (p.firstChoice === null) {
                            p.firstChoice = 'arm_a';
                            p.latencyMs = this.paradigmElapsedSec * 1000;
                        }
                        p.choiceCounts.arm_a += 1;
                        rewardSignal = 1.0;
                        this.paradigmStatus = 'ARM A (CS+ SUCROSE)';
                    } else if (this.fly.x > 95.0) {
                        if (p.firstChoice === null) {
                            p.firstChoice = 'arm_b';
                            p.latencyMs = this.paradigmElapsedSec * 1000;
                        }
                        p.choiceCounts.arm_b += 1;
                        punishmentSignal = 1.0;
                        p.shockPulse = 1.0;
                        this.paradigmStatus = 'ARM B (CS- SHOCK 60V)';
                    }
                }
                p.shockPulse = Math.max(0, p.shockPulse - dt * 2.0);
                const totalC = p.choiceCounts.arm_a + p.choiceCounts.arm_b;
                p.performanceIndex = totalC > 0 ? (p.choiceCounts.arm_a - p.choiceCounts.arm_b) / totalC : 0.0;
                break;
            }

            case 'y-maze': {
                const cx = 60.0, cy = 60.0;
                const dHub = Math.hypot(this.fly.x - cx, this.fly.y - cy);
                let currentArm = -1;
                if (dHub > 14.0) {
                    const ang = Math.atan2(this.fly.y - cy, this.fly.x - cx);
                    const angDeg = ((ang * 180 / Math.PI) + 360) % 360;
                    if (angDeg >= 45 && angDeg < 150) currentArm = 0;
                    else if (angDeg >= 150 && angDeg < 270) currentArm = 1;
                    else currentArm = 2;
                }

                if (currentArm !== -1 && currentArm !== p.lastZone) {
                    p.armCounts[currentArm] += 1;
                    p.armSequence.push(currentArm);
                    if (p.armSequence.length >= 2) {
                        const prev = p.armSequence[p.armSequence.length - 2];
                        const diff = (currentArm - prev + 3) % 3;
                        p.turnDirections.push(diff === 1 ? 'L' : 'R');
                    }
                    if (p.armSequence.length >= 3) {
                        let alternatingTriads = 0;
                        const totalTriads = p.armSequence.length - 2;
                        for (let i = 0; i < totalTriads; i++) {
                            const t1 = p.armSequence[i], t2 = p.armSequence[i + 1], t3 = p.armSequence[i + 2];
                            if (t1 !== t2 && t2 !== t3 && t1 !== t3) alternatingTriads++;
                        }
                        p.sar = alternatingTriads / totalTriads;
                    }
                    p.lastZone = currentArm;
                    this.paradigmStatus = `ARM ${currentArm} ENTERED`;
                } else if (dHub <= 12.0) {
                    p.lastZone = 'hub';
                    this.paradigmStatus = 'IN DECISION HUB';
                }
                break;
            }

            case 'heat-maze': {
                const dxRef = this.fly.x - p.refugePos[0];
                const dyRef = this.fly.y - p.refugePos[1];
                const distRef = Math.hypot(dxRef, dyRef);

                if (distRef <= p.refugeRadius) {
                    p.temp = 24.0;
                    if (!p.refugeReached) {
                        p.refugeReached = true;
                        p.escapeLatencyMs = this.paradigmElapsedSec * 1000;
                    }
                    rewardSignal = 1.0;
                    this.paradigmStatus = 'COOL REFUGE (24.0°C)';
                } else {
                    const excess = distRef - p.refugeRadius;
                    const gauss = Math.exp(-(excess * excess) / (2 * 8.0 * 8.0));
                    p.temp = 36.5 - (36.5 - 24.0) * gauss;
                    punishmentSignal = Math.max(0.0, (p.temp - 25.0) / 11.5);
                    p.cumulativeDose += Math.max(0.0, p.temp - 25.0) * dt;
                    this.paradigmStatus = `HOT FLOOR (${p.temp.toFixed(1)}°C)`;
                }
                break;
            }

            case 'buridan': {
                const dxC = this.fly.x - p.center[0];
                const dyC = this.fly.y - p.center[1];
                const distC = Math.hypot(dxC, dyC);

                if (distC < 25.0) {
                    p.timeCenterMs += dt * 1000;
                } else {
                    p.timePerimeterMs += dt * 1000;
                }
                const totalT = Math.max(1, p.timeCenterMs + p.timePerimeterMs);
                p.centrophobism = 1.0 - (p.timeCenterMs / totalT);

                const b0 = Math.atan2(60.0 - this.fly.y, 110.0 - this.fly.x) - this.fly.heading;
                const b180 = Math.atan2(60.0 - this.fly.y, 10.0 - this.fly.x) - this.fly.heading;
                const normB0 = ((b0 + Math.PI) % (2 * Math.PI) + 2 * Math.PI) % (2 * Math.PI) - Math.PI;
                const normB180 = ((b180 + Math.PI) % (2 * Math.PI) + 2 * Math.PI) % (2 * Math.PI) - Math.PI;
                const minDev = Math.min(Math.abs(normB0), Math.abs(normB180));
                const fix = Math.cos(minDev);
                p.fixationScores.push(fix);
                if (p.fixationScores.length > 200) p.fixationScores.shift();
                p.meanFixation = p.fixationScores.reduce((a, b) => a + b, 0) / p.fixationScores.length;

                if (distC > 48.0) {
                    const steerBack = Math.atan2(p.center[1] - this.fly.y, p.center[0] - this.fly.x);
                    this.fly.heading = steerBack;
                }

                const side = Math.abs(normB0) < Math.PI / 2 ? 0 : 1;
                if (p.lastSide !== null && p.lastSide !== side) p.stripeCrossings++;
                p.lastSide = side;
                this.paradigmStatus = distC < 25.0 ? 'OPEN CENTER' : 'PERIMETER STRIPE TRAIL';
                break;
            }

            case 'visual-operant': {
                this.fly.x = 40.0;
                this.fly.y = 40.0;
                p.yawTorque = this.fly.yawRate;

                const omega = -p.couplingGain * p.yawTorque * 0.05;
                p.drumAngleDeg = ((p.drumAngleDeg + omega * dt * 10.0) % 360 + 360) % 360;

                const quad = Math.floor(p.drumAngleDeg / 90);
                const isPunished = (quad === 1 || quad === 3);
                p.laserActive = isPunished;

                if (isPunished) {
                    p.timePunishedMs += dt * 1000;
                    punishmentSignal = 1.0;
                    this.paradigmStatus = 'PUNISHED QUADRANT (⊥ LASER)';
                } else {
                    p.timeSafeMs += dt * 1000;
                    this.paradigmStatus = 'SAFE QUADRANT (T)';
                }

                const totOperant = Math.max(1, p.timeSafeMs + p.timePunishedMs);
                p.learningIndex = (p.timeSafeMs - p.timePunishedMs) / totOperant;
                break;
            }

            case 'wind-tunnel': {
                if (this.fly.x <= p.nozzlePos[0] + 5.0) {
                    const dy = this.fly.y - p.nozzlePos[1];
                    const conc = Math.exp(-(dy * dy) / (2 * p.filamentSigma * p.filamentSigma)) * Math.exp(-Math.max(0, p.nozzlePos[0] - this.fly.x) / 250);
                    odorA = conc;
                } else {
                    odorA = 0.0;
                }

                const upwindAngle = Math.PI;
                egocentricWind = (((upwindAngle - this.fly.heading + Math.PI) % (2 * Math.PI) + 2 * Math.PI) % (2 * Math.PI) - Math.PI);

                if (odorA > 0.05) {
                    p.surgeSteps += 1;
                    p.behavioralState = 'SURGE';
                    this.paradigmStatus = 'SURGING UPWIND';
                    rewardSignal = 0.5;
                } else {
                    p.castSteps += 1;
                    p.behavioralState = 'CAST';
                    this.paradigmStatus = 'CASTING CROSSWIND';
                }

                p.upwindProgress = this.fly.x - 25.0;
                if (Math.hypot(this.fly.x - p.nozzlePos[0], this.fly.y - p.nozzlePos[1]) < 12.0) {
                    if (!p.sourceReached) {
                        p.sourceReached = true;
                        p.timeToSourceMs = this.paradigmElapsedSec * 1000;
                    }
                    rewardSignal = 1.0;
                    this.paradigmStatus = 'SOURCE REACHED!';
                }
                break;
            }

            case 'looming-escape': {
                this.fly.x = 40.0;
                this.fly.y = 40.0;
                const timeToColl = Math.max(0.001, p.tCollisionS - this.paradigmElapsedSec);
                const thetaRad = 2.0 * Math.atan(p.rOverVS / timeToColl);
                p.thetaDeg = (thetaRad * 180) / Math.PI;
                p.vm = Math.min(20.0, -70.0 + 55.0 * (thetaRad / p.gfThresholdRad));

                if (thetaRad >= p.gfThresholdRad && !p.escapeInitiated) {
                    p.escapeInitiated = true;
                    p.gfSpike = true;
                    p.timeToCollisionJumpMs = timeToColl * 1000;
                    p.loomingSizeJumpDeg = p.thetaDeg;
                    this.dn.escapeActive = true;
                    this.dn.escapeTimer = 0.35;
                    this.dn.dnp01Gf += 1;
                    this.paradigmStatus = 'GF ESCAPE TAKEOFF!';
                }

                if (this.paradigmElapsedSec > p.tCollisionS + 0.3) {
                    this.resetTrial();
                }
                break;
            }

            case 'optomotor': {
                this.fly.x = 45.0;
                this.fly.y = 45.0;
                p.drumAngleDeg = (p.drumAngleDeg + p.drumVelocityDegS * dt) % 360;

                p.saccadeTimer += dt;
                const isSaccade = p.saccadeTimer > 1.8 && p.saccadeTimer < 2.05;
                if (p.saccadeTimer > 2.2) p.saccadeTimer = 0.0;

                const flyYawDegS = isSaccade ? 180.0 : (p.drumVelocityDegS * p.gain);
                p.retinalSlip = p.drumVelocityDegS - flyYawDegS;

                if (isSaccade) {
                    p.effectiveSlip = p.retinalSlip * (1.0 - 0.85);
                    p.efferenceCopyActive = true;
                } else {
                    p.effectiveSlip = p.retinalSlip;
                    p.efferenceCopyActive = false;
                }

                p.hsFiringRate = Math.min(150.0, Math.max(0.0, 40.0 + 1.2 * p.effectiveSlip));
                this.paradigmStatus = isSaccade ? 'SACCADIC EFFERENCE SHUNT' : 'OPTO-STABILIZATION';
                break;
            }

            case 'gap-crossing': {
                const chasmStart = 45.0;
                const chasmEnd = 45.0 + p.gapWidthMm;
                p.isProbing = (this.fly.x >= chasmStart - 3.0 && this.fly.x <= chasmStart + 0.5);

                if (p.isProbing) {
                    p.probingDurationMs += dt * 1000;
                    if (p.decisionOutcome === null) {
                        p.decisionOutcome = (p.gapWidthMm <= p.reachabilityThreshMm) ? 'CROSS' : 'ABORT';
                    }
                    this.paradigmStatus = `PROBING CHASM (${p.decisionOutcome})`;
                }

                if (p.decisionOutcome === 'CROSS') {
                    if (this.fly.x >= chasmEnd + 2.0) {
                        p.crossingSuccess = true;
                        rewardSignal = 1.0;
                        this.paradigmStatus = 'SUCCESSFUL STEP-OVER';
                    }
                } else if (p.decisionOutcome === 'ABORT') {
                    this.fly.heading = Math.PI;
                    this.paradigmStatus = 'ABORT 180° TURN';
                }
                break;
            }

            case 'circadian-dam': {
                const midTubeX = 32.5;
                if ((p.lastX < midTubeX && this.fly.x >= midTubeX) || (p.lastX > midTubeX && this.fly.x <= midTubeX)) {
                    p.beamCrossings += 1;
                }
                p.lastX = this.fly.x;

                if (this.fly.speed < 1.0) {
                    p.consecutiveImmobileMin += dt * 2.0;
                    if (p.consecutiveImmobileMin >= 5.0) {
                        p.totalSleepMin += dt * 2.0;
                        if (!p.inSleepBout) {
                            p.inSleepBout = true;
                            p.sleepBouts += 1;
                        }
                    }
                } else {
                    p.consecutiveImmobileMin = 0.0;
                    p.inSleepBout = false;
                }

                if (this.fly.x > 60.0) this.fly.heading = Math.PI;
                if (this.fly.x < 5.0) this.fly.heading = 0.0;

                this.paradigmStatus = p.inSleepBout ? 'SLEEP BOUT (>=5m)' : 'LOCOMOTING [AWAKE]';
                break;
            }

            case 'courtship': {
                p.totalSteps += 1;
                const dFem = Math.hypot(this.fly.x - p.femalePos[0], this.fly.y - p.femalePos[1]);

                if (dFem < 3.5) {
                    p.courtshipActiveSteps += 1;
                    p.wingAngleDeg = Math.min(90.0, 30.0 + (3.5 - dFem) * 20.0);
                    this.paradigmStatus = 'WING EXTENSION SONG';

                    if (p.femaleType === 'mated' && dFem < 2.0) {
                        p.rejectionKicks += 1;
                        punishmentSignal = 1.0;
                        this.paradigmStatus = 'REJECTION KICK!';
                    }
                } else {
                    p.wingAngleDeg = 0.0;
                    this.paradigmStatus = 'APPROACHING FEMALE';
                }
                p.courtshipIndex = p.courtshipActiveSteps / Math.max(1, p.totalSteps);
                break;
            }

            case 'labyrinth': {
                const distGoal = Math.hypot(this.fly.x - p.goalPos[0], this.fly.y - p.goalPos[1]);
                odorA = Math.exp(-distGoal / 35.0);

                if (distGoal < 10.0) {
                    if (!p.goalReached) {
                        p.goalReached = true;
                        p.timeToGoalMs = this.paradigmElapsedSec * 1000;
                    }
                    rewardSignal = 1.0;
                    this.paradigmStatus = 'FOOD GOAL REACHED!';
                }

                p.pathLength += Math.hypot(this.fly.x - p.lastPathX, this.fly.y - p.lastPathY);
                p.lastPathX = this.fly.x;
                p.lastPathY = this.fly.y;
                const straightDist = Math.hypot(p.goalPos[0] - 12.0, p.goalPos[1] - 15.0);
                p.tortuosity = Math.max(1.0, p.pathLength / straightDist);
                break;
            }
        }

        this.mb.encodeOdor(odorA * 4.0, odorB * 4.0);
        const netValence = this.mb.forward();
        this.mb.stepPlasticity(rewardSignal, punishmentSignal, 1.0 + (1.0 - this.fly.energy), dt);

        const goalError = this.cx.step(this.fly.yawRate, egocentricWind, dt);

        const hasOdor = (odorA + odorB) > 0.04;
        let chemotaxisSteer = -0.55 * ((odorA - odorB) * 3.5);
        if (hasOdor) chemotaxisSteer *= Math.max(0.2, 1.0 + 0.35 * netValence);

        const totalSteerError = hasOdor ? (0.6 * goalError + 0.4 * chemotaxisSteer) : goalError;
        const lalDriveL = Math.max(0, -totalSteerError);
        const lalDriveR = Math.max(0, totalSteerError);

        this.dn.dna02L = lalDriveL * 45.0;
        this.dn.dna02R = lalDriveR * 45.0;
        this.dn.dna02Diff = this.dn.dna02R - this.dn.dna02L;
        this.dn.dnp09 = hasOdor ? Math.min(65.0, odorA * 80.0) : Math.max(0, this.dn.dnp09 - 25.0 * dt);
        this.dn.bpn = 22.0 * (1.0 + 0.8 * (1.0 - this.fly.energy));

        if (minThreatDist < 14.0 && !loomingTrigger) {
            this.dn.mdn = 45.0;
        } else if (netValence < -0.25 && hasOdor) {
            this.dn.mdn = Math.min(40.0, -netValence * 55.0);
        } else {
            this.dn.mdn = Math.max(0, this.dn.mdn - 45.0 * dt);
        }

        if (!this.gfLesioned && loomingTrigger && !this.dn.escapeActive) {
            this.dn.dnp01Gf += 1;
            this.dn.escapeActive = true;
            this.dn.escapeTimer = 0.30;
        }
        if (this.dn.escapeActive) {
            this.dn.escapeTimer -= dt;
            if (this.dn.escapeTimer <= 0) this.dn.escapeActive = false;
        }

        const isReversing = this.dn.mdn > 25.0;
        this.cpg.step(this.dn.dnp09 + this.dn.bpn, isReversing, dt);

        if (this.activeParadigmId !== 'visual-operant' && this.activeParadigmId !== 'optomotor') {
            if (this.dn.escapeActive) {
                this.fly.speed = 42.0;
                this.fly.yawRate = 6.0 * Math.sign(this.dn.dna02Diff || 1.0);
            } else {
                const fwdThrust = isReversing ? -9.5 : (1.4 * (this.dn.dnp09 + this.dn.bpn) / 50.0) * 16.0;
                const drag = 4.2 * this.fly.speed;
                this.fly.speed += (fwdThrust - drag) * dt;
                this.fly.speed = Math.max(-8.0, Math.min(32.0, this.fly.speed));

                const yawTorque = 0.085 * this.dn.dna02Diff;
                this.fly.yawRate += (yawTorque - 3.5 * this.fly.yawRate) * dt;
            }

            this.fly.heading = ((this.fly.heading + this.fly.yawRate * dt + Math.PI) % (2 * Math.PI) + 2 * Math.PI) % (2 * Math.PI) - Math.PI;

            let vx = this.fly.speed * Math.cos(this.fly.heading);
            let vy = this.fly.speed * Math.sin(this.fly.heading);
            let proposedX = this.fly.x + vx * dt;
            let proposedY = this.fly.y + vy * dt;

            if (this.currentWalls && this.currentWalls.length > 0) {
                for (const wall of this.currentWalls) {
                    const col = wall.resolveCircleCollision(proposedX, proposedY, vx, vy, this.fly.radius);
                    if (col.collided) {
                        proposedX = col.x;
                        proposedY = col.y;
                        vx = col.vx;
                        vy = col.vy;
                        this.collisionNormals.push({
                            x: proposedX,
                            y: proposedY,
                            nx: col.normal[0],
                            ny: col.normal[1]
                        });
                        if (this.paradigmState && this.paradigmState.wallCollisions !== undefined) {
                            this.paradigmState.wallCollisions += 1;
                        }
                    }
                }
                this.fly.x = proposedX;
                this.fly.y = proposedY;
                this.fly.speed = Math.hypot(vx, vy);
                if (this.fly.speed > 0.5) this.fly.heading = Math.atan2(vy, vx);
            } else if (this.activeParadigmId === 'open-arena') {
                this.fly.x = proposedX;
                this.fly.y = proposedY;
                if (Math.abs(this.fly.x) > 150) this.fly.x *= -0.98;
                if (Math.abs(this.fly.y) > 110) this.fly.y *= -0.98;
            } else {
                this.fly.x = proposedX;
                this.fly.y = proposedY;
            }
        }

        this.fly.trail.push({ x: this.fly.x, y: this.fly.y, speed: this.fly.speed, escape: this.dn.escapeActive });
        if (this.fly.trail.length > 150) this.fly.trail.shift();

        const metricInfo = this.getCanonicalMetricInfo();
        if (this.isRecording || this.stepCount % 5 === 0) {
            this.telemetryBuffer.push({
                step: this.stepCount,
                simTime: this.simTime.toFixed(3),
                paradigm: this.activeParadigmId,
                trial: this.currentTrial,
                x: this.fly.x.toFixed(2),
                y: this.fly.y.toFixed(2),
                heading: this.fly.heading.toFixed(3),
                speed: this.fly.speed.toFixed(2),
                yawRate: this.fly.yawRate.toFixed(3),
                metricLabel: metricInfo.label,
                metricValue: metricInfo.value,
                pamDopamine: this.mb.pamRate.toFixed(1),
                ppl1Dopamine: this.mb.ppl1Rate.toFixed(1),
                netValence: netValence.toFixed(4)
            });
            if (this.telemetryBuffer.length > 4000) this.telemetryBuffer.shift();
        }
    }

    getCanonicalMetricInfo() {
        const p = this.paradigmState;
        switch (this.activeParadigmId) {
            case 'open-arena':
                return {
                    label: 'Preference Index (PI)',
                    value: (this.mb.netValence >= 0 ? '+' : '') + this.mb.netValence.toFixed(2),
                    sub: 'Net Associative Valence'
                };
            case 't-maze':
                return {
                    label: 'Performance Index (PI)',
                    value: (p.performanceIndex >= 0 ? '+' : '') + (p.performanceIndex || 0).toFixed(2),
                    sub: `CS+ vs CS- (${p.choiceCounts.arm_a} / ${p.choiceCounts.arm_b})`
                };
            case 'y-maze':
                return {
                    label: 'Alternation Rate (SAR)',
                    value: ((p.sar || 0) * 100).toFixed(1) + '%',
                    sub: `Arm Visits [${p.armCounts.join(', ')}]`
                };
            case 'heat-maze':
                return {
                    label: 'Escape Latency (ms)',
                    value: p.escapeLatencyMs ? p.escapeLatencyMs.toFixed(0) + ' ms' : 'SEARCHING',
                    sub: `Temp: ${p.temp.toFixed(1)}°C | Dose: ${p.cumulativeDose.toFixed(0)}`
                };
            case 'buridan':
                return {
                    label: 'Centrophobism Index',
                    value: (p.centrophobism || 1.0).toFixed(2),
                    sub: `Fixation: ${(p.meanFixation || 0).toFixed(2)} | Crossings: ${p.stripeCrossings}`
                };
            case 'visual-operant':
                return {
                    label: 'Learning Index (LI)',
                    value: ((p.learningIndex || 0) >= 0 ? '+' : '') + (p.learningIndex || 0).toFixed(2),
                    sub: `Safe: ${((p.timeSafeMs / Math.max(1, p.timeSafeMs + p.timePunishedMs)) * 100).toFixed(0)}%`
                };
            case 'wind-tunnel':
                const s2c = p.castSteps > 0 ? (p.surgeSteps / p.castSteps).toFixed(2) : (p.surgeSteps > 0 ? 'INF' : '0.00');
                return {
                    label: 'Surge-to-Cast Ratio',
                    value: s2c,
                    sub: `Progress: ${p.upwindProgress.toFixed(1)} mm | State: ${p.behavioralState}`
                };
            case 'looming-escape':
                return {
                    label: 'GF Escape Latency',
                    value: p.timeToCollisionJumpMs ? p.timeToCollisionJumpMs.toFixed(0) + ' ms' : 'APPROACHING',
                    sub: `Looming Size: ${p.thetaDeg.toFixed(1)}° | Vm: ${p.vm.toFixed(0)} mV`
                };
            case 'optomotor':
                return {
                    label: 'Optomotor Gain',
                    value: (p.gain || 0.88).toFixed(2),
                    sub: `HS Rate: ${p.hsFiringRate.toFixed(0)} Hz | Efference: 85%`
                };
            case 'gap-crossing':
                return {
                    label: 'Gap Decision',
                    value: p.decisionOutcome ? p.decisionOutcome : (p.isProbing ? 'PROBING' : 'APPROACH'),
                    sub: `Gap: ${p.gapWidthMm} mm | Thresh: 3.8 mm`
                };
            case 'circadian-dam':
                return {
                    label: 'Total Sleep Minutes',
                    value: (p.totalSleepMin || 0).toFixed(0) + ' min',
                    sub: `Beams: ${p.beamCrossings} | Bouts: ${p.sleepBouts}`
                };
            case 'courtship':
                return {
                    label: 'Courtship Index (CI)',
                    value: ((p.courtshipIndex || 0) * 100).toFixed(1) + '%',
                    sub: `Wing: ${p.wingAngleDeg.toFixed(0)}° | Kicks: ${p.rejectionKicks}`
                };
            case 'labyrinth':
                return {
                    label: 'Path Tortuosity',
                    value: (p.tortuosity || 1.0).toFixed(2),
                    sub: `Hits: ${p.wallCollisions} | Goal: ${p.goalReached ? 'REACHED' : 'SEARCH'}`
                };
            default:
                return { label: 'Canonical Metric', value: '0.00', sub: 'Standard' };
        }
    }

    getParadigmMetrics() {
        return {
            paradigm: this.activeParadigmId,
            trial: this.currentTrial,
            elapsedSec: this.paradigmElapsedSec,
            metric: this.getCanonicalMetricInfo(),
            state: this.paradigmState
        };
    }

    render() {
        this.ctx.clearRect(0, 0, this.viewWidth, this.viewHeight);

        switch (this.activeParadigmId) {
            case 'open-arena': this.renderOpenArena(this.ctx); break;
            case 't-maze': this.renderTMaze(this.ctx); break;
            case 'y-maze': this.renderYMaze(this.ctx); break;
            case 'heat-maze': this.renderHeatMaze(this.ctx); break;
            case 'buridan': this.renderBuridan(this.ctx); break;
            case 'visual-operant': this.renderVisualOperant(this.ctx); break;
            case 'wind-tunnel': this.renderWindTunnel(this.ctx); break;
            case 'looming-escape': this.renderLooming(this.ctx); break;
            case 'optomotor': this.renderOptomotor(this.ctx); break;
            case 'gap-crossing': this.renderGapCrossing(this.ctx); break;
            case 'circadian-dam': this.renderCircadianDAM(this.ctx); break;
            case 'courtship': this.renderCourtship(this.ctx); break;
            case 'labyrinth': this.renderLabyrinth(this.ctx); break;
        }

        if (this.fly.trail.length > 1) {
            for (let i = 1; i < this.fly.trail.length; i++) {
                const pt1 = this.worldToScreen(this.fly.trail[i - 1].x, this.fly.trail[i - 1].y);
                const pt2 = this.worldToScreen(this.fly.trail[i].x, this.fly.trail[i].y);
                this.ctx.strokeStyle = this.fly.trail[i].escape ? '#f43f5e' : 'rgba(56, 189, 248, 0.4)';
                this.ctx.lineWidth = this.fly.trail[i].escape ? 2.5 : 1.2;
                this.ctx.beginPath();
                this.ctx.moveTo(pt1.x, pt1.y);
                this.ctx.lineTo(pt2.x, pt2.y);
                this.ctx.stroke();
            }
        }

        this.renderFlyBody(this.ctx);
    }

    renderOpenArena(ctx) {
        ctx.strokeStyle = 'rgba(56, 189, 248, 0.15)';
        ctx.lineWidth = 1.2;
        for (const p of this.particles) {
            const s = this.worldToScreen(p.x, p.y);
            ctx.beginPath();
            ctx.moveTo(s.x, s.y);
            ctx.lineTo(s.x + this.windVector[0] * 0.4, s.y - this.windVector[1] * 0.4);
            ctx.stroke();
        }

        for (const food of this.foodItems) {
            const s = this.worldToScreen(food.x, food.y);
            const grad = ctx.createRadialGradient(s.x, s.y, 2, s.x, s.y, 65);
            grad.addColorStop(0, 'rgba(34, 197, 94, 0.45)');
            grad.addColorStop(1, 'rgba(34, 197, 94, 0.0)');
            ctx.fillStyle = grad;
            ctx.beginPath(); ctx.arc(s.x, s.y, 65, 0, 2 * Math.PI); ctx.fill();
            ctx.fillStyle = '#4ade80';
            ctx.beginPath(); ctx.arc(s.x, s.y, 7, 0, 2 * Math.PI); ctx.fill();
        }

        for (const a of this.alarms) {
            const s = this.worldToScreen(a.x, a.y);
            ctx.fillStyle = `rgba(244, 63, 94, ${0.35 * a.life})`;
            ctx.beginPath(); ctx.arc(s.x, s.y, 50, 0, 2 * Math.PI); ctx.fill();
        }

        for (const pred of this.predators) {
            const s = this.worldToScreen(pred.x, pred.y);
            ctx.fillStyle = '#ef4444';
            ctx.beginPath(); ctx.arc(s.x, s.y, 8, 0, 2 * Math.PI); ctx.fill();
        }
    }

    renderTMaze(ctx) {
        ctx.strokeStyle = '#38bdf8';
        ctx.lineWidth = 2.0;
        ctx.shadowColor = 'rgba(56, 189, 248, 0.5)';
        ctx.shadowBlur = 8;
        for (const w of this.currentWalls) {
            const p1 = this.worldToScreen(w.p1[0], w.p1[1]);
            const p2 = this.worldToScreen(w.p2[0], w.p2[1]);
            ctx.beginPath();
            ctx.moveTo(p1.x, p1.y);
            ctx.lineTo(p2.x, p2.y);
            ctx.stroke();
        }
        ctx.shadowBlur = 0;

        const sa = this.worldToScreen(15.0, 50.0);
        const gradA = ctx.createRadialGradient(sa.x, sa.y, 2, sa.x, sa.y, 70);
        gradA.addColorStop(0, 'rgba(34, 197, 94, 0.5)');
        gradA.addColorStop(1, 'rgba(34, 197, 94, 0.0)');
        ctx.fillStyle = gradA;
        ctx.beginPath(); ctx.arc(sa.x, sa.y, 70, 0, 2 * Math.PI); ctx.fill();

        ctx.fillStyle = '#22c55e';
        ctx.beginPath(); ctx.arc(sa.x, sa.y, 8, 0, 2 * Math.PI); ctx.fill();
        ctx.fillStyle = '#86efac';
        ctx.beginPath(); ctx.arc(sa.x - 2, sa.y - 2, 3, 0, 2 * Math.PI); ctx.fill();

        const sb = this.worldToScreen(125.0, 50.0);
        const gradB = ctx.createRadialGradient(sb.x, sb.y, 2, sb.x, sb.y, 70);
        gradB.addColorStop(0, 'rgba(244, 63, 94, 0.5)');
        gradB.addColorStop(1, 'rgba(244, 63, 94, 0.0)');
        ctx.fillStyle = gradB;
        ctx.beginPath(); ctx.arc(sb.x, sb.y, 70, 0, 2 * Math.PI); ctx.fill();

        ctx.strokeStyle = this.paradigmState.shockPulse > 0.1 ? '#fbbf24' : 'rgba(244, 63, 94, 0.35)';
        ctx.lineWidth = 1.0;
        for (let gx = 82; gx < 128; gx += 4) {
            const sTop = this.worldToScreen(gx, 56.0);
            const sBot = this.worldToScreen(gx, 44.0);
            ctx.beginPath(); ctx.moveTo(sTop.x, sTop.y); ctx.lineTo(sBot.x, sBot.y); ctx.stroke();
        }

        ctx.strokeStyle = 'rgba(56, 189, 248, 0.35)';
        for (let y = 15; y < 42; y += 8) {
            const sStem = this.worldToScreen(70.0, y);
            ctx.beginPath();
            ctx.moveTo(sStem.x, sStem.y);
            ctx.lineTo(sStem.x, sStem.y + 6);
            ctx.stroke();
        }

        const b1 = this.worldToScreen(70.0, 43.0);
        const b2 = this.worldToScreen(70.0, 57.0);
        ctx.strokeStyle = 'rgba(251, 191, 36, 0.6)';
        ctx.setLineDash([4, 4]);
        ctx.beginPath(); ctx.moveTo(b1.x, b1.y); ctx.lineTo(b2.x, b2.y); ctx.stroke();
        ctx.setLineDash([]);
    }

    renderYMaze(ctx) {
        ctx.strokeStyle = '#38bdf8';
        ctx.lineWidth = 2.0;
        ctx.shadowColor = 'rgba(56, 189, 248, 0.4)';
        ctx.shadowBlur = 6;
        for (const w of this.currentWalls) {
            const p1 = this.worldToScreen(w.p1[0], w.p1[1]);
            const p2 = this.worldToScreen(w.p2[0], w.p2[1]);
            ctx.beginPath(); ctx.moveTo(p1.x, p1.y); ctx.lineTo(p2.x, p2.y); ctx.stroke();
        }
        ctx.shadowBlur = 0;

        const tips = [
            { pos: [60.0, 100.0], label: `Arm 0 (N): ${this.paradigmState.armCounts[0]}` },
            { pos: [25.4, 40.0], label: `Arm 1 (SW): ${this.paradigmState.armCounts[1]}` },
            { pos: [94.6, 40.0], label: `Arm 2 (SE): ${this.paradigmState.armCounts[2]}` }
        ];
        ctx.font = '10px monospace';
        ctx.fillStyle = '#fbbf24';
        tips.forEach(t => {
            const s = this.worldToScreen(t.pos[0], t.pos[1]);
            ctx.fillText(t.label, s.x - 30, s.y - 8);
        });

        const hubS = this.worldToScreen(60.0, 60.0);
        ctx.strokeStyle = 'rgba(192, 132, 252, 0.5)';
        ctx.beginPath(); ctx.arc(hubS.x, hubS.y, 14, 0, 2 * Math.PI); ctx.stroke();
    }

    renderHeatMaze(ctx) {
        const p = this.paradigmState;
        const sRef = this.worldToScreen(p.refugePos[0], p.refugePos[1]);
        const sCenter = this.worldToScreen(60.0, 60.0);

        const rArenaS = 55.0 * sCenter.scale;
        const gradFloor = ctx.createRadialGradient(sRef.x, sRef.y, 5, sCenter.x, sCenter.y, rArenaS);
        gradFloor.addColorStop(0, 'rgba(34, 211, 238, 0.7)');
        gradFloor.addColorStop(0.3, 'rgba(245, 158, 11, 0.4)');
        gradFloor.addColorStop(1, 'rgba(239, 68, 68, 0.55)');
        ctx.fillStyle = gradFloor;
        ctx.beginPath(); ctx.arc(sCenter.x, sCenter.y, rArenaS, 0, 2 * Math.PI); ctx.fill();

        const rRefS = p.refugeRadius * sCenter.scale;
        ctx.strokeStyle = '#22d3ee';
        ctx.lineWidth = 2.0;
        ctx.beginPath(); ctx.arc(sRef.x, sRef.y, rRefS, 0, 2 * Math.PI); ctx.stroke();
        ctx.fillStyle = 'rgba(34, 211, 238, 0.3)';
        ctx.beginPath(); ctx.arc(sRef.x, sRef.y, rRefS, 0, 2 * Math.PI); ctx.fill();

        const landmarks = [
            { x: 115.0, y: 60.0, glyph: 'STRIPE 0°' },
            { x: 60.0, y: 115.0, glyph: 'DOUBLE 90°' },
            { x: 5.0, y: 60.0, glyph: 'RECT 180°' },
            { x: 60.0, y: 5.0, glyph: 'CROSS 270°' }
        ];
        ctx.font = '9px monospace';
        ctx.fillStyle = '#f8fafc';
        landmarks.forEach(lm => {
            const sl = this.worldToScreen(lm.x, lm.y);
            ctx.fillText(lm.glyph, sl.x - 24, sl.y + 4);
        });

        const sf = this.worldToScreen(this.fly.x, this.fly.y);
        ctx.strokeStyle = 'rgba(56, 189, 248, 0.5)';
        ctx.setLineDash([3, 3]);
        ctx.beginPath(); ctx.moveTo(sf.x, sf.y); ctx.lineTo(sRef.x, sRef.y); ctx.stroke();
        ctx.setLineDash([]);
    }

    renderBuridan(ctx) {
        const p = this.paradigmState;
        const sCenter = this.worldToScreen(p.center[0], p.center[1]);
        const rPlatS = p.platformRadius * sCenter.scale;

        ctx.fillStyle = 'rgba(14, 116, 144, 0.35)';
        ctx.beginPath(); ctx.arc(sCenter.x, sCenter.y, rPlatS + 20, 0, 2 * Math.PI); ctx.fill();

        ctx.fillStyle = '#0f172a';
        ctx.strokeStyle = '#38bdf8';
        ctx.lineWidth = 2.0;
        ctx.beginPath(); ctx.arc(sCenter.x, sCenter.y, rPlatS, 0, 2 * Math.PI); ctx.fill(); ctx.stroke();

        ctx.strokeStyle = 'rgba(56, 189, 248, 0.4)';
        ctx.setLineDash([4, 4]);
        ctx.beginPath(); ctx.arc(sCenter.x, sCenter.y, 25.0 * sCenter.scale, 0, 2 * Math.PI); ctx.stroke();
        ctx.setLineDash([]);

        const s1 = this.worldToScreen(110.0, 60.0);
        const s2 = this.worldToScreen(10.0, 60.0);
        ctx.fillStyle = '#000000';
        ctx.strokeStyle = '#ffffff';
        ctx.lineWidth = 1.5;
        ctx.fillRect(s1.x - 5, s1.y - 18, 10, 36); ctx.strokeRect(s1.x - 5, s1.y - 18, 10, 36);
        ctx.fillRect(s2.x - 5, s2.y - 18, 10, 36); ctx.strokeRect(s2.x - 5, s2.y - 18, 10, 36);

        const sf = this.worldToScreen(this.fly.x, this.fly.y);
        ctx.strokeStyle = '#fbbf24';
        ctx.beginPath(); ctx.moveTo(sf.x, sf.y); ctx.lineTo(sf.x + Math.cos(this.fly.heading) * 20, sf.y - Math.sin(this.fly.heading) * 20); ctx.stroke();
    }

    renderVisualOperant(ctx) {
        const p = this.paradigmState;
        const sc = this.worldToScreen(40.0, 40.0);

        const rDrum = 75;
        ctx.strokeStyle = '#334155';
        ctx.lineWidth = 8;
        ctx.beginPath(); ctx.arc(sc.x, sc.y, rDrum, 0, 2 * Math.PI); ctx.stroke();

        for (let q = 0; q < 4; q++) {
            const startAng = ((p.drumAngleDeg + q * 90) * Math.PI) / 180;
            const endAng = startAng + Math.PI / 2;
            const isPunished = (q % 2 === 1);
            ctx.strokeStyle = isPunished ? '#f43f5e' : '#22c55e';
            ctx.lineWidth = 6;
            ctx.beginPath(); ctx.arc(sc.x, sc.y, rDrum, startAng, endAng); ctx.stroke();
        }

        if (p.laserActive) {
            ctx.strokeStyle = '#f43f5e';
            ctx.lineWidth = 3.0;
            ctx.shadowColor = '#f43f5e';
            ctx.shadowBlur = 12;
            ctx.beginPath();
            ctx.moveTo(sc.x, sc.y - 70);
            ctx.lineTo(sc.x, sc.y);
            ctx.stroke();
            ctx.shadowBlur = 0;
        }

        ctx.fillStyle = '#94a3b8';
        ctx.font = '10px monospace';
        ctx.fillText(`Yaw Torque: ${p.yawTorque.toFixed(2)} | Drum: ${p.drumAngleDeg.toFixed(0)}°`, sc.x - 70, sc.y + 95);
    }

    renderWindTunnel(ctx) {
        const p = this.paradigmState;
        ctx.strokeStyle = 'rgba(56, 189, 248, 0.6)';
        ctx.lineWidth = 2.0;
        for (const w of this.currentWalls) {
            const p1 = this.worldToScreen(w.p1[0], w.p1[1]);
            const p2 = this.worldToScreen(w.p2[0], w.p2[1]);
            ctx.beginPath(); ctx.moveTo(p1.x, p1.y); ctx.lineTo(p2.x, p2.y); ctx.stroke();
        }

        ctx.strokeStyle = 'rgba(56, 189, 248, 0.2)';
        ctx.lineWidth = 1.0;
        for (const pt of this.particles) {
            const s = this.worldToScreen(pt.x + 100, (pt.y % 55) + 5);
            ctx.beginPath(); ctx.moveTo(s.x, s.y); ctx.lineTo(s.x - 15, s.y); ctx.stroke();
        }

        const sn = this.worldToScreen(p.nozzlePos[0], p.nozzlePos[1]);
        ctx.fillStyle = '#22c55e';
        ctx.beginPath(); ctx.arc(sn.x, sn.y, 6, 0, 2 * Math.PI); ctx.fill();

        const gradPuff = ctx.createRadialGradient(sn.x, sn.y, 2, sn.x - 90, sn.y, 110);
        gradPuff.addColorStop(0, 'rgba(34, 197, 94, 0.45)');
        gradPuff.addColorStop(1, 'rgba(34, 197, 94, 0.0)');
        ctx.fillStyle = gradPuff;
        ctx.fillRect(sn.x - 140, sn.y - 20, 140, 40);

        ctx.font = 'bold 11px monospace';
        ctx.fillStyle = p.behavioralState === 'SURGE' ? '#22c55e' : '#f59e0b';
        ctx.fillText(`[${p.behavioralState}]`, sn.x - 130, sn.y - 25);
    }

    renderLooming(ctx) {
        const p = this.paradigmState;
        const sc = this.worldToScreen(40.0, 40.0);

        const radiusPx = Math.min(130, Math.tan(p.thetaDeg * Math.PI / 360) * 110);
        ctx.fillStyle = '#05070e';
        ctx.beginPath(); ctx.arc(sc.x, sc.y, radiusPx, 0, 2 * Math.PI); ctx.fill();

        ctx.strokeStyle = 'rgba(244, 63, 94, 0.4)';
        ctx.lineWidth = 1.5;
        ctx.beginPath(); ctx.arc(sc.x, sc.y, radiusPx + 8, 0, 2 * Math.PI); ctx.stroke();

        const threshPx = Math.tan(65.0 * Math.PI / 360) * 110;
        ctx.strokeStyle = 'rgba(239, 68, 68, 0.6)';
        ctx.setLineDash([4, 4]);
        ctx.beginPath(); ctx.arc(sc.x, sc.y, threshPx, 0, 2 * Math.PI); ctx.stroke();
        ctx.setLineDash([]);

        ctx.font = '10px monospace';
        ctx.fillStyle = '#f8fafc';
        ctx.fillText(`θ = ${p.thetaDeg.toFixed(1)}° (GF Thresh: 65°)`, sc.x - 55, sc.y + 115);
    }

    renderOptomotor(ctx) {
        const p = this.paradigmState;
        const sc = this.worldToScreen(45.0, 45.0);

        const numStripes = 24;
        for (let i = 0; i < numStripes; i++) {
            const a1 = ((p.drumAngleDeg + i * (360 / numStripes)) * Math.PI) / 180;
            const a2 = a1 + (Math.PI / numStripes);
            ctx.fillStyle = i % 2 === 0 ? '#020617' : '#e2e8f0';
            ctx.beginPath(); ctx.moveTo(sc.x, sc.y); ctx.arc(sc.x, sc.y, 85, a1, a2); ctx.fill();
        }

        ctx.fillStyle = '#0f172a';
        ctx.beginPath(); ctx.arc(sc.x, sc.y, 65, 0, 2 * Math.PI); ctx.fill();

        ctx.font = '10px monospace';
        ctx.fillStyle = '#38bdf8';
        ctx.fillText(`HS Rate: ${p.hsFiringRate.toFixed(1)} Hz | Efference Shunt: 85%`, sc.x - 75, sc.y + 95);
    }

    renderGapCrossing(ctx) {
        const p = this.paradigmState;
        const sTrack1 = this.worldToScreen(0.0, 10.0);
        const sChasm = this.worldToScreen(45.0, 10.0);
        const sLand = this.worldToScreen(45.0 + p.gapWidthMm, 10.0);
        const sEnd = this.worldToScreen(100.0, 10.0);

        ctx.fillStyle = '#334155';
        ctx.fillRect(sTrack1.x, sTrack1.y - 12, (sChasm.x - sTrack1.x), 24);
        ctx.fillRect(sLand.x, sLand.y - 12, (sEnd.x - sLand.x), 24);

        ctx.fillStyle = '#020617';
        ctx.fillRect(sChasm.x, sChasm.y - 25, (sLand.x - sChasm.x), 50);

        ctx.font = 'bold 10px monospace';
        ctx.fillStyle = p.decisionOutcome === 'CROSS' ? '#22c55e' : '#f43f5e';
        ctx.fillText(`Chasm: ${p.gapWidthMm}mm [${p.decisionOutcome || 'APPROACHING'}]`, sChasm.x - 20, sChasm.y - 32);
    }

    renderCircadianDAM(ctx) {
        const p = this.paradigmState;
        for (let i = 0; i < 16; i++) {
            const sTopL = this.worldToScreen(0.0, (i + 1) * 10.0);
            const sBotR = this.worldToScreen(65.0, i * 10.0);
            const w = sBotR.x - sTopL.x;
            const h = sBotR.y - sTopL.y;

            ctx.strokeStyle = (i === p.activeTube) ? '#38bdf8' : 'rgba(255, 255, 255, 0.15)';
            ctx.lineWidth = (i === p.activeTube) ? 1.5 : 0.8;
            ctx.strokeRect(sTopL.x, sTopL.y, w, h);

            ctx.fillStyle = 'rgba(245, 158, 11, 0.5)';
            ctx.fillRect(sTopL.x, sTopL.y, 8, h);

            const sBeam = this.worldToScreen(32.5, i * 10.0 + 5.0);
            ctx.strokeStyle = 'rgba(244, 63, 94, 0.7)';
            ctx.beginPath(); ctx.moveTo(sBeam.x, sTopL.y); ctx.lineTo(sBeam.x, sTopL.y + h); ctx.stroke();
        }
    }

    renderCourtship(ctx) {
        const p = this.paradigmState;
        const sc = this.worldToScreen(p.chamberCenter[0], p.chamberCenter[1]);
        const rChamberS = p.chamberRadius * sc.scale;

        ctx.strokeStyle = '#c084fc';
        ctx.lineWidth = 2.0;
        ctx.beginPath(); ctx.arc(sc.x, sc.y, rChamberS, 0, 2 * Math.PI); ctx.stroke();

        const sf = this.worldToScreen(p.femalePos[0], p.femalePos[1]);
        ctx.fillStyle = '#fb7185';
        ctx.beginPath(); ctx.ellipse(sf.x, sf.y, 6, 4.5, 0, 0, 2 * Math.PI); ctx.fill();
        ctx.font = '9px monospace';
        ctx.fillStyle = '#f8fafc';
        ctx.fillText(`♀ (${p.femaleType})`, sf.x - 14, sf.y - 8);

        if (p.rejectionKicks > 0 && Math.hypot(this.fly.x - p.femalePos[0], this.fly.y - p.femalePos[1]) < 2.2) {
            ctx.strokeStyle = '#ef4444';
            ctx.lineWidth = 2.0;
            ctx.beginPath(); ctx.arc(sf.x, sf.y, 14, 0, 2 * Math.PI); ctx.stroke();
            ctx.fillStyle = '#ef4444';
            ctx.fillText('! REJECTION KICK !', sf.x - 35, sf.y + 20);
        }
    }

    renderLabyrinth(ctx) {
        const p = this.paradigmState;
        ctx.strokeStyle = '#38bdf8';
        ctx.lineWidth = 2.0;
        for (const w of this.currentWalls) {
            const p1 = this.worldToScreen(w.p1[0], w.p1[1]);
            const p2 = this.worldToScreen(w.p2[0], w.p2[1]);
            ctx.beginPath(); ctx.moveTo(p1.x, p1.y); ctx.lineTo(p2.x, p2.y); ctx.stroke();
        }

        const sg = this.worldToScreen(p.goalPos[0], p.goalPos[1]);
        const gradGoal = ctx.createRadialGradient(sg.x, sg.y, 2, sg.x, sg.y, 60);
        gradGoal.addColorStop(0, 'rgba(34, 197, 94, 0.6)');
        gradGoal.addColorStop(1, 'rgba(34, 197, 94, 0.0)');
        ctx.fillStyle = gradGoal;
        ctx.beginPath(); ctx.arc(sg.x, sg.y, 60, 0, 2 * Math.PI); ctx.fill();
        ctx.fillStyle = '#22c55e';
        ctx.beginPath(); ctx.arc(sg.x, sg.y, 8, 0, 2 * Math.PI); ctx.fill();

        ctx.strokeStyle = '#fbbf24';
        ctx.lineWidth = 1.5;
        for (const cn of this.collisionNormals) {
            const scn = this.worldToScreen(cn.x, cn.y);
            ctx.beginPath();
            ctx.moveTo(scn.x, scn.y);
            ctx.lineTo(scn.x + cn.nx * 14, scn.y - cn.ny * 14);
            ctx.stroke();
        }
    }

    renderFlyBody(ctx) {
        const sf = this.worldToScreen(this.fly.x, this.fly.y);
        ctx.save();
        ctx.translate(sf.x, sf.y);
        ctx.rotate(-this.fly.heading);

        ctx.strokeStyle = 'rgba(168, 85, 247, 0.2)';
        ctx.lineWidth = 0.8;
        for (let a = -Math.PI * 0.7; a <= Math.PI * 0.7; a += 0.35) {
            ctx.beginPath();
            ctx.moveTo(6, 0);
            ctx.lineTo(6 + Math.cos(a) * 35, Math.sin(a) * 35);
            ctx.stroke();
        }

        const legPhaseA = Math.sin(this.cpg.phaseA) > 0;
        const legPhaseB = Math.sin(this.cpg.phaseB) > 0;
        ctx.strokeStyle = '#94a3b8';
        ctx.lineWidth = 1.6;
        this.drawLeg(ctx, 2, -4, 10, -0.6 + (legPhaseA ? 0.3 : -0.2));
        this.drawLeg(ctx, -2, -5, 10, -1.5 + (legPhaseB ? 0.2 : -0.2));
        this.drawLeg(ctx, -6, -4, 10, -2.4 + (legPhaseA ? -0.2 : 0.2));
        this.drawLeg(ctx, 2, 4, 10, 0.6 + (legPhaseB ? -0.3 : 0.2));
        this.drawLeg(ctx, -2, 5, 10, 1.5 + (legPhaseA ? -0.2 : 0.2));
        this.drawLeg(ctx, -6, 4, 10, 2.4 + (legPhaseB ? 0.2 : -0.2));

        if (this.activeParadigmId === 'courtship' && this.paradigmState.wingAngleDeg > 5) {
            const wingAng = (this.paradigmState.wingAngleDeg * Math.PI) / 180;
            ctx.fillStyle = 'rgba(56, 189, 248, 0.35)';
            ctx.strokeStyle = '#38bdf8';
            ctx.lineWidth = 1.2;
            ctx.beginPath();
            ctx.ellipse(-2, -8, 12, 4, -wingAng, 0, 2 * Math.PI);
            ctx.fill(); ctx.stroke();
        }

        ctx.fillStyle = '#1e293b';
        ctx.strokeStyle = '#475569';
        ctx.lineWidth = 1.2;
        ctx.beginPath(); ctx.ellipse(-6, 0, 6, 4, 0, 0, 2 * Math.PI); ctx.fill(); ctx.stroke();

        ctx.fillStyle = this.dn.escapeActive ? '#f43f5e' : '#38bdf8';
        ctx.beginPath(); ctx.ellipse(0, 0, 4.5, 3.5, 0, 0, 2 * Math.PI); ctx.fill(); ctx.stroke();

        ctx.fillStyle = '#e2e8f0';
        ctx.beginPath(); ctx.ellipse(5.5, 0, 3, 3, 0, 0, 2 * Math.PI); ctx.fill();
        ctx.fillStyle = '#ef4444';
        ctx.beginPath(); ctx.arc(6.5, -2.2, 1.8, 0, 2 * Math.PI); ctx.arc(6.5, 2.2, 1.8, 0, 2 * Math.PI); ctx.fill();

        ctx.strokeStyle = '#fbbf24';
        ctx.lineWidth = 1.2;
        ctx.beginPath();
        ctx.moveTo(7, -1.5); ctx.lineTo(12, -3.5);
        ctx.moveTo(7, 1.5); ctx.lineTo(12, 3.5);
        ctx.stroke();

        ctx.restore();
    }

    drawLeg(ctx, baseX, baseY, length, angle) {
        ctx.beginPath();
        ctx.moveTo(baseX, baseY);
        const midX = baseX + Math.cos(angle) * (length * 0.55);
        const midY = baseY + Math.sin(angle) * (length * 0.55);
        const tipX = midX + Math.cos(angle + 0.3) * (length * 0.55);
        const tipY = midY + Math.sin(angle + 0.3) * (length * 0.55);
        ctx.lineTo(midX, midY);
        ctx.lineTo(tipX, tipY);
        ctx.stroke();
    }
}

// =============================================================================
// 7. SCIENTIFIC HUD & ELECTROPHYSIOLOGY DASHBOARD
// =============================================================================

class ScientificHUD {
    constructor(arena) {
        this.arena = arena;

        this.kcCanvas = document.getElementById('kcCanvas');
        this.kcCtx = this.kcCanvas.getContext('2d');

        this.compassCanvas = document.getElementById('compassCanvas');
        this.compassCtx = this.compassCanvas.getContext('2d');

        this.scopeCanvas = document.getElementById('oscilloscopeCanvas');
        this.scopeCtx = this.scopeCanvas.getContext('2d');
        this.scopeHistory = [];

        this.curveCanvas = document.getElementById('curveCanvas');
        this.curveCtx = this.curveCanvas.getContext('2d');
        this.learningTrials = [0.0];

        this.setupEventListeners();
    }

    setupEventListeners() {
        const selParadigm = document.getElementById('paradigm-select');
        if (selParadigm) {
            selParadigm.addEventListener('change', (e) => {
                this.arena.initParadigm(e.target.value);
                const badge = document.getElementById('activeParadigmBadge');
                if (badge) badge.textContent = e.target.value.toUpperCase();
            });
        }

        const btnResetTrial = document.getElementById('btnResetTrial');
        if (btnResetTrial) {
            btnResetTrial.addEventListener('click', () => this.arena.resetTrial());
        }

        const btnDlCsv = document.getElementById('btnDownloadParadigmCsv');
        if (btnDlCsv) {
            btnDlCsv.addEventListener('click', () => this.downloadCsv());
        }

        const btnDlJson = document.getElementById('btnDownloadParadigmJson');
        if (btnDlJson) {
            btnDlJson.addEventListener('click', () => this.downloadJson());
        }

        const selLesion = document.getElementById('selLesion');
        if (selLesion) {
            selLesion.addEventListener('change', (e) => {
                this.arena.setLesion(e.target.value);
                document.getElementById('activeLesionBadge').textContent = e.target.value;
                const desc = {
                    WT: 'Control condition with complete MaleCNS v1.0 and MANC v1.0 circuits functional.',
                    DELTA_MB: 'Kenyon Cell plastic synapses frozen (eta = 0). Abolishes associative memory.',
                    DELTA_CX: 'Central Complex compass lesion. Heading bump decoupled from sensory/motor integration.',
                    DELTA_GF: 'Giant Fiber escape pathway silenced. Optical expansion cannot trigger escape takeoff.',
                    DELTA_JO: 'Johnston’s Organ arista mechanosensory deafened. Anemotactic wind alignment abolished.'
                };
                document.getElementById('lesionDesc').textContent = desc[e.target.value] || '';
                document.getElementById('lesionStatusText').textContent = e.target.value === 'WT' ? 'INTACT' : 'LESIONED';
            });
        }

        const tools = ['toolSelect', 'toolFood', 'toolAlarm', 'toolPredator', 'toolWind'];
        tools.forEach((id) => {
            const btn = document.getElementById(id);
            if (btn) {
                btn.addEventListener('click', () => {
                    tools.forEach((t) => document.getElementById(t).classList.remove('active'));
                    btn.classList.add('active');
                    this.arena.toolMode = id.replace('tool', '').toLowerCase();
                });
            }
        });

        document.getElementById('btnRunAppetitive').addEventListener('click', () => this.runAppetitiveAssay());
        document.getElementById('btnRunAversive').addEventListener('click', () => this.runAversiveAssay());
        document.getElementById('btnResetMemory').addEventListener('click', () => {
            this.arena.mb.reset(false);
            this.learningTrials = [0.0];
            this.updateLearningCurve();
        });

        const recBtn = document.getElementById('btnRecordToggle');
        recBtn.addEventListener('click', () => {
            this.arena.isRecording = !this.arena.isRecording;
            recBtn.textContent = this.arena.isRecording ? 'Stop Recording' : 'Start Recording (CSV)';
            recBtn.classList.toggle('danger', this.arena.isRecording);
        });
        document.getElementById('btnDownloadCsv').addEventListener('click', () => this.downloadCsv());
        document.getElementById('btnDownloadJson').addEventListener('click', () => this.downloadJson());
    }

    runAppetitiveAssay() {
        let steps = 0;
        const trainInterval = setInterval(() => {
            this.arena.mb.stepPlasticity(1.0, 0.0, 1.5, 0.02);
            steps++;
            if (steps > 60) {
                clearInterval(trainInterval);
                const postPi = this.arena.mb.netValence;
                this.learningTrials.push(postPi);
                this.updateLearningCurve();
            }
        }, 30);
    }

    runAversiveAssay() {
        let steps = 0;
        const trainInterval = setInterval(() => {
            this.arena.mb.stepPlasticity(0.0, 1.0, 1.0, 0.02);
            steps++;
            if (steps > 60) {
                clearInterval(trainInterval);
                const postPi = this.arena.mb.netValence;
                this.learningTrials.push(postPi);
                this.updateLearningCurve();
            }
        }, 30);
    }

    downloadCsv() {
        if (this.arena.telemetryBuffer.length === 0) {
            alert('Simulation is buffering telemetry steps. Please wait a moment.');
            return;
        }
        const headers = Object.keys(this.arena.telemetryBuffer[0]).join(',');
        const rows = this.arena.telemetryBuffer.map((r) => Object.values(r).join(',')).join('\n');
        const blob = new Blob([headers + '\n' + rows], { type: 'text/csv;charset=utf-8;' });
        const url = URL.createObjectURL(blob);
        const link = document.createElement('a');
        link.href = url;
        link.download = `flybrain_${this.arena.activeParadigmId}_telemetry_${Date.now()}.csv`;
        document.body.appendChild(link);
        link.click();
        document.body.removeChild(link);
        URL.revokeObjectURL(url);
    }

    downloadJson() {
        const data = {
            metadata: {
                instrument: 'FlyBrain In-Silico Neuroethology Rig',
                activeParadigm: this.arena.activeParadigmId,
                paradigmTitle: this.arena.activeParadigmTitle,
                reference: this.arena.activeParadigmRef,
                lesion: this.arena.lesion,
                currentTrial: this.arena.currentTrial,
                elapsedTimeSec: this.arena.paradigmElapsedSec,
                date: new Date().toISOString(),
                totalSteps: this.arena.stepCount
            },
            canonicalMetrics: this.arena.getParadigmMetrics(),
            telemetrySampleCount: this.arena.telemetryBuffer.length,
            telemetry: this.arena.telemetryBuffer.slice(-500)
        };
        const blob = new Blob([JSON.stringify(data, null, 2)], { type: 'application/json' });
        const url = URL.createObjectURL(blob);
        const a = document.createElement('a');
        a.href = url;
        a.download = `flybrain_${this.arena.activeParadigmId}_trial_${Date.now()}.json`;
        document.body.appendChild(a);
        a.click();
        document.body.removeChild(a);
        URL.revokeObjectURL(url);
    }

    update() {
        document.getElementById('statSimTime').textContent = this.arena.simTime.toFixed(2) + 's';
        document.getElementById('statStep').textContent = this.arena.stepCount;

        document.getElementById('paradigmTitle').textContent = this.arena.activeParadigmTitle;
        document.getElementById('paradigmRef').textContent = this.arena.activeParadigmRef;
        document.getElementById('paradigmTrial').textContent = '#' + this.arena.currentTrial;
        document.getElementById('paradigmTrialBadge').textContent = 'TRIAL #' + this.arena.currentTrial;
        document.getElementById('paradigmElapsed').textContent = this.arena.paradigmElapsedSec.toFixed(2) + 's';
        document.getElementById('paradigmStatus').textContent = this.arena.paradigmStatus;

        const metric = this.arena.getCanonicalMetricInfo();
        document.getElementById('paradigmMetricLabel').textContent = metric.label;
        document.getElementById('paradigmMetricValue').textContent = metric.value;
        document.getElementById('paradigmMetricSub').textContent = metric.sub;

        const pamHz = this.arena.mb.pamRate;
        const ppl1Hz = this.arena.mb.ppl1Rate;
        document.getElementById('paradigmPamRate').textContent = pamHz.toFixed(1) + ' Hz';
        document.getElementById('paradigmPpl1Rate').textContent = ppl1Hz.toFixed(1) + ' Hz';
        document.getElementById('paradigmPamFill').style.width = `${(pamHz / 40.0) * 100}%`;
        document.getElementById('paradigmPpl1Fill').style.width = `${(ppl1Hz / 40.0) * 100}%`;

        const dStateEl = document.getElementById('valDopamineState');
        if (pamHz > 5.0) {
            dStateEl.textContent = 'PAM REWARD BURST';
            dStateEl.style.color = '#f59e0b';
        } else if (ppl1Hz > 5.0) {
            dStateEl.textContent = 'PPL1 SHOCK BURST';
            dStateEl.style.color = '#f43f5e';
        } else {
            dStateEl.textContent = 'QUIESCENT';
            dStateEl.style.color = '#38bdf8';
        }

        const activeKcs = Array.from(this.arena.mb.kcFiring).filter((r) => r > 0).length;
        document.getElementById('kcActiveFraction').textContent = `${activeKcs} / 120 (${((activeKcs / 120) * 100).toFixed(0)}%)`;
        document.getElementById('valPamDopamine').textContent = pamHz.toFixed(1) + ' Hz';
        document.getElementById('valPpl1Dopamine').textContent = ppl1Hz.toFixed(1) + ' Hz';
        document.getElementById('barPamFill').style.width = `${(pamHz / 40.0) * 100}%`;
        document.getElementById('barPpl1Fill').style.width = `${(ppl1Hz / 40.0) * 100}%`;

        const valence = this.arena.mb.netValence;
        document.getElementById('valNetValence').textContent = `Valence: ${(valence >= 0 ? '+' : '') + valence.toFixed(2)}`;
        document.getElementById('needleValence').style.left = `${((valence + 1.0) / 2.0) * 100}%`;
        document.getElementById('currentPiVal').textContent = (valence >= 0 ? '+' : '') + valence.toFixed(2);
        this.renderKcMatrix();

        document.getElementById('valCompassHeading').textContent = `${((this.arena.fly.heading * 180) / Math.PI).toFixed(1)}°`;
        document.getElementById('valBumpHead').textContent = `${((this.arena.cx.headingBump * 180) / Math.PI).toFixed(0)}°`;
        const windGlobal = Math.atan2(-this.arena.windVector[1], -this.arena.windVector[0]);
        document.getElementById('valUpwindHead').textContent = `${((windGlobal * 180) / Math.PI).toFixed(0)}°`;
        document.getElementById('valPfl3Error').textContent = (this.arena.cx.pfl3ErrorR - this.arena.cx.pfl3ErrorL).toFixed(2);
        this.renderCompass();

        this.scopeHistory.push({
            dna02: this.arena.dn.dna02Diff,
            dnp09: this.arena.dn.dnp09,
            bpn: this.arena.dn.bpn,
            mdn: this.arena.dn.mdn,
            dnp01: this.arena.dn.escapeActive ? 50.0 : 0.0
        });
        if (this.scopeHistory.length > 150) this.scopeHistory.shift();
        this.renderOscilloscope();

        document.getElementById('valCpgFreq').textContent = this.arena.cpg.steppingFreq.toFixed(1) + ' Hz';
        for (const [leg, isStance] of Object.entries(this.arena.cpg.legStates)) {
            const el = document.getElementById(`leg${leg}`);
            if (el) {
                el.className = `leg-cell ${isStance ? 'stance' : 'swing'}`;
                el.textContent = `${leg}: ${isStance ? 'STANCE' : 'SWING'}`;
            }
        }
    }

    renderKcMatrix() {
        const w = this.kcCanvas.width;
        const h = this.kcCanvas.height;
        this.kcCtx.clearRect(0, 0, w, h);

        const cols = 24, rows = 5;
        const cellW = w / cols, cellH = h / rows;

        for (let i = 0; i < 120; i++) {
            const c = i % cols;
            const r = Math.floor(i / cols);
            const firing = this.arena.mb.kcFiring[i];
            if (firing > 0) {
                this.kcCtx.fillStyle = '#38bdf8';
                this.kcCtx.shadowColor = '#38bdf8';
                this.kcCtx.shadowBlur = 6;
            } else {
                this.kcCtx.fillStyle = '#0f172a';
                this.kcCtx.shadowBlur = 0;
            }
            this.kcCtx.fillRect(c * cellW + 1, r * cellH + 1, cellW - 2, cellH - 2);
        }
        this.kcCtx.shadowBlur = 0;
    }

    renderCompass() {
        const w = this.compassCanvas.width;
        const h = this.compassCanvas.height;
        const cx = w / 2, cy = h / 2, r = w / 2 - 8;

        this.compassCtx.clearRect(0, 0, w, h);

        for (let i = 0; i < 16; i++) {
            const a1 = -Math.PI + (i * 2 * Math.PI) / 16;
            const a2 = a1 + (2 * Math.PI) / 16;
            const act = this.arena.cx.bumpProfile[i];

            this.compassCtx.fillStyle = `rgba(56, 189, 248, ${0.15 + act * 0.75})`;
            this.compassCtx.beginPath();
            this.compassCtx.moveTo(cx, cy);
            this.compassCtx.arc(cx, cy, r, a1, a2);
            this.compassCtx.closePath();
            this.compassCtx.fill();
        }

        this.compassCtx.strokeStyle = '#ffffff';
        this.compassCtx.lineWidth = 2.5;
        this.compassCtx.beginPath();
        this.compassCtx.moveTo(cx, cy);
        this.compassCtx.lineTo(cx + Math.cos(this.arena.fly.heading) * (r - 4), cy - Math.sin(this.arena.fly.heading) * (r - 4));
        this.compassCtx.stroke();
    }

    renderOscilloscope() {
        const w = this.scopeCanvas.width;
        const h = this.scopeCanvas.height;
        this.scopeCtx.clearRect(0, 0, w, h);

        this.scopeCtx.strokeStyle = 'rgba(255, 255, 255, 0.05)';
        this.scopeCtx.lineWidth = 1;
        this.scopeCtx.beginPath();
        this.scopeCtx.moveTo(0, h / 2); this.scopeCtx.lineTo(w, h / 2); this.scopeCtx.stroke();

        const channels = [
            { key: 'dna02', color: '#38bdf8', scale: 0.8 },
            { key: 'dnp09', color: '#4ade80', scale: 1.0 },
            { key: 'bpn', color: '#fbbf24', scale: 0.8 },
            { key: 'mdn', color: '#c084fc', scale: 1.0 },
            { key: 'dnp01', color: '#f43f5e', scale: 1.2 }
        ];

        for (const ch of channels) {
            this.scopeCtx.strokeStyle = ch.color;
            this.scopeCtx.lineWidth = 1.4;
            this.scopeCtx.beginPath();
            for (let i = 0; i < this.scopeHistory.length; i++) {
                const x = (i / (this.scopeHistory.length - 1)) * w;
                const val = this.scopeHistory[i][ch.key];
                const y = h / 2 - (val / 60.0) * (h / 2) * ch.scale;
                if (i === 0) this.scopeCtx.moveTo(x, y);
                else this.scopeCtx.lineTo(x, y);
            }
            this.scopeCtx.stroke();
        }
    }

    updateLearningCurve() {
        const w = this.curveCanvas.width;
        const h = this.curveCanvas.height;
        this.curveCtx.clearRect(0, 0, w, h);

        this.curveCtx.strokeStyle = 'rgba(255, 255, 255, 0.15)';
        this.curveCtx.lineWidth = 1;
        this.curveCtx.beginPath();
        this.curveCtx.moveTo(0, h / 2); this.curveCtx.lineTo(w, h / 2); this.curveCtx.stroke();

        if (this.learningTrials.length < 2) return;

        this.curveCtx.strokeStyle = '#38bdf8';
        this.curveCtx.lineWidth = 2.0;
        this.curveCtx.beginPath();

        for (let i = 0; i < this.learningTrials.length; i++) {
            const x = (i / (this.learningTrials.length - 1)) * (w - 20) + 10;
            const pi = this.learningTrials[i];
            const y = h / 2 - pi * (h / 2 - 10);
            if (i === 0) this.curveCtx.moveTo(x, y);
            else this.curveCtx.lineTo(x, y);

            this.curveCtx.fillStyle = '#f8fafc';
            this.curveCtx.beginPath();
            this.curveCtx.arc(x, y, 3.5, 0, 2 * Math.PI);
            this.curveCtx.fill();
        }
        this.curveCtx.stroke();
    }
}

// =============================================================================
// 8. APPLICATION INITIALIZATION & 60 FPS MAIN LOOP
// =============================================================================

window.addEventListener('load', () => {
    const arena = new ScientificBioArena('arenaCanvas');
    const hud = new ScientificHUD(arena);
    window.arena = arena;
    window.hud = hud;

    let isPaused = false;
    document.getElementById('btnPauseToggle').addEventListener('click', () => {
        isPaused = !isPaused;
        document.getElementById('btnPauseToggle').textContent = isPaused ? 'Resume' : 'Pause';
    });

    document.getElementById('btnResetArena').addEventListener('click', () => {
        arena.resetTrial();
    });

    function loop() {
        if (!isPaused) {
            arena.step(0.02);
            hud.update();
        }
        arena.render();
        requestAnimationFrame(loop);
    }
    requestAnimationFrame(loop);
});
