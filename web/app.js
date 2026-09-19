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
        this.goalHeading = 0.0;
        this.hasGoalHeading = false;
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

    step(flyHeading, flyYawRate, egocentricWind = 0, goalAngle = null, dt = 0.02) {
        if (this.isLesioned) {
            this.updateBump(this.headingBump + (Math.random() - 0.5) * 0.8);
            this.pfl3ErrorL = Math.random() * 0.5;
            this.pfl3ErrorR = Math.random() * 0.5;
            return (Math.random() - 0.5) * 2.0;
        }

        // Biological E-PG compass tracking (FlyWire / Seelig & Jayaraman 2015, Green et al. 2017)
        this.updateBump(flyHeading);

        let goalError = 0.0;
        if (goalAngle !== null && goalAngle !== undefined) {
            // Visual landmark / place goal navigation via PFL3 premotor interneurons
            const diff = ((goalAngle - this.headingBump + Math.PI) % (2 * Math.PI) + 2 * Math.PI) % (2 * Math.PI) - Math.PI;
            goalError = Math.max(-1.5, Math.min(1.5, diff));
            this.goalHeading = goalAngle;
            this.hasGoalHeading = true;
        } else if (Math.abs(egocentricWind) > 0.02) {
            // Upwind anemotaxis via Johnston's organ
            goalError = 0.60 * (0.45 * egocentricWind);
            this.hasGoalHeading = false;
        } else {
            this.hasGoalHeading = false;
        }

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
        this.currentTrial = 1;
        this.paradigmElapsedSec = 0.0;
        this.trialCompletionTimer = 0.0;
        this.trialHistory = [];
        this.fly.trail = [];
        this.collisionNormals = [];
        this.currentWalls = [];
        this.mb.reset(false);
        if (window.hud && window.hud.resetLearningCurve) {
            window.hud.resetLearningCurve(paradigmId);
        }

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
                    chosenArm: null,
                    latencyMs: null,
                    performanceIndex: 0.0,
                    shockPulse: 0.0
                };
                break;

            case 'y-maze':
                this.activeParadigmTitle = 'Y-Maze Spontaneous Alternation & Handedness';
                this.activeParadigmRef = 'Buchanan, Kain & de Bivort (Nature 2015)';
                this.worldBounds = { minX: 0, maxX: 120, minY: 0, maxY: 120 };
                this.fly.x = 60.0; this.fly.y = 60.0; this.fly.heading = Math.PI / 2; this.fly.speed = 12.0;
                this.paradigmStatus = 'EXPLORING ARM 0 (N)';
                this.currentWalls = [
                    new WallSegment([67.0, 100.0], [53.0, 100.0]),  // Cap 0 (North)
                    new WallSegment([53.0, 100.0], [49.61, 66.0]),  // Arm 0 Left -> Corner 01
                    new WallSegment([49.61, 66.0], [21.86, 46.06]), // Corner 01 -> Arm 1 Right
                    new WallSegment([21.86, 46.06], [28.86, 33.94]), // Cap 1 (South-West)
                    new WallSegment([28.86, 33.94], [60.0, 48.0]),  // Arm 1 Left -> Corner 12
                    new WallSegment([60.0, 48.0], [91.14, 33.94]),  // Corner 12 -> Arm 2 Right
                    new WallSegment([91.14, 33.94], [98.14, 46.06]), // Cap 2 (South-East)
                    new WallSegment([98.14, 46.06], [70.39, 66.0]), // Arm 2 Left -> Corner 20
                    new WallSegment([70.39, 66.0], [67.0, 100.0])   // Corner 20 -> Arm 0 Right
                ];
                this.paradigmState = {
                    armCounts: [0, 0, 0],
                    armSequence: [],
                    turnDirections: [],
                    lastZone: 'hub',
                    targetArm: 0,
                    headingTowardsHub: false,
                    lastArmIdx: 0,
                    sar: 0.0,
                    handedness: 0.0,
                    completing: false,
                    armTips: [
                        [60.0, 95.0],
                        [26.0, 41.0],
                        [94.0, 41.0]
                    ]
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
                    temp: 36.5,
                    learnedConfidence: 0.0,
                    searchAngle: 0.8
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

            case 'multisensory-sandbox':
                this.activeParadigmTitle = 'Multisensory Ingress & 6-Limb Biomechanics Benchmark';
                this.activeParadigmRef = 'Project NeuroFly v1.0 Integrated Sensorimotor Benchmark';
                this.worldBounds = { minX: -80, maxX: 80, minY: -80, maxY: 80 };
                this.fly.x = 0.0; this.fly.y = 0.0; this.fly.heading = 0.0; this.fly.speed = 12.0;
                this.windVector = [-15.0, 0.0];
                this.paradigmStatus = 'MULTI-SENSORY BENCHMARK ACTIVE';

                // 32 boundary circular segments (r=75) + 4 internal pillars (r=6) at (+-35, +-35)
                const wallsMb = [];
                const rOuter = 75.0;
                for (let i = 0; i < 32; i++) {
                    const a1 = (2 * Math.PI * i) / 32;
                    const a2 = (2 * Math.PI * (i + 1)) / 32;
                    wallsMb.push(new WallSegment([rOuter * Math.cos(a1), rOuter * Math.sin(a1)], [rOuter * Math.cos(a2), rOuter * Math.sin(a2)]));
                }
                const pillars = [[35.0, 35.0], [-35.0, 35.0], [-35.0, -35.0], [35.0, -35.0]];
                for (const pc of pillars) {
                    for (let i = 0; i < 8; i++) {
                        const a1 = (2 * Math.PI * i) / 8;
                        const a2 = (2 * Math.PI * (i + 1)) / 8;
                        wallsMb.push(new WallSegment([pc[0] + 6.0 * Math.cos(a1), pc[1] + 6.0 * Math.sin(a1)], [pc[0] + 6.0 * Math.cos(a2), pc[1] + 6.0 * Math.sin(a2)]));
                    }
                }
                this.currentWalls = wallsMb;

                this.paradigmState = {
                    foodPos: [45.0, 45.0],
                    repellentPos: [-45.0, -45.0],
                    pheromonePos: [45.0, -45.0],
                    hotspotPos: [-40.0, 40.0],
                    hotspotTemp: 38.5,
                    coolPos: [45.0, 45.0],
                    pillars: pillars,
                    temp: 24.0,
                    jointAngles: {
                        L1: { ctr: 0, fti: 80, tita: 35 },
                        L2: { ctr: 0, fti: 80, tita: 35 },
                        L3: { ctr: 0, fti: 80, tita: 35 },
                        R1: { ctr: 0, fti: 80, tita: 35 },
                        R2: { ctr: 0, fti: 80, tita: 35 },
                        R3: { ctr: 0, fti: 80, tita: 35 }
                    },
                    cuticularLoads: { L1: 1.85, L2: 0, L3: 1.85, R1: 0, R2: 1.85, R3: 0 },
                    manualActive: false,
                    overrideDna02: 0.0,
                    overrideThrust: 0.0,
                    overrideMdn: 0.0,
                    overrideGf: false,
                    overrideWings: 0.0,
                    overrideLegs: {},
                    coordinationScore: 0.94,
                    sensoryIntegrationScore: 0.88,
                    efficiencyScore: 0.82,
                    smoothnessScore: 0.91,
                    compositeScore: 88.5,
                    wallCollisions: 0,
                    totalDistance: 0.0,
                    totalEnergy: 0.0
                };
                break;
        }
    }

    resetTrial(advanceTrial = true, keepMemory = true) {
        if (advanceTrial) {
            const metric = this.getCanonicalMetricInfo();
            if (window.hud && window.hud.recordTrialData) {
                window.hud.recordTrialData(this.currentTrial, metric);
            }
            if (!this.trialHistory) this.trialHistory = [];
            this.trialHistory.push({
                trial: this.currentTrial,
                paradigm: this.activeParadigmId,
                metric: metric.value,
                rawMetric: this.getRawTrialMetric(),
                elapsed: this.paradigmElapsedSec
            });
            this.currentTrial += 1;
        }
        this.paradigmElapsedSec = 0.0;
        this.trialCompletionTimer = 0.0;
        this.fly.trail = [];
        this.collisionNormals = [];

        switch (this.activeParadigmId) {
            case 'open-arena':
                this.fly.x = 0; this.fly.y = 0; this.fly.heading = 0; this.fly.speed = 0;
                this.paradigmStatus = 'FORAGING';
                break;
            case 't-maze':
                this.fly.x = 70.0; this.fly.y = 15.0; this.fly.heading = Math.PI / 2; this.fly.speed = 10.0;
                if (this.paradigmState) {
                    this.paradigmState.firstChoice = null;
                    this.paradigmState.chosenArm = null;
                    this.paradigmState.latencyMs = null;
                    this.paradigmState.shockPulse = 0.0;
                    this.paradigmState.completing = false;
                }
                this.paradigmStatus = 'ASCENDING STEM';
                break;
            case 'y-maze':
                this.fly.x = 60.0; this.fly.y = 60.0; this.fly.heading = Math.PI / 2; this.fly.speed = 12.0;
                if (this.paradigmState) {
                    this.paradigmState.lastZone = 'hub';
                    this.paradigmState.targetArm = 0;
                    this.paradigmState.headingTowardsHub = false;
                    this.paradigmState.lastArmIdx = 0;
                    this.paradigmState.completing = false;
                }
                this.paradigmStatus = 'EXPLORING ARM 0 (N)';
                break;
            case 'heat-maze':
                this.fly.x = 50.0; this.fly.y = 50.0;
                this.fly.heading = Math.random() * 2 * Math.PI;
                this.fly.speed = 10.0;
                if (this.paradigmState) {
                    this.paradigmState.refugeReached = false;
                    this.paradigmState.escapeLatencyMs = null;
                    this.paradigmState.cumulativeDose = 0.0;
                    this.paradigmState.temp = 36.5;
                    this.paradigmState.completing = false;
                }
                this.paradigmStatus = 'HOT FLOOR (36.5°C)';
                break;
            case 'buridan':
                this.fly.x = 60.0; this.fly.y = 60.0; this.fly.heading = 0.0; this.fly.speed = 10.0;
                if (this.paradigmState) {
                    this.paradigmState.timeCenterMs = 0;
                    this.paradigmState.timePerimeterMs = 0;
                    this.paradigmState.targetStripe = 'east';
                }
                this.paradigmStatus = 'STRIPE FIXATION';
                break;
            case 'visual-operant':
                this.fly.x = 40.0; this.fly.y = 40.0; this.fly.heading = 0.0; this.fly.speed = 0.0;
                if (this.paradigmState) {
                    this.paradigmState.drumAngleDeg = 0.0;
                    this.paradigmState.timeSafeMs = 0;
                    this.paradigmState.timePunishedMs = 0;
                    this.paradigmState.laserActive = false;
                }
                this.paradigmStatus = 'SAFE QUADRANT (T)';
                break;
            case 'wind-tunnel':
                this.fly.x = 25.0; this.fly.y = 30.0; this.fly.heading = 0.0; this.fly.speed = 10.0;
                if (this.paradigmState) {
                    this.paradigmState.sourceReached = false;
                    this.paradigmState.timeToSourceMs = null;
                    this.paradigmState.completing = false;
                }
                this.paradigmStatus = 'SEARCHING (CAST)';
                break;
            case 'looming-escape':
                this.fly.x = 40.0; this.fly.y = 40.0; this.fly.heading = 0.0; this.fly.speed = 0.0;
                if (this.paradigmState) {
                    this.paradigmState.escapeInitiated = false;
                    this.paradigmState.timeToCollisionJumpMs = null;
                    this.paradigmState.tCollisionS = 1.2;
                    this.paradigmState.completing = false;
                }
                this.dn.escapeActive = false;
                this.dn.escapeTimer = 0.0;
                this.paradigmStatus = 'APPROACHING THREAT';
                break;
            case 'optomotor':
                this.fly.x = 45.0; this.fly.y = 45.0; this.fly.heading = 0.0; this.fly.speed = 0.0;
                if (this.paradigmState) {
                    this.paradigmState.drumAngleDeg = 0.0;
                    this.paradigmState.saccadeTimer = 0.0;
                }
                this.paradigmStatus = 'OPTO-STABILIZATION';
                break;
            case 'gap-crossing':
                this.fly.x = 12.0; this.fly.y = 10.0; this.fly.heading = 0.0; this.fly.speed = 8.0;
                if (this.paradigmState) {
                    this.paradigmState.isProbing = false;
                    this.paradigmState.decisionOutcome = null;
                    this.paradigmState.crossingSuccess = false;
                    this.paradigmState.probingDurationMs = 0;
                    this.paradigmState.completing = false;
                }
                this.paradigmStatus = 'APPROACHING CHASM';
                break;
            case 'circadian-dam':
                this.fly.x = 10.0; this.fly.y = 75.0; this.fly.heading = 0.0; this.fly.speed = 8.0;
                if (this.paradigmState) {
                    this.paradigmState.beamCrossings = 0;
                    this.paradigmState.totalSleepMin = 0;
                    this.paradigmState.consecutiveImmobileMin = 0;
                    this.paradigmState.inSleepBout = false;
                    this.paradigmState.sleepBouts = 0;
                }
                this.paradigmStatus = 'LOCOMOTING [AWAKE]';
                break;
            case 'courtship':
                this.fly.x = 7.0; this.fly.y = 9.0; this.fly.heading = 0.8; this.fly.speed = 8.0;
                if (this.paradigmState) {
                    this.paradigmState.totalSteps = 0;
                    this.paradigmState.courtshipActiveSteps = 0;
                    this.paradigmState.courtshipIndex = 0.0;
                    this.paradigmState.wingAngleDeg = 0.0;
                    this.paradigmState.rejectionKicks = 0;
                }
                this.paradigmStatus = 'APPROACHING FEMALE';
                break;
            case 'labyrinth':
                this.fly.x = 12.0; this.fly.y = 15.0; this.fly.heading = Math.PI / 2; this.fly.speed = 10.0;
                if (this.paradigmState) {
                    this.paradigmState.goalReached = false;
                    this.paradigmState.timeToGoalMs = null;
                    this.paradigmState.pathLength = 0;
                    this.paradigmState.lastPathX = 12.0;
                    this.paradigmState.lastPathY = 15.0;
                    this.paradigmState.wallCollisions = 0;
                    this.paradigmState.completing = false;
                }
                this.paradigmStatus = 'NAVIGATING MAZE';
                break;

            case 'multisensory-sandbox':
                this.fly.x = 0.0; this.fly.y = 0.0; this.fly.heading = 0.0; this.fly.speed = 12.0;
                if (this.paradigmState) {
                    this.paradigmState.totalDistance = 0.0;
                    this.paradigmState.totalEnergy = 0.0;
                    this.paradigmState.wallCollisions = 0;
                }
                this.paradigmStatus = 'MULTI-SENSORY BENCHMARK ACTIVE';
                break;
        }

        this.mb.reset(keepMemory);
        this.cx.headingBump = 0.0;
        this.dn.dna02L = 0.0;
        this.dn.dna02R = 0.0;
        this.dn.dna02Diff = 0.0;
        this.dn.mdn = 0.0;
        this.dn.escapeActive = false;
        if (window.hud && window.hud.update) window.hud.update();
    }

    getRawTrialMetric() {
        const pid = this.activeParadigmId;
        const p = this.paradigmState || {};
        if (pid === 'heat-maze') return p.escapeLatencyMs ? (p.escapeLatencyMs / 1000) : 25.0;
        if (pid === 't-maze') return p.performanceIndex !== undefined ? p.performanceIndex : 0.0;
        if (pid === 'y-maze') return p.sar || 0.0;
        if (pid === 'buridan') return p.centrophobism || 1.0;
        if (pid === 'visual-operant') return p.learningIndex || 0.0;
        if (pid === 'wind-tunnel') return p.timeToSourceMs ? (p.timeToSourceMs / 1000) : 25.0;
        if (pid === 'looming-escape') return p.timeToCollisionJumpMs || 0.0;
        if (pid === 'optomotor') return p.gain || 0.88;
        if (pid === 'gap-crossing') return p.gapWidthMm || 3.5;
        if (pid === 'circadian-dam') return p.totalSleepMin || 0.0;
        if (pid === 'courtship') return p.courtshipIndex || 0.0;
        if (pid === 'labyrinth') return p.timeToGoalMs ? (p.timeToGoalMs / 1000) : 35.0;
        if (pid === 'multisensory-sandbox') return p.compositeScore || 0.0;
        return this.mb.netValence;
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
        let paradigmGoalAngle = null;

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
                if (this.paradigmElapsedSec >= 25.0) {
                    this.resetTrial(true, true);
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

                if (this.fly.y < 43.0) {
                    // Ascend stem towards T-junction (70, 50)
                    paradigmGoalAngle = Math.PI / 2;
                    this.paradigmStatus = 'ASCENDING STEM';
                } else {
                    // At junction: decide arm based on Mushroom Body valence memory
                    if (p.chosenArm === null) {
                        if (this.mb.netValence < -0.05) {
                            p.chosenArm = 'arm_a'; // Conditioned avoidance of Arm B (shock) -> Choose Arm A
                        } else if (this.mb.netValence > 0.05) {
                            p.chosenArm = 'arm_a';
                        } else {
                            p.chosenArm = Math.random() < 0.5 ? 'arm_a' : 'arm_b';
                        }
                    }
                    // Steer West (PI) into Arm A or East (0.0) into Arm B
                    paradigmGoalAngle = (p.chosenArm === 'arm_a') ? Math.PI : 0.0;
                    this.paradigmStatus = `CHOOSING ${p.chosenArm === 'arm_a' ? 'ARM A' : 'ARM B'}`;
                }

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

                if (p.firstChoice !== null) {
                    if (this.trialCompletionTimer <= 0 && !p.completing) {
                        p.completing = true;
                        this.trialCompletionTimer = 1.4;
                    } else if (p.completing) {
                        this.trialCompletionTimer -= dt;
                        const armLabel = p.firstChoice === 'arm_a' ? 'ARM A (CS+ SUCROSE)' : 'ARM B (CS- SHOCK 60V)';
                        this.paradigmStatus = `${armLabel} (TRIAL ${this.currentTrial + 1} IN ${Math.max(0, this.trialCompletionTimer).toFixed(1)}s)`;
                        if (this.trialCompletionTimer <= 0) {
                            p.completing = false;
                            this.resetTrial(true, true);
                        }
                    }
                }
                if (this.paradigmElapsedSec >= 20.0) {
                    this.resetTrial(true, true);
                }
                break;
            }

            case 'y-maze': {
                const cx = 60.0, cy = 60.0;
                const dHub = Math.hypot(this.fly.x - cx, this.fly.y - cy);
                const tips = p.armTips || [[60.0, 95.0], [26.0, 41.0], [94.0, 41.0]];

                if (p.headingTowardsHub) {
                    paradigmGoalAngle = Math.atan2(cy - this.fly.y, cx - this.fly.x);
                    if (dHub <= 5.0) {
                        p.headingTowardsHub = false;
                        // Spontaneous alternation rule (~72% alternation probability)
                        const lastTurn = p.turnDirections.length > 0 ? p.turnDirections[p.turnDirections.length - 1] : (Math.random() < 0.5 ? 'L' : 'R');
                        const nextTurn = Math.random() < 0.72 ? (lastTurn === 'L' ? 'R' : 'L') : lastTurn;
                        const turnDelta = nextTurn === 'L' ? 1 : 2;
                        p.targetArm = (p.lastArmIdx + turnDelta) % 3;
                        p.lastZone = 'hub';
                        this.paradigmStatus = `HUB: TURNING TO ARM ${p.targetArm}`;
                    } else {
                        this.paradigmStatus = `RETURNING FROM ARM ${p.lastArmIdx}`;
                    }
                } else {
                    const targetTip = tips[p.targetArm];
                    paradigmGoalAngle = Math.atan2(targetTip[1] - this.fly.y, targetTip[0] - this.fly.x);
                    this.paradigmStatus = `EXPLORING ARM ${p.targetArm}`;

                    const dTip = Math.hypot(this.fly.x - targetTip[0], this.fly.y - targetTip[1]);
                    if (dTip < 6.5 || dHub >= 33.0) {
                        if (p.lastZone !== `arm_${p.targetArm}`) {
                            p.armCounts[p.targetArm] += 1;
                            p.armSequence.push(p.targetArm);
                            if (p.armSequence.length >= 2) {
                                const prev = p.armSequence[p.armSequence.length - 2];
                                const turnDiff = (p.targetArm - prev + 3) % 3;
                                p.turnDirections.push(turnDiff === 1 ? 'L' : 'R');
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
                            p.lastArmIdx = p.targetArm;
                            p.lastZone = `arm_${p.targetArm}`;
                        }
                        rewardSignal = 0.8;
                        p.headingTowardsHub = true;
                        paradigmGoalAngle = Math.atan2(cy - this.fly.y, cx - this.fly.x);
                        this.paradigmStatus = `ARM ${p.targetArm} REACHED! RETURNING`;
                    }
                }

                if (p.armSequence && p.armSequence.length >= 6) {
                    if (this.trialCompletionTimer <= 0 && !p.completing) {
                        p.completing = true;
                        this.trialCompletionTimer = 1.4;
                    } else if (p.completing) {
                        this.trialCompletionTimer -= dt;
                        this.paradigmStatus = `6 ARMS VISITED (SAR: ${((p.sar || 0) * 100).toFixed(0)}%) (TRIAL ${this.currentTrial + 1} IN ${Math.max(0, this.trialCompletionTimer).toFixed(1)}s)`;
                        if (this.trialCompletionTimer <= 0) {
                            p.completing = false;
                            this.resetTrial(true, true);
                        }
                    }
                }
                if (this.paradigmElapsedSec >= 60.0) {
                    this.resetTrial(true, true);
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
                        p.learnedConfidence = Math.min(1.0, (p.learnedConfidence || 0.0) + 0.35);
                        this.trialCompletionTimer = 1.4;
                    } else {
                        this.trialCompletionTimer -= dt;
                        this.paradigmStatus = `COOL REFUGE! ESCAPE: ${(p.escapeLatencyMs / 1000).toFixed(1)}s (TRIAL ${this.currentTrial + 1} IN ${Math.max(0, this.trialCompletionTimer).toFixed(1)}s)`;
                        if (this.trialCompletionTimer <= 0) {
                            this.resetTrial(true, true);
                        }
                    }
                    rewardSignal = 1.0;
                } else {
                    const excess = distRef - p.refugeRadius;
                    const gauss = Math.exp(-(excess * excess) / (2 * 8.0 * 8.0));
                    p.temp = 36.5 - (36.5 - 24.0) * gauss;
                    punishmentSignal = Math.max(0.0, (p.temp - 25.0) / 11.5);
                    p.cumulativeDose += Math.max(0.0, p.temp - 25.0) * dt;
                    this.paradigmStatus = `HOT FLOOR (${p.temp.toFixed(1)}°C)`;
                    if (this.paradigmElapsedSec >= 25.0) {
                        p.escapeLatencyMs = 25000;
                        this.resetTrial(true, true);
                    }
                }

                // Ofstad & Zuker (Nature 2011) Place Navigation:
                if (p.learnedConfidence && p.learnedConfidence > 0) {
                    // Central Complex landmark triangulation towards cool refuge
                    paradigmGoalAngle = Math.atan2(p.refugePos[1] - this.fly.y, p.refugePos[0] - this.fly.x);
                } else {
                    // Naive exploratory area search
                    if (p.searchAngle === undefined) p.searchAngle = this.fly.heading;
                    p.searchAngle += (Math.random() - 0.5) * 1.5 * dt;
                    const dCenter = Math.hypot(this.fly.x - 60.0, this.fly.y - 60.0);
                    if (dCenter > 42.0) {
                        p.searchAngle = Math.atan2(60.0 - this.fly.y, 60.0 - this.fly.x);
                    }
                    paradigmGoalAngle = p.searchAngle;
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

                if (!p.targetStripe) p.targetStripe = 'east';
                if (p.targetStripe === 'east') {
                    paradigmGoalAngle = 0.0;
                    if (this.fly.x >= 95.0 || (distC > 46.0 && this.fly.x > 60.0)) {
                        p.targetStripe = 'west';
                        p.stripeCrossings++;
                    }
                } else {
                    paradigmGoalAngle = Math.PI;
                    if (this.fly.x <= 25.0 || (distC > 46.0 && this.fly.x < 60.0)) {
                        p.targetStripe = 'east';
                        p.stripeCrossings++;
                    }
                }

                const side = Math.abs(normB0) < Math.PI / 2 ? 0 : 1;
                if (p.lastSide !== null && p.lastSide !== side) p.stripeCrossings++;
                p.lastSide = side;
                this.paradigmStatus = distC < 25.0 ? 'OPEN CENTER' : `STRIPE FIXATION (${p.targetStripe.toUpperCase()})`;
                if (this.paradigmElapsedSec >= 15.0) {
                    this.resetTrial(true, true);
                }
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
                if (this.paradigmElapsedSec >= 15.0) {
                    this.resetTrial(true, true);
                }
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
                if (Math.hypot(this.fly.x - p.nozzlePos[0], this.fly.y - p.nozzlePos[1]) < 14.0) {
                    if (!p.sourceReached) {
                        p.sourceReached = true;
                        p.timeToSourceMs = this.paradigmElapsedSec * 1000;
                        this.trialCompletionTimer = 1.4;
                    } else {
                        this.trialCompletionTimer -= dt;
                        this.paradigmStatus = `SOURCE REACHED! ${(p.timeToSourceMs / 1000).toFixed(1)}s (TRIAL ${this.currentTrial + 1} IN ${Math.max(0, this.trialCompletionTimer).toFixed(1)}s)`;
                        if (this.trialCompletionTimer <= 0) {
                            this.resetTrial(true, true);
                        }
                    }
                    rewardSignal = 1.0;
                }
                if (this.paradigmElapsedSec >= 25.0) {
                    this.resetTrial(true, true);
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

                if (this.dn.escapeActive || p.escapeInitiated) {
                    if (this.trialCompletionTimer <= 0 && !p.completing) {
                        p.completing = true;
                        this.trialCompletionTimer = 1.6;
                    } else if (p.completing) {
                        this.trialCompletionTimer -= dt;
                        this.paradigmStatus = `GF ESCAPE TAKEOFF! (TRIAL ${this.currentTrial + 1} IN ${Math.max(0, this.trialCompletionTimer).toFixed(1)}s)`;
                        if (this.trialCompletionTimer <= 0) {
                            p.completing = false;
                            this.resetTrial(true, true);
                        }
                    }
                }
                if (this.paradigmElapsedSec > p.tCollisionS + 0.6) {
                    this.resetTrial(true, true);
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
                if (this.paradigmElapsedSec >= 15.0) {
                    this.resetTrial(true, true);
                }
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

                if (p.crossingSuccess || p.decisionOutcome === 'ABORT') {
                    if (this.trialCompletionTimer <= 0 && !p.completing) {
                        p.completing = true;
                        this.trialCompletionTimer = 1.4;
                    } else if (p.completing) {
                        this.trialCompletionTimer -= dt;
                        const resText = p.crossingSuccess ? 'SUCCESSFUL STEP-OVER' : 'ABORT 180° TURN';
                        this.paradigmStatus = `${resText} (TRIAL ${this.currentTrial + 1} IN ${Math.max(0, this.trialCompletionTimer).toFixed(1)}s)`;
                        if (this.trialCompletionTimer <= 0) {
                            p.completing = false;
                            this.resetTrial(true, true);
                        }
                    }
                }
                if (this.paradigmElapsedSec >= 15.0) {
                    this.resetTrial(true, true);
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
                if (this.paradigmElapsedSec >= 20.0) {
                    this.resetTrial(true, true);
                }
                break;
            }

            case 'courtship': {
                p.totalSteps += 1;
                const dFem = Math.hypot(this.fly.x - p.femalePos[0], this.fly.y - p.femalePos[1]);
                paradigmGoalAngle = Math.atan2(p.femalePos[1] - this.fly.y, p.femalePos[0] - this.fly.x);

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
                if (this.paradigmElapsedSec >= 15.0) {
                    this.resetTrial(true, true);
                }
                break;
            }

            case 'labyrinth': {
                const distGoal = Math.hypot(this.fly.x - p.goalPos[0], this.fly.y - p.goalPos[1]);
                odorA = Math.exp(-distGoal / 35.0);
                paradigmGoalAngle = Math.atan2(p.goalPos[1] - this.fly.y, p.goalPos[0] - this.fly.x);

                if (distGoal < 12.0) {
                    if (!p.goalReached) {
                        p.goalReached = true;
                        p.timeToGoalMs = this.paradigmElapsedSec * 1000;
                        this.trialCompletionTimer = 1.4;
                    } else {
                        this.trialCompletionTimer -= dt;
                        this.paradigmStatus = `FOOD GOAL REACHED! ${(p.timeToGoalMs / 1000).toFixed(1)}s (TRIAL ${this.currentTrial + 1} IN ${Math.max(0, this.trialCompletionTimer).toFixed(1)}s)`;
                        if (this.trialCompletionTimer <= 0) {
                            this.resetTrial(true, true);
                        }
                    }
                    rewardSignal = 1.0;
                }
                if (this.paradigmElapsedSec >= 35.0) {
                    this.resetTrial(true, true);
                }

                p.pathLength += Math.hypot(this.fly.x - p.lastPathX, this.fly.y - p.lastPathY);
                p.lastPathX = this.fly.x;
                p.lastPathY = this.fly.y;
                const straightDist = Math.hypot(p.goalPos[0] - 12.0, p.goalPos[1] - 15.0);
                p.tortuosity = Math.max(1.0, p.pathLength / straightDist);
                break;
            }

            case 'multisensory-sandbox': {
                const da = Math.hypot(this.fly.x - p.foodPos[0], this.fly.y - p.foodPos[1]);
                const db = Math.hypot(this.fly.x - p.repellentPos[0], this.fly.y - p.repellentPos[1]);
                const dc = Math.hypot(this.fly.x - p.pheromonePos[0], this.fly.y - p.pheromonePos[1]);
                odorA = Math.exp(-(da * da) / (2 * 20 * 20));
                odorB = Math.exp(-(db * db) / (2 * 20 * 20));
                p.odorCva = 0.8 * Math.exp(-(dc * dc) / (2 * 18 * 18));

                const dh = Math.hypot(this.fly.x - p.hotspotPos[0], this.fly.y - p.hotspotPos[1]);
                const dcool = Math.hypot(this.fly.x - p.coolPos[0], this.fly.y - p.coolPos[1]);
                const tHot = (p.hotspotTemp - 24.0) * Math.exp(-(dh * dh) / (2 * 18 * 18));
                const tCool = -2.0 * Math.exp(-(dcool * dcool) / (2 * 12 * 12));
                p.temp = Math.max(20.0, Math.min(42.0, 24.0 + tHot + tCool));

                const upwind = Math.atan2(-this.windVector[1], -this.windVector[0]);
                egocentricWind = (((upwind - this.fly.heading + Math.PI) % (2 * Math.PI) + 2 * Math.PI) % (2 * Math.PI) - Math.PI);

                paradigmGoalAngle = Math.atan2(p.foodPos[1] - this.fly.y, p.foodPos[0] - this.fly.x);

                if (odorA > 0.5 && p.temp < 25.0) rewardSignal = 1.0;
                if (p.temp > 35.0 || odorB > 0.5) punishmentSignal = 1.0;

                if (p.manualActive) {
                    if (Math.abs(p.overrideDna02) > 0.05) {
                        this.dn.dna02Diff = p.overrideDna02 * 60.0;
                    }
                    if (p.overrideThrust > 0.05) {
                        this.dn.dnp09 = p.overrideThrust * 65.0;
                    }
                    if (p.overrideMdn > 0.05) {
                        this.dn.mdn = p.overrideMdn * 50.0;
                    }
                    if (p.overrideGf) {
                        this.dn.escapeActive = true;
                        this.dn.escapeTimer = 0.4;
                        p.overrideGf = false;
                    }
                }

                // 6-Leg Joint Kinematics Calculation
                const phiA = this.cpg.phaseA;
                const phiB = this.cpg.phaseB;
                const legPhases = { L1: phiA, R2: phiA, L3: phiA, R1: phiB, L2: phiB, R3: phiB };
                for (const [leg, phi] of Object.entries(legPhases)) {
                    p.jointAngles[leg] = {
                        ctr: 22.0 * Math.sin(phi),
                        fti: 80.0 + 32.0 * Math.cos(phi),
                        tita: 38.0 - 14.0 * Math.sin(phi)
                    };
                    p.cuticularLoads[leg] = this.cpg.legStates[leg] ? 1.85 : 0.0;
                }

                // Benchmark Scoring
                p.totalDistance += this.fly.speed * dt;
                p.totalEnergy += (this.cpg.steppingFreq * 0.8 + this.fly.speed * 0.5) * dt;

                const headingError = Math.abs(((paradigmGoalAngle - this.fly.heading + Math.PI) % (2 * Math.PI)) - Math.PI);
                const sensoryAlign = Math.cos(headingError * 0.5);
                p.sensoryIntegrationScore = Math.max(0.0, Math.min(1.0, 0.98 * p.sensoryIntegrationScore + 0.02 * sensoryAlign));

                const coord = 0.92 + 0.08 * (this.cpg.steppingFreq >= 6.0 ? 1.0 : 0.5);
                p.coordinationScore = Math.max(0.0, Math.min(1.0, 0.99 * p.coordinationScore + 0.01 * coord));

                const eff = Math.min(1.0, p.totalDistance / Math.max(1.0, p.totalEnergy * 10.0));
                p.efficiencyScore = Math.max(0.0, Math.min(1.0, 0.99 * p.efficiencyScore + 0.01 * eff));

                p.smoothnessScore = 0.92;
                p.compositeScore = (0.30 * p.coordinationScore + 0.25 * p.sensoryIntegrationScore + 0.25 * p.efficiencyScore + 0.20 * p.smoothnessScore) * 100.0;

                this.paradigmStatus = p.manualActive ? 'MANUAL NEURO-STIMULATION ACTIVE' : `BENCHMARK SCORE: ${p.compositeScore.toFixed(1)} / 100`;
                break;
            }
        }

        this.mb.encodeOdor(odorA * 4.0, odorB * 4.0);
        const netValence = this.mb.forward();
        this.mb.stepPlasticity(rewardSignal, punishmentSignal, 1.0 + (1.0 - this.fly.energy), dt);

        const goalError = this.cx.step(this.fly.heading, this.fly.yawRate, egocentricWind, paradigmGoalAngle, dt);

        const hasOdor = (odorA + odorB) > 0.04;
        let chemotaxisSteer = 0.0;
        if (this.activeParadigmId === 'open-arena') {
            chemotaxisSteer = -0.55 * ((odorA - odorB) * 3.5);
            if (hasOdor) chemotaxisSteer *= Math.max(0.2, 1.0 + 0.35 * netValence);
        }

        const totalSteerError = (hasOdor && this.activeParadigmId === 'open-arena') ? (0.6 * goalError + 0.4 * chemotaxisSteer) : goalError;
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
                const fwdThrust = isReversing ? -12.0 : (1.0 + (this.dn.dnp09 + this.dn.bpn) / 35.0) * 18.0;
                const drag = 2.5 * this.fly.speed;
                this.fly.speed += (fwdThrust - drag) * dt;
                this.fly.speed = Math.max(-8.0, Math.min(32.0, this.fly.speed));

                const yawTorque = 0.14 * this.dn.dna02Diff;
                this.fly.yawRate += (yawTorque - 4.5 * this.fly.yawRate) * dt;
            }

            this.fly.heading = ((this.fly.heading + this.fly.yawRate * dt + Math.PI) % (2 * Math.PI) + 2 * Math.PI) % (2 * Math.PI) - Math.PI;
            this.cx.updateBump(this.fly.heading);

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

        // Subsample trail to preserve long visible history even at ultra-high speeds up to 100x
        const curSpeed = (window.hud && window.hud.simSpeed) ? window.hud.simSpeed : 1.0;
        const trailInterval = curSpeed >= 50 ? 5 : (curSpeed >= 10 ? 2 : 1);
        if (this.stepCount % trailInterval === 0) {
            this.fly.trail.push({ x: this.fly.x, y: this.fly.y, speed: this.fly.speed, escape: this.dn.escapeActive });
            if (this.fly.trail.length > 200) {
                this.fly.trail = this.fly.trail.slice(-150);
            }
        }

        const metricInfo = this.getCanonicalMetricInfo();
        const telemInterval = curSpeed >= 50 ? 10 : (curSpeed >= 10 ? 5 : 2);
        if (this.isRecording || this.stepCount % telemInterval === 0) {
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
            if (this.telemetryBuffer.length > 4500) {
                this.telemetryBuffer = this.telemetryBuffer.slice(-4000);
            }
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
            case 'multisensory-sandbox':
                return {
                    label: 'Benchmark Score',
                    value: (p.compositeScore || 0).toFixed(1) + ' / 100',
                    sub: `Coord: ${((p.coordinationScore || 0) * 100).toFixed(0)}% | Sensory: ${((p.sensoryIntegrationScore || 0) * 100).toFixed(0)}%`
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
            case 'multisensory-sandbox': this.renderMultisensorySandbox(this.ctx); break;
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

        const tips = [
            { pos: [60.0, 95.0], label: `Arm 0 (N): ${this.paradigmState.armCounts[0]}`, color: '#22c55e' },
            { pos: [26.0, 41.0], label: `Arm 1 (SW): ${this.paradigmState.armCounts[1]}`, color: '#38bdf8' },
            { pos: [94.0, 41.0], label: `Arm 2 (SE): ${this.paradigmState.armCounts[2]}`, color: '#fbbf24' }
        ];

        // Draw glowing arm terminal goals
        tips.forEach(t => {
            const s = this.worldToScreen(t.pos[0], t.pos[1]);
            const grad = ctx.createRadialGradient(s.x, s.y, 2, s.x, s.y, 35);
            grad.addColorStop(0, `${t.color}66`);
            grad.addColorStop(1, `${t.color}00`);
            ctx.fillStyle = grad;
            ctx.beginPath(); ctx.arc(s.x, s.y, 35, 0, 2 * Math.PI); ctx.fill();

            ctx.fillStyle = t.color;
            ctx.beginPath(); ctx.arc(s.x, s.y, 6, 0, 2 * Math.PI); ctx.fill();

            ctx.font = 'bold 10px monospace';
            ctx.fillText(t.label, s.x - 36, s.y - 10);
        });

        // Hub circle
        const hubS = this.worldToScreen(60.0, 60.0);
        ctx.strokeStyle = 'rgba(192, 132, 252, 0.6)';
        ctx.lineWidth = 1.5;
        ctx.beginPath(); ctx.arc(hubS.x, hubS.y, 16, 0, 2 * Math.PI); ctx.stroke();
        ctx.fillStyle = 'rgba(192, 132, 252, 0.8)';
        ctx.font = '9px monospace';
        ctx.fillText('HUB', hubS.x - 9, hubS.y + 3);
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

        renderMultisensorySandbox(ctx) {
        const p = this.paradigmState;
        if (!p) return;

        // 1. Boundary circle & walls
        ctx.strokeStyle = '#38bdf8';
        ctx.lineWidth = 2.0;
        ctx.shadowColor = '#38bdf8';
        ctx.shadowBlur = 6;
        for (const w of this.currentWalls) {
            const p1 = this.worldToScreen(w.p1[0], w.p1[1]);
            const p2 = this.worldToScreen(w.p2[0], w.p2[1]);
            ctx.beginPath(); ctx.moveTo(p1.x, p1.y); ctx.lineTo(p2.x, p2.y); ctx.stroke();
        }
        ctx.shadowBlur = 0;

        // 2. Thermal Gradients: Hotspot (-40, 40) in red/amber & Cool Refuge (45, 45) in teal
        const sHot = this.worldToScreen(p.hotspotPos[0], p.hotspotPos[1]);
        const gradHot = ctx.createRadialGradient(sHot.x, sHot.y, 4, sHot.x, sHot.y, 55);
        gradHot.addColorStop(0, 'rgba(244, 63, 94, 0.45)');
        gradHot.addColorStop(1, 'rgba(244, 63, 94, 0.0)');
        ctx.fillStyle = gradHot;
        ctx.beginPath(); ctx.arc(sHot.x, sHot.y, 55, 0, 2 * Math.PI); ctx.fill();

        const sCool = this.worldToScreen(p.coolPos[0], p.coolPos[1]);
        const gradCool = ctx.createRadialGradient(sCool.x, sCool.y, 4, sCool.x, sCool.y, 45);
        gradCool.addColorStop(0, 'rgba(34, 197, 94, 0.45)');
        gradCool.addColorStop(1, 'rgba(34, 197, 94, 0.0)');
        ctx.fillStyle = gradCool;
        ctx.beginPath(); ctx.arc(sCool.x, sCool.y, 45, 0, 2 * Math.PI); ctx.fill();

        // 3. Olfactory Markers
        // Food CS+ at (45, 45)
        ctx.fillStyle = '#22c55e';
        ctx.beginPath(); ctx.arc(sCool.x, sCool.y, 7, 0, 2 * Math.PI); ctx.fill();
        ctx.fillStyle = '#f8fafc'; ctx.font = '8px monospace';
        ctx.fillText('ODOR A (FOOD)', sCool.x - 28, sCool.y + 16);

        // Repellent CS- at (-45, -45)
        const sRepel = this.worldToScreen(p.repellentPos[0], p.repellentPos[1]);
        ctx.fillStyle = '#ef4444';
        ctx.beginPath(); ctx.arc(sRepel.x, sRepel.y, 7, 0, 2 * Math.PI); ctx.fill();
        ctx.fillText('ODOR B (ALARM)', sRepel.x - 30, sRepel.y + 16);

        // cVA Pheromone at (45, -45)
        const sPhero = this.worldToScreen(p.pheromonePos[0], p.pheromonePos[1]);
        ctx.fillStyle = '#c084fc';
        ctx.beginPath(); ctx.arc(sPhero.x, sPhero.y, 6, 0, 2 * Math.PI); ctx.fill();
        ctx.fillText('cVA PHEROMONE', sPhero.x - 30, sPhero.y + 16);

        // 4. 4 Visual Pillars
        const pillarColors = ['#38bdf8', '#f97316', '#a855f7', '#22c55e'];
        const pillarLabels = ['NE', 'NW', 'SW', 'SE'];
        for (let i = 0; i < p.pillars.length; i++) {
            const sp = this.worldToScreen(p.pillars[i][0], p.pillars[i][1]);
            ctx.fillStyle = pillarColors[i];
            ctx.beginPath(); ctx.arc(sp.x, sp.y, 6 * sp.scale, 0, 2 * Math.PI); ctx.fill();
            ctx.strokeStyle = '#ffffff'; ctx.lineWidth = 1; ctx.stroke();
            ctx.fillStyle = '#ffffff'; ctx.font = '8px monospace';
            ctx.fillText(pillarLabels[i], sp.x - 5, sp.y - 10);
        }

        // 5. Wind Flow Arrows
        ctx.strokeStyle = 'rgba(56, 189, 248, 0.4)';
        ctx.lineWidth = 1.2;
        ctx.setLineDash([3, 3]);
        for (let yOff = -40; yOff <= 40; yOff += 20) {
            const sw = this.worldToScreen(50, yOff);
            const ew = this.worldToScreen(-50, yOff);
            ctx.beginPath(); ctx.moveTo(sw.x, sw.y); ctx.lineTo(ew.x, ew.y); ctx.stroke();
        }
        ctx.setLineDash([]);

        // Collision normals
        ctx.strokeStyle = '#fbbf24';
        ctx.lineWidth = 1.5;
        for (const cn of this.collisionNormals) {
            const scn = this.worldToScreen(cn.x, cn.y);
            ctx.beginPath(); ctx.moveTo(scn.x, scn.y); ctx.lineTo(scn.x + cn.nx * 14, scn.y - cn.ny * 14); ctx.stroke();
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


// =============================================================================
// 7. EXPERIMENT DIRECTORS GUIDE & METADATA DICTIONARY
// =============================================================================

const EXPERIMENT_GUIDES = {
    'open-arena': {
        title: "Open Arena Multi-Modal Foraging",
        ref: "Budick & Dickinson (2006); Maimon et al. (Nature 2010)",
        whatToWatch: [
            "Watch fly track green food patches (Odor A) using bilateral antennal gradient comparison.",
            "Notice avoidance of red alarm pheromone (Odor B) emitted near danger sites.",
            "Observe emergency ballistic escape sprints when looming predator threats approach."
        ],
        params: [
            { key: 'predatorSpeed', label: 'Predator Speed', min: 10, max: 60, step: 5, val: 25, unit: 'mm/s', apply: (a, v) => { a.predators.forEach(p => { const sp = Math.hypot(p.vx, p.vy) || 1; p.vx = (p.vx / sp) * v; p.vy = (p.vy / sp) * v; }); } },
            { key: 'windVelocity', label: 'Wind Velocity', min: 0, max: 40, step: 5, val: 15, unit: 'mm/s', apply: (a, v) => { a.windVector = [-v, 0]; } }
        ]
    },
    't-maze': {
        title: "T-Maze Olfactory Conditioning",
        ref: "Tully & Quinn (1985) J. Comp. Physiol. A; Dudai (1976)",
        whatToWatch: [
            "Fly ascends vertical stem and pauses at the decision bifurcation line.",
            "Left Arm dispenses appetitive Odor A (CS+) paired with green sucrose reward.",
            "Right Arm dispenses aversive Odor B (CS-) paired with red pulsing electroshock.",
            "Watch anti-Hebbian depression shift net behavioral valence from 0.00 toward +1.00."
        ],
        params: [
            { key: 'shockPulse', label: 'Shock Voltage', min: 0, max: 100, step: 10, val: 60, unit: 'V', apply: (a, v) => { a.paradigmState.shockPulse = v / 100; } }
        ]
    },
    'y-maze': {
        title: "Y-Maze Spontaneous Alternation",
        ref: "Buchanan, Kain & de Bivort (Nature 2015); Churgin (2017)",
        whatToWatch: [
            "Fly explores 3 symmetric arms oriented at 120° intervals.",
            "Central Complex Protocerebral Bridge Delta7 interneurons promote alternating triads (A->B->C).",
            "Look for high Spontaneous Alternation Rate (SAR > 0.60) across consecutive choices.",
            "DNa02 premotor firing asymmetry sets individual fly idiosyncratic turn handedness."
        ],
        params: [
            { key: 'turnBias', label: 'Premotor Turn Bias', min: -20, max: 20, step: 2, val: 0, unit: 'Hz', apply: (a, v) => { a.dn.dna02Diff += v; } }
        ]
    },
    'heat-maze': {
        title: "Thermal Heat-Maze Place Learning",
        ref: "Ofstad, Zuker & Reiser (Nature 2011) Nature 474:204–207",
        whatToWatch: [
            "Circular floor is an aversive heated bath (36.5°C red glow), driving nociceptive PPL1 dopamine.",
            "Fly uses 4 distal perimeter visual stripes (0°, 90°, 180°, 270°) to orient its E-PG heading compass.",
            "Watch the dashed vector guiding the fly toward the 24.0°C cool target refuge.",
            "Stepping onto the cool tile triggers an immediate PAM pain-relief reward burst!"
        ],
        params: [
            { key: 'floorTemp', label: 'Floor Temperature', min: 28, max: 42, step: 0.5, val: 36.5, unit: '°C', apply: (a, v) => { a.paradigmState.temp = v; } },
            { key: 'refugeRadius', label: 'Refuge Radius', min: 6, max: 15, step: 1, val: 9, unit: 'mm', apply: (a, v) => { a.paradigmState.refugeRadius = v; } }
        ]
    },
    'buridan': {
        title: "Buridan's Visual Landmark Fixation",
        ref: "Götz (1980); Colomb & Brembs (2012) PLoS ONE",
        whatToWatch: [
            "Elevated circular platform is surrounded by an inescapable dark blue water moat.",
            "Two opposing vertical high-contrast black stripes are positioned at 0° and 180°.",
            "Fly fixates on one stripe, walks toward it, then turns 180° to oscillate between them.",
            "Notice the fly avoids the open center (centrophobism index CI > 0.70)."
        ],
        params: [
            { key: 'platformRadius', label: 'Platform Radius', min: 35, max: 60, step: 5, val: 50, unit: 'mm', apply: (a, v) => { a.paradigmState.platformRadius = v; } }
        ]
    },
    'visual-operant': {
        title: "Visual Operant Flight Simulator",
        ref: "Wolf & Heisenberg (1991); Liu et al. (Nature 2006)",
        whatToWatch: [
            "A tethered fly controls a 360° panoramic pattern drum via its own yaw torque.",
            "Facing the inverted 'T' triggers an intense infrared laser heating pulse.",
            "Facing the upright 'T' is safe.",
            "Watch fly learn to exert corrective yaw torque to stabilize arena in safe quadrants!"
        ],
        params: [
            { key: 'couplingGain', label: 'Yaw Coupling Gain', min: 50, max: 200, step: 10, val: 120, unit: '°/s', apply: (a, v) => { a.paradigmState.couplingGain = v; } }
        ]
    },
    'wind-tunnel': {
        title: "Wind Tunnel Plume Tracking (Surge-Cast)",
        ref: "Alvarez-Salvado et al. (2018); Demir et al. (eLife 2020)",
        whatToWatch: [
            "Downwind airflow (-25 mm/s) channels intermittent odor puffs from upstream nozzle.",
            "Upon contacting an odor filament (plume ON), fly UPWIND SURGES (DNp09 active).",
            "When plume is lost (plume OFF), fly executes CROSSWIND CASTING zigzags (DNa02)!"
        ],
        params: [
            { key: 'windVelocity', label: 'Wind Velocity', min: 10, max: 50, step: 5, val: 25, unit: 'mm/s', apply: (a, v) => { a.paradigmState.windFlow = [-v, 0]; } }
        ]
    },
    'looming-escape': {
        title: "Visual Looming Giant Fiber Escape",
        ref: "Card & Dickinson (PNAS 2008); von Reyn et al. (Nature 2014)",
        whatToWatch: [
            "An approaching predator shadow expands with angular velocity dTheta/dt.",
            "Lobula LPLC2/Col4 neurons drive Giant Fiber membrane potential ramp.",
            "The instant expansion hits 65°, Giant Fiber fires an all-or-none spike, commanding jump takeoff!"
        ],
        params: [
            { key: 'rOverV', label: 'Looming r/v Ratio', min: 10, max: 50, step: 5, val: 25, unit: 'ms', apply: (a, v) => { a.paradigmState.rOverVS = v / 1000; } }
        ]
    },
    'optomotor': {
        title: "Optomotor Gaze Stabilization",
        ref: "Götz (1964); Kim et al. (Nature 2017) Nature 551:517–521",
        whatToWatch: [
            "Surrounding cylindrical drum rotates with high-contrast vertical grating stripes.",
            "T4/T5 motion cells drive Lobula Plate Tangential Cells (HS/VS) compensatory turning.",
            "When fly makes a voluntary saccade, an ascending efference copy shunts >80% of retinal slip!"
        ],
        params: [
            { key: 'drumSpeed', label: 'Grating Velocity', min: -60, max: 60, step: 5, val: 30, unit: '°/s', apply: (a, v) => { a.paradigmState.drumSpeedDegS = v; } }
        ]
    },
    'gap-crossing': {
        title: "Gap Crossing & Spatial Motor Planning",
        ref: "Pick & Strauss (Current Biology 2005); Triphan et al. (2010)",
        whatToWatch: [
            "Fly advances along an elevated linear runway toward a physical abyss.",
            "Antennae and front legs reach forward to probe the gap.",
            "If gap width is < 3.8 mm, Central Complex triggers step-over; if > 4.2 mm, it aborts!"
        ],
        params: [
            { key: 'gapWidth', label: 'Chasm Width', min: 2.0, max: 5.5, step: 0.2, val: 3.5, unit: 'mm', apply: (a, v) => { a.paradigmState.gapWidthMm = v; } }
        ]
    },
    'circadian-dam': {
        title: "Circadian Locomotor Sleep/Wake Monitor",
        ref: "Konopka & Benzer (PNAS 1971); Allada & Chung (2010)",
        whatToWatch: [
            "Cylindrical glass capillary tube equipped with mid-tube infrared optical beam break.",
            "Fly shuttles between sucrose food plug and cotton stopper.",
            "Watch morning and evening anticipation peaks followed by consolidated sleep bouts (>= 5 min)."
        ],
        params: [
            { key: 'dayNight', label: 'Day / Night Light', min: 0, max: 1, step: 1, val: 1, unit: ' (0=DD, 1=LD)', apply: (a, v) => { a.paradigmState.isLightsOn = (v === 1); } }
        ]
    },
    'courtship': {
        title: "Courtship Conditioning & Pheromone Memory",
        ref: "Siegel & Hall (1979); Keleman et al. (Nature 2007)",
        whatToWatch: [
            "Male fly approaches female in circular mating chamber.",
            "Male extends unilateral wing to vibrate courtship song (P1 neurons active).",
            "Mated female emits anti-aphrodisiac cVA and kicks, delivering aversive dopaminergic conditioning."
        ],
        params: [
            { key: 'femaleMated', label: 'Female Mated State', min: 0, max: 1, step: 1, val: 0, unit: ' (0=Virgin, 1=Mated)', apply: (a, v) => { a.paradigmState.isFemaleVirgin = (v === 0); } }
        ]
    },
    'labyrinth': {
        title: "Corridor Obstacle Labyrinth",
        ref: "Biomimetic Complex Multi-Junction Navigation",
        whatToWatch: [
            "Challenging 4-decision junction labyrinth with blind alleys and dead ends.",
            "Watch tangential sliding velocity resolution (mu=0.5, eps=0.1) prevent sticking or tunneling.",
            "Watch fly use Johnston's organ wall deflection to recover from dead ends toward green goal!"
        ],
        params: [
            { key: 'friction', label: 'Wall Friction', min: 0.1, max: 0.9, step: 0.1, val: 0.5, unit: 'mu', apply: (a, v) => { a.currentWalls.forEach(w => w.friction = v); } }
        ]
    },
    'multisensory-sandbox': {
        title: "Multisensory Ingress & Limb Biomechanics Sandbox",
        ref: "Project NeuroFly v1.0 Benchmark (FlyWire & MaleCNS v1.0 Architecture)",
        whatToWatch: [
            "Full multi-sensory cue integration: Food Odor A, Repellent Odor B, cVA Pheromone, Thermal Gradient, and Vector Wind.",
            "Inspect 6 articulated tripod legs with real-time Coxa, Femur, and Tibia joint angle flexions.",
            "Toggle between Autonomous Connectome Mode and Direct Neuro-Stimulation / Limb Override Deck.",
            "Evaluate composite benchmark score across Coordination, Sensory Integration, Smoothness, and Efficiency."
        ],
        params: [
            { key: 'windMagnitude', label: 'Wind Velocity', min: 0, max: 40, step: 5, val: 15, unit: 'mm/s', apply: (a, v) => { a.windVector = [-v, 0]; } },
            { key: 'hotspotTemp', label: 'Hotspot Temp', min: 28, max: 45, step: 1, val: 38.5, unit: '°C', apply: (a, v) => { if (a.paradigmState) a.paradigmState.hotspotTemp = v; } },
            { key: 'cpgBaseFreq', label: 'CPG Cadence', min: 3, max: 14, step: 0.5, val: 8.5, unit: 'Hz', apply: (a, v) => { a.cpg.baseFreq = v; } }
        ]
    }
};


// =============================================================================
// 8. MODERN SCIENTIFIC HUD & DIRECTOR'S DASHBOARD
// =============================================================================

class ScientificHUD {
    constructor(arena) {
        this.arena = arena;
        this.simSpeed = 1.0;
        this.speedOptions = [1.0, 2.0, 5.0, 10.0, 20.0, 50.0, 100.0, 0.5];
        this.speedIdx = 0;

        this.kcCanvas = document.getElementById('kcCanvas');
        this.kcCtx = this.kcCanvas ? this.kcCanvas.getContext('2d') : null;

        this.compassCanvas = document.getElementById('compassCanvas');
        this.compassCtx = this.compassCanvas ? this.compassCanvas.getContext('2d') : null;

        this.scopeCanvas = document.getElementById('oscilloscopeCanvas');
        this.scopeCtx = this.scopeCanvas ? this.scopeCanvas.getContext('2d') : null;
        this.scopeHistory = [];

        this.curveCanvas = document.getElementById('curveCanvas');
        this.curveCtx = this.curveCanvas ? this.curveCanvas.getContext('2d') : null;
        this.learningTrials = [];
        this.resizeCanvases();
        window.addEventListener('resize', () => this.resizeCanvases());

        this.setupCatalogEvents();
        this.setupDeckTabs();
        this.setupNeuroStimControls();
        this.setupLesionEvents();
        this.setupControlBarEvents();
        this.setupToolEvents();

        // Initialize with default paradigm guide
        this.updateActiveCard('open-arena');
        this.renderExperimentGuide('open-arena');
    }

    setSpeed(speed) {
        const num = Math.max(0.1, Math.min(100.0, Number(speed) || 1.0));
        this.simSpeed = num;
        const matchedIdx = this.speedOptions.indexOf(num);
        this.speedIdx = matchedIdx !== -1 ? matchedIdx : 0;

        const btnSpeed = document.getElementById('btnSpeedToggle');
        if (btnSpeed) btnSpeed.textContent = `Speed: ${this.simSpeed}x`;

        const statSpeed = document.getElementById('statSpeed');
        if (statSpeed) statSpeed.textContent = `${this.simSpeed}x`;

        const selSpeed = document.getElementById('selectSpeed');
        if (selSpeed) selSpeed.value = String(this.simSpeed);
    }

    setupCatalogEvents() {
        const cards = document.querySelectorAll('.experiment-card');
        cards.forEach(card => {
            card.addEventListener('click', () => {
                const pid = card.dataset.paradigm;
                if (!pid) return;
                this.arena.initParadigm(pid);
                this.updateActiveCard(pid);
                this.renderExperimentGuide(pid);
                if (pid === 'multisensory-sandbox') {
                    const tabLimbDeck = document.getElementById('tabLimbDeck');
                    if (tabLimbDeck) tabLimbDeck.click();
                } else {
                    const tabGuide = document.getElementById('tabGuide');
                    if (tabGuide) tabGuide.click();
                }
                const badge = document.getElementById('navbarParadigmBadge');
                if (badge) badge.textContent = pid.toUpperCase().replace('-', ' ');
            });
        });
    }

        setupDeckTabs() {
        const tabGuide = document.getElementById('tabGuide');
        const tabLimbDeck = document.getElementById('tabLimbDeck');
        const guideContent = document.getElementById('guideTabContent');
        const limbPanel = document.getElementById('limbDeckPanel');

        if (!tabGuide || !tabLimbDeck || !guideContent || !limbPanel) return;

        tabGuide.addEventListener('click', () => {
            tabGuide.classList.add('active');
            tabLimbDeck.classList.remove('active');
            guideContent.style.display = 'block';
            limbPanel.style.display = 'none';
        });

        tabLimbDeck.addEventListener('click', () => {
            tabLimbDeck.classList.add('active');
            tabGuide.classList.remove('active');
            guideContent.style.display = 'none';
            limbPanel.style.display = 'flex';
        });
    }

    setupNeuroStimControls() {
        const btnToggle = document.getElementById('btnToggleManualControl');
        const lblMode = document.getElementById('labelControlMode');

        if (btnToggle) {
            btnToggle.addEventListener('click', () => {
                const p = this.arena.paradigmState;
                if (!p) return;
                p.manualActive = !p.manualActive;
                if (p.manualActive) {
                    btnToggle.textContent = 'Disable Manual Stim';
                    btnToggle.classList.remove('primary');
                    if (lblMode) {
                        lblMode.textContent = 'DIRECT NEURO-STIMULATION';
                        lblMode.style.color = '#fbbf24';
                    }
                } else {
                    btnToggle.textContent = 'Enable Manual Stim';
                    btnToggle.classList.add('primary');
                    if (lblMode) {
                        lblMode.textContent = 'AUTONOMOUS BRAIN';
                        lblMode.style.color = '#4ade80';
                    }
                }
            });
        }

        // Sliders
        const sDna02 = document.getElementById('sliderStimDna02');
        const sThrust = document.getElementById('sliderStimThrust');
        const sMdn = document.getElementById('sliderStimMdn');
        const sCpg = document.getElementById('sliderStimCpg');
        const sWing = document.getElementById('sliderStimWing');

        if (sDna02) {
            sDna02.addEventListener('input', (e) => {
                const v = parseFloat(e.target.value);
                document.getElementById('valStimDna02').textContent = v.toFixed(2);
                if (this.arena.paradigmState) this.arena.paradigmState.overrideDna02 = v;
            });
        }
        if (sThrust) {
            sThrust.addEventListener('input', (e) => {
                const v = parseFloat(e.target.value);
                document.getElementById('valStimThrust').textContent = v.toFixed(0);
                if (this.arena.paradigmState) this.arena.paradigmState.overrideThrust = v / 100.0;
            });
        }
        if (sMdn) {
            sMdn.addEventListener('input', (e) => {
                const v = parseFloat(e.target.value);
                document.getElementById('valStimMdn').textContent = v.toFixed(0);
                if (this.arena.paradigmState) this.arena.paradigmState.overrideMdn = v / 100.0;
            });
        }
        if (sCpg) {
            sCpg.addEventListener('input', (e) => {
                const v = parseFloat(e.target.value);
                document.getElementById('valStimCpg').textContent = v.toFixed(1);
                this.arena.cpg.baseFreq = v;
            });
        }
        if (sWing) {
            sWing.addEventListener('input', (e) => {
                const v = parseFloat(e.target.value);
                document.getElementById('valStimWing').textContent = v.toFixed(0);
                if (this.arena.paradigmState) this.arena.paradigmState.overrideWings = v;
            });
        }

        // Sensory Flash triggers
        const bGf = document.getElementById('btnFlareGf');
        const bHeat = document.getElementById('btnFlareHeat');
        const bOdor = document.getElementById('btnFlareOdor');
        const bWind = document.getElementById('btnFlareWind');

        if (bGf) {
            bGf.addEventListener('click', () => {
                this.arena.dn.dnp01Gf += 1;
                this.arena.dn.escapeActive = true;
                this.arena.dn.escapeTimer = 0.40;
            });
        }
        if (bHeat) {
            bHeat.addEventListener('click', () => {
                if (this.arena.paradigmState) {
                    this.arena.paradigmState.temp = 40.0;
                    this.arena.mb.stepPlasticity(0.0, 1.0, 1.0, 0.2);
                }
            });
        }
        if (bOdor) {
            bOdor.addEventListener('click', () => {
                this.arena.mb.stepPlasticity(1.0, 0.0, 1.0, 0.2);
                this.arena.dn.dnp09 = 65.0;
            });
        }
        if (bWind) {
            bWind.addEventListener('click', () => {
                this.arena.windVector = [-35.0, 0.0];
                setTimeout(() => { this.arena.windVector = [-15.0, 0.0]; }, 2000);
            });
        }
    }

    updateActiveCard(activePid) {
        document.querySelectorAll('.experiment-card').forEach(card => {
            if (card.dataset.paradigm === activePid) {
                card.classList.add('active');
            } else {
                card.classList.remove('active');
            }
        });
    }

    renderExperimentGuide(pid) {
        const guide = EXPERIMENT_GUIDES[pid] || EXPERIMENT_GUIDES['open-arena'];
        const titleEl = document.getElementById('guideTitle');
        const refEl = document.getElementById('guideRef');
        const watchEl = document.getElementById('guideWhatToWatch');
        const container = document.getElementById('dynamicSlidersContainer');

        if (titleEl) titleEl.textContent = guide.title;
        if (refEl) refEl.textContent = guide.ref;

        if (watchEl) {
            watchEl.innerHTML = '<ul>' + guide.whatToWatch.map(pt => `<li>${pt}</li>`).join('') + '</ul>';
        }

        if (container) {
            container.innerHTML = '';
            if (guide.params && guide.params.length > 0) {
                guide.params.forEach(p => {
                    const row = document.createElement('div');
                    row.className = 'param-slider-row';
                    row.innerHTML = `
                        <div class="slider-header">
                            <span>${p.label}</span>
                            <span><b id="val_${p.key}">${p.val}</b>${p.unit}</span>
                        </div>
                        <input type="range" id="slider_${p.key}" min="${p.min}" max="${p.max}" step="${p.step}" value="${p.val}">
                    `;
                    container.appendChild(row);

                    const slider = row.querySelector(`#slider_${p.key}`);
                    slider.addEventListener('input', (e) => {
                        const val = parseFloat(e.target.value);
                        document.getElementById(`val_${p.key}`).textContent = val;
                        p.apply(this.arena, val);
                    });
                });
            } else {
                container.innerHTML = '<div style="font-size:9.5px; color:#94a3b8; font-style:italic;">Standard autonomous parameters active.</div>';
            }
        }
    }

    setupLesionEvents() {
        const lesions = [
            { id: 'btnLesionWT', type: 'WT', label: 'WT CONTROL' },
            { id: 'btnLesionMB', type: 'DELTA_MB', label: 'ΔMB (KENYON)' },
            { id: 'btnLesionCX', type: 'DELTA_CX', label: 'ΔCX (COMPASS)' },
            { id: 'btnLesionGF', type: 'DELTA_GF', label: 'ΔGF (ESCAPE)' },
            { id: 'btnLesionJO', type: 'DELTA_JO', label: 'ΔJO (WIND)' },
            { id: 'btnLesionOFF', type: 'DELTA_OFF', label: 'ΔOFF (T5)' }
        ];

        lesions.forEach(item => {
            const btn = document.getElementById(item.id);
            if (btn) {
                btn.addEventListener('click', () => {
                    this.arena.setLesion(item.type);
                    document.querySelectorAll('.lesion-btn').forEach(b => {
                        b.classList.remove('active', 'active-wt');
                    });
                    btn.classList.add(item.type === 'WT' ? 'active-wt' : 'active');
                    const label = document.getElementById('activeLesionLabel');
                    if (label) {
                        label.textContent = item.label;
                        label.style.color = item.type === 'WT' ? '#4ade80' : '#fda4af';
                    }
                });
            }
        });
    }

    setupControlBarEvents() {
        const btnReset = document.getElementById('btnResetTrial');
        if (btnReset) {
            btnReset.addEventListener('click', () => this.arena.resetTrial());
        }

        const btnSpeed = document.getElementById('btnSpeedToggle');
        if (btnSpeed) {
            btnSpeed.addEventListener('click', () => {
                this.speedIdx = (this.speedIdx + 1) % this.speedOptions.length;
                this.setSpeed(this.speedOptions[this.speedIdx]);
            });
        }

        const selSpeed = document.getElementById('selectSpeed');
        if (selSpeed) {
            selSpeed.addEventListener('change', (e) => {
                this.setSpeed(parseFloat(e.target.value));
            });
        }

        const btnDlCsv = document.getElementById('btnDownloadCsv');
        if (btnDlCsv) {
            btnDlCsv.addEventListener('click', () => this.downloadCsv());
        }

        const btnDlJson = document.getElementById('btnDownloadJson');
        if (btnDlJson) {
            btnDlJson.addEventListener('click', () => this.downloadJson());
        }

        const btnClear = document.getElementById('btnClearTelemetry');
        if (btnClear) {
            btnClear.addEventListener('click', () => {
                this.arena.telemetryBuffer = [];
                const stepCountEl = document.getElementById('telemetryStepCount');
                if (stepCountEl) stepCountEl.textContent = '0 steps';
            });
        }
    }

    setupToolEvents() {
        const tools = ['toolSelect', 'toolFood', 'toolAlarm', 'toolPredator', 'toolWind'];
        tools.forEach((id) => {
            const btn = document.getElementById(id);
            if (btn) {
                btn.addEventListener('click', () => {
                    tools.forEach((t) => {
                        const b = document.getElementById(t);
                        if (b) b.classList.remove('active');
                    });
                    btn.classList.add('active');
                    this.arena.toolMode = id.replace('tool', '').toLowerCase();
                });
            }
        });
    }

    downloadCsv() {
        if (!this.arena.telemetryBuffer || this.arena.telemetryBuffer.length === 0) {
            alert('Telemetry buffer is currently empty. Run simulation steps first.');
            return;
        }
        const headers = Object.keys(this.arena.telemetryBuffer[0]).join(',');
        const rows = this.arena.telemetryBuffer.map(r => Object.values(r).join(',')).join('\n');
        const blob = new Blob([headers + '\n' + rows], { type: 'text/csv;charset=utf-8;' });
        const url = URL.createObjectURL(blob);
        const link = document.createElement('a');
        link.href = url;
        link.download = `neurofly_${this.arena.activeParadigmId}_telemetry_${Date.now()}.csv`;
        document.body.appendChild(link);
        link.click();
        document.body.removeChild(link);
        URL.revokeObjectURL(url);
    }

    downloadJson() {
        const data = {
            metadata: {
                project: 'Project NeuroFly (v1.0 Alpha)',
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
            telemetrySampleCount: this.arena.telemetryBuffer ? this.arena.telemetryBuffer.length : 0,
            telemetry: (this.arena.telemetryBuffer || []).slice(-500)
        };
        const blob = new Blob([JSON.stringify(data, null, 2)], { type: 'application/json' });
        const url = URL.createObjectURL(blob);
        const a = document.createElement('a');
        a.href = url;
        a.download = `neurofly_${this.arena.activeParadigmId}_trial_${Date.now()}.json`;
        document.body.appendChild(a);
        a.click();
        document.body.removeChild(a);
        URL.revokeObjectURL(url);
    }

        resizeCanvases() {
        const dpr = window.devicePixelRatio || 1;
        if (this.curveCanvas) {
            const rect = this.curveCanvas.getBoundingClientRect();
            if (rect.width > 0 && rect.height > 0) {
                this.curveCanvas.width = rect.width * dpr;
                this.curveCanvas.height = rect.height * dpr;
                this.curveCtx = this.curveCanvas.getContext('2d');
                this.curveCtx.setTransform(1, 0, 0, 1, 0, 0);
                this.curveCtx.scale(dpr, dpr);
                this.curveWidth = rect.width;
                this.curveHeight = rect.height;
            }
        }
        if (this.scopeCanvas) {
            const rect = this.scopeCanvas.getBoundingClientRect();
            if (rect.width > 0 && rect.height > 0) {
                this.scopeCanvas.width = rect.width * dpr;
                this.scopeCanvas.height = rect.height * dpr;
                this.scopeCtx = this.scopeCanvas.getContext('2d');
                this.scopeCtx.setTransform(1, 0, 0, 1, 0, 0);
                this.scopeCtx.scale(dpr, dpr);
                this.scopeWidth = rect.width;
                this.scopeHeight = rect.height;
            }
        }
    }

    recordTrialData(trialNum, metric) {
        const val = this.arena.getRawTrialMetric();
        this.learningTrials.push({ trial: trialNum, value: val, formatted: metric ? metric.value : String(val) });
        if (this.learningTrials.length > 25) this.learningTrials.shift();
        this.renderLearningCurve();
    }

    resetLearningCurve(pid) {
        this.learningTrials = [];
        this.renderLearningCurve();
    }

    renderLearningCurve() {
        if (!this.curveCanvas || !this.curveCtx) return;
        const w = this.curveWidth || this.curveCanvas.clientWidth || 300;
        const h = this.curveHeight || this.curveCanvas.clientHeight || 105;
        const ctx = this.curveCtx;
        ctx.clearRect(0, 0, w, h);

        const pid = this.arena.activeParadigmId;
        const isLatency = (pid === 'heat-maze' || pid === 'labyrinth' || pid === 'wind-tunnel' || pid === 'multisensory-sandbox');

        // Layout padding
        const padding = { left: 32, right: 14, top: 14, bottom: 18 };
        const plotW = w - padding.left - padding.right;
        const plotH = h - padding.top - padding.bottom;

        // Y bounds
        const yMin = isLatency ? 0 : -1.0;
        const yMax = isLatency ? 25.0 : 1.0;

        // Grid lines
        ctx.strokeStyle = 'rgba(255, 255, 255, 0.08)';
        ctx.lineWidth = 1;
        const zeroY = padding.top + plotH * (1 - (0 - yMin) / (yMax - yMin));

        ctx.setLineDash([3, 3]);
        ctx.beginPath();
        ctx.moveTo(padding.left, zeroY);
        ctx.lineTo(w - padding.right, zeroY);
        ctx.stroke();
        ctx.setLineDash([]);

        // Labels
        ctx.fillStyle = '#64748b';
        ctx.font = '8.5px monospace';
        ctx.textAlign = 'right';
        ctx.fillText(isLatency ? '25s' : '+1.0', padding.left - 4, padding.top + 8);
        ctx.fillText(isLatency ? '0s' : '-1.0', padding.left - 4, padding.top + plotH);
        if (!isLatency) ctx.fillText('0.0', padding.left - 4, zeroY + 3);

        const n = this.learningTrials.length;
        if (n === 0) {
            ctx.fillStyle = '#475569';
            ctx.font = 'italic 9.5px sans-serif';
            ctx.textAlign = 'center';
            ctx.fillText('Awaiting trial completions...', padding.left + plotW / 2, padding.top + plotH / 2);
            return;
        }

        // Draw line connecting trials
        ctx.strokeStyle = isLatency ? '#fbbf24' : '#38bdf8';
        ctx.lineWidth = 2.0;
        ctx.beginPath();

        const pts = [];
        for (let i = 0; i < n; i++) {
            const x = padding.left + (n === 1 ? plotW / 2 : (i / (n - 1)) * plotW);
            const clamped = Math.max(yMin, Math.min(yMax, this.learningTrials[i].value));
            const y = padding.top + plotH * (1 - (clamped - yMin) / (yMax - yMin));
            pts.push({ x, y, item: this.learningTrials[i] });
            if (i === 0) ctx.moveTo(x, y);
            else ctx.lineTo(x, y);
        }
        ctx.stroke();

        // Draw trial dots and labels
        for (const pt of pts) {
            ctx.fillStyle = isLatency ? '#f59e0b' : '#38bdf8';
            ctx.shadowColor = isLatency ? '#f59e0b' : '#38bdf8';
            ctx.shadowBlur = 6;
            ctx.beginPath();
            ctx.arc(pt.x, pt.y, 3.5, 0, Math.PI * 2);
            ctx.fill();
            ctx.shadowBlur = 0;

            ctx.fillStyle = '#94a3b8';
            ctx.font = '8px monospace';
            ctx.textAlign = 'center';
            ctx.fillText(`T${pt.item.trial}`, pt.x, padding.top + plotH + 13);
        }

        // Update currentPiVal in header
        const curPiEl = document.getElementById('currentPiVal');
        if (curPiEl && n > 0) {
            curPiEl.textContent = pts[pts.length - 1].item.formatted;
        }
    }

    update() {
        const simTimeEl = document.getElementById('statSimTime');
        if (simTimeEl) simTimeEl.textContent = this.arena.simTime.toFixed(2) + 's';
        const stepEl = document.getElementById('statStep');
        if (stepEl) stepEl.textContent = this.arena.stepCount;

        const trialEl = document.getElementById('paradigmTrial');
        if (trialEl) trialEl.textContent = '#' + this.arena.currentTrial;
        const guideTrialEl = document.getElementById('guideTrialBadge');
        if (guideTrialEl) guideTrialEl.textContent = 'TRIAL #' + this.arena.currentTrial;
        const elapsedEl = document.getElementById('paradigmElapsed');
        if (elapsedEl) elapsedEl.textContent = this.arena.paradigmElapsedSec.toFixed(2) + 's';

        const metric = this.arena.getCanonicalMetricInfo();
        const mLabelEl = document.getElementById('paradigmMetricLabel');
        const mValEl = document.getElementById('paradigmMetricValue');
        if (mLabelEl) mLabelEl.textContent = metric.label;
        if (mValEl) mValEl.textContent = metric.value;

        // Update live card badge in left sidebar
        const activeCardMetricEl = {
            't-maze': 'cardMetricTMaze',
            'y-maze': 'cardMetricYMaze',
            'heat-maze': 'cardMetricHeatMaze',
            'buridan': 'cardMetricBuridan',
            'visual-operant': 'cardMetricVisOperant',
            'wind-tunnel': 'cardMetricWindTunnel',
            'looming-escape': 'cardMetricLooming',
            'optomotor': 'cardMetricOptomotor',
            'gap-crossing': 'cardMetricGapCrossing',
            'circadian-dam': 'cardMetricCircadian',
            'courtship': 'cardMetricCourtship',
            'labyrinth': 'cardMetricLabyrinth',
            'multisensory-sandbox': 'cardMetricMultisensory'
        }[this.arena.activeParadigmId];

        if (activeCardMetricEl) {
            const el = document.getElementById(activeCardMetricEl);
            if (el) el.textContent = metric.value;
        }

        // Dopamine readouts
        const pamHz = this.arena.mb.pamRate;
        const ppl1Hz = this.arena.mb.ppl1Rate;
        const pamRateEl = document.getElementById('paradigmPamRate');
        const ppl1RateEl = document.getElementById('paradigmPpl1Rate');
        const pamFillEl = document.getElementById('paradigmPamFill');
        const ppl1FillEl = document.getElementById('paradigmPpl1Fill');

        if (pamRateEl) pamRateEl.textContent = pamHz.toFixed(1) + ' Hz';
        if (ppl1RateEl) ppl1RateEl.textContent = ppl1Hz.toFixed(1) + ' Hz';
        if (pamFillEl) pamFillEl.style.width = `${Math.min(100, (pamHz / 40.0) * 100)}%`;
        if (ppl1FillEl) ppl1FillEl.style.width = `${Math.min(100, (ppl1Hz / 40.0) * 100)}%`;

        const dStateEl = document.getElementById('valDopamineState');
        if (dStateEl) {
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
        }

        // MB and KC Matrix
        const activeKcs = Array.from(this.arena.mb.kcFiring).filter(r => r > 0).length;
        const kcActiveEl = document.getElementById('valKcActive');
        if (kcActiveEl) kcActiveEl.textContent = `${activeKcs} / 120 (${((activeKcs / 120) * 100).toFixed(0)}%)`;

        const valence = this.arena.mb.netValence;
        const valNetEl = document.getElementById('valNetValence');
        const needleEl = document.getElementById('valNeedle');
        const curPiEl = document.getElementById('currentPiVal');
        if (valNetEl) valNetEl.textContent = (valence >= 0 ? '+' : '') + valence.toFixed(2);
        if (needleEl) needleEl.style.left = `${((valence + 1.0) / 2.0) * 100}%`;
        if (curPiEl) curPiEl.textContent = (valence >= 0 ? '+' : '') + valence.toFixed(2);
        this.renderKcMatrix();
        this.renderLearningCurve();

        // Compass & E-PG
        const compassHeadingEl = document.getElementById('valCompassHeading');
        const headingBumpEl = document.getElementById('valHeadingBump');
        const upwindAngleEl = document.getElementById('valUpwindAngle');
        const pfl3ErrorEl = document.getElementById('valPfl3Error');
        const antennaDeflectEl = document.getElementById('valAntennaDeflect');

        if (compassHeadingEl) compassHeadingEl.textContent = `${((this.arena.fly.heading * 180) / Math.PI).toFixed(1)}°`;
        if (headingBumpEl) headingBumpEl.textContent = `${((this.arena.cx.headingBump * 180) / Math.PI).toFixed(0)}°`;
        const windGlobal = Math.atan2(-this.arena.windVector[1], -this.arena.windVector[0]);
        if (upwindAngleEl) upwindAngleEl.textContent = `${((windGlobal * 180) / Math.PI).toFixed(0)}°`;
        if (pfl3ErrorEl) pfl3ErrorEl.textContent = (this.arena.cx.pfl3ErrorR - this.arena.cx.pfl3ErrorL).toFixed(2);
        if (antennaDeflectEl) antennaDeflectEl.textContent = `${(Math.hypot(this.arena.windVector[0], this.arena.windVector[1]) * 0.12).toFixed(1)} μN`;
        this.renderCompass();

        // Oscilloscope Channels
        this.scopeHistory.push({
            dna02: this.arena.dn.dna02Diff,
            dnp09: this.arena.dn.dnp09,
            bpn: this.arena.dn.bpn,
            mdn: this.arena.dn.mdn,
            dnp01: this.arena.dn.escapeActive ? 50.0 : 0.0
        });
        if (this.scopeHistory.length > 150) this.scopeHistory.shift();
        this.renderOscilloscope();

        // CPG Tripod Gait
        const cpgFreqEl = document.getElementById('valCpgFreq');
        if (cpgFreqEl) cpgFreqEl.textContent = this.arena.cpg.steppingFreq.toFixed(1) + ' Hz';

        // Update Benchmark Scorecard if in multisensory-sandbox
        const p = this.arena.paradigmState;
        if (p && this.arena.activeParadigmId === 'multisensory-sandbox') {
            const compEl = document.getElementById('deckCompositeScore');
            const coordEl = document.getElementById('deckCoordScore');
            const sensEl = document.getElementById('deckSensoryScore');
            const effEl = document.getElementById('deckEfficacyScore');
            const smoothEl = document.getElementById('deckSmoothScore');
            const cardMetricEl = document.getElementById('cardMetricMultisensory');

            if (compEl) compEl.textContent = `${(p.compositeScore || 0).toFixed(1)} / 100`;
            if (coordEl) coordEl.textContent = `${((p.coordinationScore || 0) * 100).toFixed(1)}%`;
            if (sensEl) sensEl.textContent = `${((p.sensoryIntegrationScore || 0) * 100).toFixed(1)}%`;
            if (effEl) effEl.textContent = `${((p.efficiencyScore || 0) * 100).toFixed(1)}%`;
            if (smoothEl) smoothEl.textContent = `${((p.smoothnessScore || 0) * 100).toFixed(1)}%`;
            if (cardMetricEl) cardMetricEl.textContent = `${(p.compositeScore || 0).toFixed(1)} / 100`;

            const jointBox = document.getElementById('jointAnglesBox');
            if (jointBox && p.jointAngles) {
                const ja = p.jointAngles;
                const ls = this.arena.cpg.legStates;
                jointBox.innerHTML = `
                    <div>L1: CTr ${ja.L1.ctr.toFixed(0)}° FTi ${ja.L1.fti.toFixed(0)}° [${ls.L1 ? 'STANCE' : 'SWING'}]</div>
                    <div>R1: CTr ${ja.R1.ctr.toFixed(0)}° FTi ${ja.R1.fti.toFixed(0)}° [${ls.R1 ? 'STANCE' : 'SWING'}]</div>
                    <div>L2: CTr ${ja.L2.ctr.toFixed(0)}° FTi ${ja.L2.fti.toFixed(0)}° [${ls.L2 ? 'STANCE' : 'SWING'}]</div>
                    <div>R2: CTr ${ja.R2.ctr.toFixed(0)}° FTi ${ja.R2.fti.toFixed(0)}° [${ls.R2 ? 'STANCE' : 'SWING'}]</div>
                    <div>L3: CTr ${ja.L3.ctr.toFixed(0)}° FTi ${ja.L3.fti.toFixed(0)}° [${ls.L3 ? 'STANCE' : 'SWING'}]</div>
                    <div>R3: CTr ${ja.R3.ctr.toFixed(0)}° FTi ${ja.R3.fti.toFixed(0)}° [${ls.R3 ? 'STANCE' : 'SWING'}]</div>
                `;
            }
        }
        for (const [leg, isStance] of Object.entries(this.arena.cpg.legStates)) {
            const el = document.getElementById(`leg${leg}`);
            if (el) {
                el.className = `leg-cell ${isStance ? 'stance' : 'swing'}`;
                el.textContent = `${leg}: ${isStance ? 'STANCE' : 'SWING'}`;
            }
        }

        // Telemetry count
        const telemCountEl = document.getElementById('telemetryStepCount');
        if (telemCountEl) telemCountEl.textContent = `${(this.arena.telemetryBuffer || []).length} steps`;
    }

    renderKcMatrix() {
        if (!this.kcCanvas || !this.kcCtx) return;
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
        if (!this.compassCanvas || !this.compassCtx) return;
        const w = this.compassCanvas.width;
        const h = this.compassCanvas.height;
        const cx = w / 2, cy = h / 2, r = w / 2 - 6;

        this.compassCtx.clearRect(0, 0, w, h);

        // 16 wedges of protocerebral bridge aligned with Cartesian needle
        for (let i = 0; i < 16; i++) {
            const a1 = -Math.PI + (i * 2 * Math.PI) / 16;
            const a2 = a1 + (2 * Math.PI) / 16;
            const act = this.arena.cx.bumpProfile[i];

            this.compassCtx.fillStyle = `rgba(56, 189, 248, ${0.12 + act * 0.80})`;
            this.compassCtx.beginPath();
            this.compassCtx.moveTo(cx, cy);
            this.compassCtx.arc(cx, cy, r, -a2, -a1);
            this.compassCtx.closePath();
            this.compassCtx.fill();
        }

        // Draw goal heading vector if active (PFL3 reference)
        if (this.arena.cx.hasGoalHeading) {
            const gh = this.arena.cx.goalHeading;
            this.compassCtx.strokeStyle = '#fbbf24';
            this.compassCtx.lineWidth = 1.8;
            this.compassCtx.setLineDash([2, 2]);
            this.compassCtx.beginPath();
            this.compassCtx.moveTo(cx, cy);
            this.compassCtx.lineTo(cx + Math.cos(gh) * (r - 2), cy - Math.sin(gh) * (r - 2));
            this.compassCtx.stroke();
            this.compassCtx.setLineDash([]);
        }

        // Heading needle (white, representing current heading bump)
        this.compassCtx.strokeStyle = '#ffffff';
        this.compassCtx.lineWidth = 2.5;
        this.compassCtx.shadowColor = 'rgba(255, 255, 255, 0.6)';
        this.compassCtx.shadowBlur = 4;
        this.compassCtx.beginPath();
        this.compassCtx.moveTo(cx, cy);
        this.compassCtx.lineTo(cx + Math.cos(this.arena.fly.heading) * (r - 4), cy - Math.sin(this.arena.fly.heading) * (r - 4));
        this.compassCtx.stroke();
        this.compassCtx.shadowBlur = 0;
    }

    renderOscilloscope() {
        if (!this.scopeCanvas || !this.scopeCtx) return;
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
}

// =============================================================================
// 9. APPLICATION INITIALIZATION & 60 FPS MAIN LOOP
// =============================================================================

window.addEventListener('load', () => {
    const arena = new ScientificBioArena('arenaCanvas');
    const hud = new ScientificHUD(arena);
    window.arena = arena;
    window.hud = hud;

    let isPaused = false;
    const btnPause = document.getElementById('btnPauseToggle');
    if (btnPause) {
        btnPause.addEventListener('click', () => {
            isPaused = !isPaused;
            btnPause.textContent = isPaused ? 'Resume' : 'Pause';
            btnPause.classList.toggle('primary', isPaused);
        });
    }

    // Expose convenient top-level app handle
    window.app = {
        arena,
        hud,
        selectParadigm: (pid) => {
            arena.initParadigm(pid);
            hud.updateActiveCard(pid);
            hud.renderExperimentGuide(pid);
            const badge = document.getElementById('navbarParadigmBadge');
            if (badge) badge.textContent = pid.toUpperCase().replace('-', ' ');
        },
        setSpeed: (spd) => hud.setSpeed(spd),
        togglePause: () => {
            isPaused = !isPaused;
            if (btnPause) {
                btnPause.textContent = isPaused ? 'Resume' : 'Pause';
                btnPause.classList.toggle('primary', isPaused);
            }
        },
        resetTrial: () => arena.resetTrial()
    };

    let lastTime = performance.now();
    let frameCount = 0;
    let fpsTime = lastTime;
    let accumulator = 0;

    function loop(currentTime) {
        frameCount++;
        if (currentTime - fpsTime >= 1000) {
            const fps = Math.round((frameCount * 1000) / (currentTime - fpsTime));
            const statFps = document.getElementById('statFps');
            if (statFps) statFps.textContent = fps;
            frameCount = 0;
            fpsTime = currentTime;
        }

        const rawDt = Math.min(0.1, (currentTime - lastTime) / 1000.0);
        lastTime = currentTime;

        if (!isPaused) {
            const speed = (hud && hud.simSpeed) ? hud.simSpeed : 1.0;
            accumulator += rawDt * speed;

            // Physical timestep: 0.02s (50 Hz) for <=20x, 0.025s for >=50x to maximize throughput while preserving sliding physics
            const stepDt = speed >= 50.0 ? 0.025 : 0.02;
            const maxStepsPerFrame = Math.max(160, Math.ceil(speed * 1.8));
            let stepsExecuted = 0;

            while (accumulator >= stepDt && stepsExecuted < maxStepsPerFrame) {
                arena.step(stepDt);
                accumulator -= stepDt;
                stepsExecuted++;
            }
            if (stepsExecuted >= maxStepsPerFrame) {
                accumulator = 0; // Prevent lag buildup on tab switch or pause
            }
            hud.update();
        }
        arena.render();
        requestAnimationFrame(loop);
    }
    requestAnimationFrame(loop);
});
