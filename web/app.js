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
                this.u[i][1] = 0.0;
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

    sweptCircleToi(p0, p1, radius) {
        const vx = p1[0] - p0[0];
        const vy = p1[1] - p0[1];
        const lSq = vx * vx + vy * vy;

        // Walls are two-sided: a circle that already overlaps the segment is a hit at
        // toi=0 with the normal pointing from the segment towards the circle centre.
        // (Being on the "back" side of a wall at a distance is NOT a hit; internal maze
        // walls, pillars and non-convex arenas all have flies legitimately behind walls.)
        const [cx0, cy0] = this.projectPoint(p0[0], p0[1]);
        const d0 = Math.hypot(p0[0] - cx0, p0[1] - cy0);
        if (d0 <= radius) {
            let nx, ny;
            if (d0 > 1e-8) {
                nx = (p0[0] - cx0) / d0;
                ny = (p0[1] - cy0) / d0;
            } else {
                // Centre exactly on the wall line: push back against the motion direction.
                const side = (vx * this.nx + vy * this.ny) > 0.0 ? -1.0 : 1.0;
                nx = side * this.nx;
                ny = side * this.ny;
            }
            return { hit: true, toi: 0.0, cp: [cx0, cy0], normal: [nx, ny] };
        }

        if (lSq < 1e-12) {
            return { hit: false, toi: 1.0, cp: [0.0, 0.0], normal: [0.0, 0.0] };
        }

        const candidates = [];
        const s0 = (p0[0] - this.p1[0]) * this.nx + (p0[1] - this.p1[1]) * this.ny;
        const s1 = (p1[0] - this.p1[0]) * this.nx + (p1[1] - this.p1[1]) * this.ny;
        const denom = s0 - s1;

        if (Math.abs(denom) > 1e-12) {
            if (s0 >= radius && s1 < radius) {
                const sCand = (s0 - radius) / denom;
                if (sCand >= 0.0 && sCand <= 1.0) {
                    const qx = p0[0] + sCand * vx;
                    const qy = p0[1] + sCand * vy;
                    const t = ((qx - this.p1[0]) * this.dx + (qy - this.p1[1]) * this.dy) / this.lengthSq;
                    if (t >= 0.0 && t <= 1.0) {
                        candidates.push({ toi: sCand, cp: [this.p1[0] + t * this.dx, this.p1[1] + t * this.dy], normal: [this.nx, this.ny] });
                    }
                }
            } else if (s0 <= -radius && s1 > -radius) {
                const sCand = (s0 + radius) / denom;
                if (sCand >= 0.0 && sCand <= 1.0) {
                    const qx = p0[0] + sCand * vx;
                    const qy = p0[1] + sCand * vy;
                    const t = ((qx - this.p1[0]) * this.dx + (qy - this.p1[1]) * this.dy) / this.lengthSq;
                    if (t >= 0.0 && t <= 1.0) {
                        candidates.push({ toi: sCand, cp: [this.p1[0] + t * this.dx, this.p1[1] + t * this.dy], normal: [-this.nx, -this.ny] });
                    }
                }
            }
        }

        for (const W of [this.p1, this.p2]) {
            const rx = p0[0] - W[0];
            const ry = p0[1] - W[1];
            const A = lSq;
            const B = 2.0 * (rx * vx + ry * vy);
            const C = rx * rx + ry * ry - radius * radius;
            const disc = B * B - 4 * A * C;
            if (disc >= 0 && A > 1e-12) {
                const sCand = (-B - Math.sqrt(disc)) / (2.0 * A);
                if (sCand >= 0.0 && sCand <= 1.0) {
                    const qx = p0[0] + sCand * vx;
                    const qy = p0[1] + sCand * vy;
                    const dist = Math.hypot(qx - W[0], qy - W[1]);
                    const nx = dist > 1e-8 ? (qx - W[0]) / dist : this.nx;
                    const ny = dist > 1e-8 ? (qy - W[1]) / dist : this.ny;
                    candidates.push({ toi: sCand, cp: [W[0], W[1]], normal: [nx, ny] });
                }
            }
        }

        if (candidates.length > 0) {
            candidates.sort((a, b) => a.toi - b.toi);
            return { hit: true, toi: candidates[0].toi, cp: candidates[0].cp, normal: candidates[0].normal };
        }

        return { hit: false, toi: 1.0, cp: [0.0, 0.0], normal: [0.0, 0.0] };
    }

    resolveCircleCollisionWithPrev(prevX, prevY, x, y, vx, vy, radius) {
        if (this.lengthSq < 1e-12) {
            return { x, y, vx, vy, collided: false, normal: [0.0, 0.0], t: 0.0 };
        }

        const p0 = [prevX !== undefined ? prevX : x, prevY !== undefined ? prevY : y];
        const p1 = [x, y];
        const swept = this.sweptCircleToi(p0, p1, radius);

        const [cx, cy, t] = this.projectPoint(x, y);
        const dx = x - cx;
        const dy = y - cy;
        const dist = Math.hypot(dx, dy);
        const signedDist = dx * this.nx + dy * this.ny;

        if (!swept.hit && dist >= radius && signedDist >= 0.0) {
            return { x, y, vx, vy, collided: false, normal: [0.0, 0.0], t };
        }

        let cnx, cny;
        if (signedDist < 0.0) {
            cnx = this.nx;
            cny = this.ny;
        } else if (dist > 1e-8) {
            const dot = (dx / dist) * this.nx + (dy / dist) * this.ny;
            if (dot >= 0.0) {
                cnx = dx / dist;
                cny = dy / dist;
            } else {
                cnx = this.nx;
                cny = this.ny;
            }
        } else {
            cnx = this.nx;
            cny = this.ny;
        }

        const resolvedX = cx + cnx * (radius + 0.001);
        const resolvedY = cy + cny * (radius + 0.001);

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

    resolveCircleCollision(x, y, vx, vy, radius) {
        const prevX = x - vx * 0.02;
        const prevY = y - vy * 0.02;
        return this.resolveCircleCollisionWithPrev(prevX, prevY, x, y, vx, vy, radius);
    }
}

// -----------------------------------------------------------------------------
// Containment regions (mirror of arena.py RectRegion / CircleRegion / UnionRegion /
// HoledRegion). signedGap(x, y) is the distance from a point to the region boundary
// (positive inside); a body of radius r is inside when signedGap >= r. inwardNormal is
// the unit vector from the nearest boundary into the region. clamp projects a point
// back inside with a margin.
// -----------------------------------------------------------------------------
class RectRegion {
    constructor(xmin, ymin, xmax, ymax) { this.xmin = xmin; this.ymin = ymin; this.xmax = xmax; this.ymax = ymax; }
    signedGap(x, y) { return Math.min(x - this.xmin, this.xmax - x, y - this.ymin, this.ymax - y); }
    inwardNormal(x, y) {
        const sides = [[x - this.xmin, [1, 0]], [this.xmax - x, [-1, 0]], [y - this.ymin, [0, 1]], [this.ymax - y, [0, -1]]];
        let best = sides[0];
        for (const s of sides) if (s[0] < best[0]) best = s;
        return best[1];
    }
    clamp(x, y, margin) {
        return [Math.max(this.xmin + margin, Math.min(this.xmax - margin, x)), Math.max(this.ymin + margin, Math.min(this.ymax - margin, y))];
    }
    bbox() { return [this.xmin, this.ymin, this.xmax, this.ymax]; }
}

class CircleRegion {
    constructor(cx, cy, radius) { this.cx = cx; this.cy = cy; this.radius = radius; }
    signedGap(x, y) { return this.radius - Math.hypot(x - this.cx, y - this.cy); }
    inwardNormal(x, y) {
        const dx = this.cx - x, dy = this.cy - y, d = Math.hypot(dx, dy);
        return d > 1e-9 ? [dx / d, dy / d] : [1, 0];
    }
    clamp(x, y, margin) {
        const maxD = Math.max(0.0, this.radius - margin);
        const dx = x - this.cx, dy = y - this.cy, d = Math.hypot(dx, dy);
        if (d <= maxD || d < 1e-9) return [x, y];
        return [this.cx + dx * (maxD / d), this.cy + dy * (maxD / d)];
    }
    bbox() { return [this.cx - this.radius, this.cy - this.radius, this.cx + this.radius, this.cy + this.radius]; }
}

class UnionRegion {
    // Union of overlapping members (T-maze stem + cross-bar): the member with the
    // largest signed gap is the one the point belongs to.
    constructor(members) { this.members = members; }
    best(x, y) {
        let b = this.members[0], bg = -Infinity;
        for (const m of this.members) { const g = m.signedGap(x, y); if (g > bg) { bg = g; b = m; } }
        return b;
    }
    signedGap(x, y) { return Math.max(...this.members.map(m => m.signedGap(x, y))); }
    inwardNormal(x, y) { return this.best(x, y).inwardNormal(x, y); }
    clamp(x, y, margin) { return this.signedGap(x, y) >= margin ? [x, y] : this.best(x, y).clamp(x, y, margin); }
    bbox() {
        const b = this.members.map(m => m.bbox());
        return [Math.min(...b.map(v => v[0])), Math.min(...b.map(v => v[1])), Math.max(...b.map(v => v[2])), Math.max(...b.map(v => v[3]))];
    }
}

class HoledRegion {
    // An outer region minus circular obstacles (multisensory arena pillars).
    constructor(outer, holes) { this.outer = outer; this.holes = holes; }
    terms(x, y) {
        const t = [[this.outer.signedGap(x, y), null]];
        for (const h of this.holes) t.push([Math.hypot(x - h.cx, y - h.cy) - h.radius, h]);
        return t;
    }
    signedGap(x, y) { return Math.min(...this.terms(x, y).map(t => t[0])); }
    inwardNormal(x, y) {
        let best = null;
        for (const t of this.terms(x, y)) if (best === null || t[0] < best[0]) best = t;
        if (best[1] === null) return this.outer.inwardNormal(x, y);
        const dx = x - best[1].cx, dy = y - best[1].cy, d = Math.hypot(dx, dy);
        return d > 1e-9 ? [dx / d, dy / d] : [1, 0];
    }
    clamp(x, y, margin) {
        [x, y] = this.outer.clamp(x, y, margin);
        for (const h of this.holes) {
            let dx = x - h.cx, dy = y - h.cy, d = Math.hypot(dx, dy);
            const keepOut = h.radius + margin;
            if (d < keepOut) {
                if (d < 1e-9) { dx = 1.0; dy = 0.0; d = 1.0; }
                x = h.cx + dx * (keepOut / d);
                y = h.cy + dy * (keepOut / d);
            }
        }
        return [x, y];
    }
    bbox() { return this.outer.bbox(); }
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
        // Illustrative local preview only: the wall-avoidance reflex is an engineered
        // assist, not physics. On by default (legacy preview), toggled in the identity bar.
        this.previewWallAssist = true;
        this.activeParadigmTitle = 'Open Arena Multi-Modal Assay';
        this.activeParadigmRef = 'General Neuroethology Open Arena with multi-sensory foraging';
        this.currentTrial = 1;
        this.paradigmElapsedSec = 0.0;
        this.paradigmStatus = 'FORAGING';

        this.worldBounds = { minX: -150, maxX: 150, minY: -110, maxY: 110 };

        // Body radius matches the daemon's FlyState (arena.py: 1.5 mm) so both engines
        // fit the same enclosures; the fly sprite is drawn at a fixed screen size.
        this.fly = {
            x: 0.0,
            y: 0.0,
            heading: 0.0,
            speed: 0.0,
            yawRate: 0.0,
            energy: 1.0,
            radius: 1.5,
            wallTurnDir: 1.0,
            trail: []
        };
        this.containment = new RectRegion(-138.0, -98.0, 138.0, 98.0);

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

            if (this.remoteDriven || this.awaitingDaemon) {
                if (this.toolMode === 'select') return;
                const packet=this.remotePacket, bridge=window.hud?.daemonBridge;
                if (!packet || packet.paradigm!=='open-arena' || !bridge?.connected) return;
                const off=daemonFrameOffset(packet);
                bridge.sendCommand('place_stimulus',{type:this.toolMode,x:worldPos.x-off[0],y:worldPos.y-off[1]}).then(r=>{
                    if(r?.status!=='ok') alert(r?.message||'Stimulus was not applied.');
                });
                return;
            }
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
        // Never render a new scene using the previous assay's telemetry schema.
        if (this.remotePacket && this.remotePacket.paradigm !== paradigmId) {
            this.remotePacket = null;
            this.remoteDriven = false;
            this.awaitingDaemon = true;
        }
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
                this.worldBounds = { minX: -140, maxX: 140, minY: -100, maxY: 100 };
                this.fly.x = 0; this.fly.y = 0; this.fly.heading = 0; this.fly.speed = 0;
                this.windVector = [-15.0, 0.0];
                this.paradigmStatus = 'FORAGING';
                this.currentWalls = [
                    new WallSegment([-140.0, -100.0], [140.0, -100.0]),
                    new WallSegment([140.0, -100.0], [140.0, 100.0]),
                    new WallSegment([140.0, 100.0], [-140.0, 100.0]),
                    new WallSegment([-140.0, 100.0], [-140.0, -100.0])
                ];
                this.paradigmState = {};
                break;

            case 't-maze':
                this.activeParadigmTitle = 'T-Maze Olfactory Associative Conditioning';
                this.activeParadigmRef = 'Tully & Quinn (1985) Science / Cell Pavlovian Conditioning';
                this.worldBounds = { minX: 0, maxX: 140, minY: 0, maxY: 80 };
                this.fly.x = 70.0; this.fly.y = 15.0; this.fly.heading = Math.PI / 2; this.fly.speed = 10.0;
                this.paradigmStatus = 'ASCENDING STEM';
                this.currentWalls = [
                    new WallSegment([63.0, 10.0], [77.0, 10.0]),  // Stem base: dy=0, dx>0 => ny=+1 (inward)
                    new WallSegment([77.0, 10.0], [77.0, 43.0]),  // Stem right wall: dx=0, dy>0 => nx=-1 (inward)
                    new WallSegment([77.0, 43.0], [130.0, 43.0]), // Right arm bottom: dy=0, dx>0 => ny=+1 (inward)
                    new WallSegment([130.0, 43.0], [130.0, 57.0]), // Right arm cap: dx=0, dy>0 => nx=-1 (inward)
                    new WallSegment([130.0, 57.0], [10.0, 57.0]),  // Top continuous ceiling: dy=0, dx<0 => ny=-1 (inward)
                    new WallSegment([10.0, 57.0], [10.0, 43.0]),  // Left arm cap: dx=0, dy<0 => nx=+1 (inward)
                    new WallSegment([10.0, 43.0], [63.0, 43.0]),  // Left arm bottom: dy=0, dx>0 => ny=+1 (inward)
                    new WallSegment([63.0, 43.0], [63.0, 10.0])   // Stem left wall: dx=0, dy<0 => nx=+1 (inward)
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
                this.worldBounds = { minX: 10, maxX: 110, minY: 10, maxY: 110 };
                this.fly.x = 60.0; this.fly.y = 60.0; this.fly.heading = 0.0; this.fly.speed = 10.0;
                this.paradigmStatus = 'STRIPE FIXATION';
                const buridanWalls = [];
                for (let i = 0; i < 32; i++) {
                    const a1 = (2 * Math.PI * i) / 32;
                    const a2 = (2 * Math.PI * (i + 1)) / 32;
                    buridanWalls.push(new WallSegment(
                        [60.0 + 47.0 * Math.cos(a1), 60.0 + 47.0 * Math.sin(a1)],
                        [60.0 + 47.0 * Math.cos(a2), 60.0 + 47.0 * Math.sin(a2)]
                    ));
                }
                this.currentWalls = buridanWalls;
                this.paradigmState = {
                    center: [60.0, 60.0],
                    platformRadius: 47.0,
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
                    new WallSegment([0.0, 0.0], [200.0, 0.0]),   // bottom: ny = +1 (inward)
                    new WallSegment([200.0, 0.0], [200.0, 60.0]), // right: nx = -1 (inward)
                    new WallSegment([200.0, 60.0], [0.0, 60.0]), // top: ny = -1 (inward)
                    new WallSegment([0.0, 60.0], [0.0, 0.0])    // left: nx = +1 (inward)
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
                this.currentWalls = [
                    new WallSegment([5.0, 7.0], [95.0, 7.0]),   // bottom
                    new WallSegment([95.0, 7.0], [95.0, 13.0]), // right cap
                    new WallSegment([95.0, 13.0], [5.0, 13.0]), // top
                    new WallSegment([5.0, 13.0], [5.0, 7.0])   // left cap
                ];
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
                // Fit all monitor tubes; only tube 1 contains the simulated animal.
                this.worldBounds = { minX: -10, maxX: 75, minY: -10, maxY: 172 };
                this.fly.x = 15.0; this.fly.y = 5.0; this.fly.heading = 0.0; this.fly.speed = 6.0;
                this.paradigmStatus = 'LOCOMOTING [AWAKE]';
                this.currentWalls = [
                    new WallSegment([5.0, 1.0], [60.0, 1.0]),  // bottom: ny = +1 (inward)
                    new WallSegment([60.0, 1.0], [60.0, 9.0]), // right: nx = -1 (inward)
                    new WallSegment([60.0, 9.0], [5.0, 9.0]),  // top: ny = -1 (inward)
                    new WallSegment([5.0, 9.0], [5.0, 1.0])   // left: nx = +1 (inward)
                ];
                this.paradigmState = {
                    activeTube: 0,
                    beamCrossings: 0,
                    consecutiveImmobileMin: 0.0,
                    totalSleepMin: 0.0,
                    sleepBouts: 0,
                    inSleepBout: false,
                    lastX: 15.0,
                    actogramHistory: []
                };
                break;

            case 'courtship':
                this.activeParadigmTitle = 'Courtship Conditioning, Wing Song & cVA Suppression';
                this.activeParadigmRef = 'Siegel & Hall (1979); Keleman et al. (Nature 2007)';
                this.worldBounds = { minX: 0, maxX: 20, minY: 0, maxY: 20 };
                this.fly.x = 7.0; this.fly.y = 9.0; this.fly.heading = 0.2; this.fly.speed = 6.0;
                this.paradigmStatus = 'SEARCHING FOR FEMALE';
                const chamberWalls = [];
                for (let i = 0; i < 24; i++) {
                    const a1 = (2 * Math.PI * i) / 24;
                    const a2 = (2 * Math.PI * (i + 1)) / 24;
                    chamberWalls.push(new WallSegment(
                        [10.0 + 8.5 * Math.cos(a1), 10.0 + 8.5 * Math.sin(a1)],
                        [10.0 + 8.5 * Math.cos(a2), 10.0 + 8.5 * Math.sin(a2)]
                    ));
                }
                this.currentWalls = chamberWalls;
                this.paradigmState = {
                    chamberCenter: [10.0, 10.0],
                    chamberRadius: 8.5,
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
                    // Closes the pocket below the goal chamber: the old 5 mm slit between
                    // (100,80) and (100,85) was narrower than the fly and trapped it.
                    new WallSegment([100.0, 70.0], [100.0, 85.0])
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
        this.containment = this.buildContainment(paradigmId);
        this.fly.wallTurnDir = 1.0;
        this.enforceContainment();
    }

    /**
     * Legal body-centre region of a paradigm (mirror of arena.py `_build_containment`,
     * same numbers, so daemon coordinates land inside the dashboard's picture of the
     * arena). The open arena keeps the browser's own ±138 x ±98 frame.
     */
    buildContainment(pid) {
        switch (pid) {
            case 't-maze': return new UnionRegion([new RectRegion(63.0, 10.0, 77.0, 57.0), new RectRegion(10.0, 43.0, 130.0, 57.0)]);
            case 'y-maze': return new CircleRegion(60.0, 60.0, 48.0);
            case 'heat-maze': return new CircleRegion(60.0, 60.0, 55.0);
            case 'buridan': return new CircleRegion(60.0, 60.0, 50.0);
            case 'courtship': return new CircleRegion(10.0, 10.0, 8.5);
            case 'visual-operant': return new RectRegion(0.0, 0.0, 80.0, 80.0);
            case 'looming-escape': return new RectRegion(0.0, 0.0, 80.0, 80.0);
            case 'optomotor': return new RectRegion(0.0, 0.0, 90.0, 90.0);
            case 'wind-tunnel': return new RectRegion(0.0, 0.0, 200.0, 60.0);
            case 'gap-crossing': return new RectRegion(5.0, 7.5, 95.0, 12.5);
            case 'circadian-dam': return new RectRegion(5.0, 1.0, 60.0, 9.0);
            case 'labyrinth': return new RectRegion(0.0, 0.0, 140.0, 100.0);
            case 'multisensory-sandbox':
                return new HoledRegion(new CircleRegion(0.0, 0.0, 75.0),
                    [[35.0, 35.0], [-35.0, 35.0], [-35.0, -35.0], [35.0, -35.0]].map(pc => new CircleRegion(pc[0], pc[1], 6.0)));
            default: return new RectRegion(-138.0, -98.0, 138.0, 98.0);
        }
    }

    resetTrial(advanceTrial = true, keepMemory = true) {
        if (this.remoteDriven) {
            window.hud?.daemonBridge?.sendCommand('reset_trial', {advance: advanceTrial, keep_memory: keepMemory});
            return;
        }
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
            // Bounded: a public stream runs for days and trials complete every few seconds.
            if (this.trialHistory.length > 500) this.trialHistory = this.trialHistory.slice(-400);
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
                this.fly.x = 15.0; this.fly.y = 5.0; this.fly.heading = 0.0; this.fly.speed = 8.0;
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
        // In live mode the daemon owns time, motion, trial boundaries and data.
        // Running the local assay here used to reset the streamed fly independently.
        if (this.remoteDriven || this.awaitingDaemon) return;
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
                // Continuous free foraging (no forced timeout snap)
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
                if (this.paradigmElapsedSec >= 60.0) {
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
                    const hotFloor = p.hotTemp || 36.5;
                    p.temp = hotFloor - (hotFloor - 24.0) * gauss;
                    punishmentSignal = Math.max(0.0, (p.temp - 25.0) / 11.5);
                    p.cumulativeDose += Math.max(0.0, p.temp - 25.0) * dt;
                    this.paradigmStatus = `HOT FLOOR (${p.temp.toFixed(1)}°C)`;
                    if (this.paradigmElapsedSec >= 60.0) {
                        p.escapeLatencyMs = 60000;
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
                if (this.paradigmElapsedSec >= 120.0) {
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
                if (this.paradigmElapsedSec >= 90.0) {
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
                if (this.paradigmElapsedSec >= 60.0) {
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
                if (this.paradigmElapsedSec >= 90.0) {
                    this.resetTrial(true, true);
                }
                if (this.paradigmElapsedSec >= 60.0) {
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
                    // Walk back toward the start; once there let the wall reflex take over
                    // rather than forcing the heading into the start cap every step.
                    if (this.fly.x > 14.0) this.fly.heading = Math.PI;
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
                if (this.paradigmElapsedSec >= 45.0) {
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

                this.paradigmStatus = p.inSleepBout ? 'SLEEP BOUT (>=5m)' : 'LOCOMOTING [AWAKE]';
                // Turn around before reaching the tube end caps (tube x = 5..60, fly radius r).
                // The thresholds must lie inside the reachable range or the fly presses
                // against the cap forever.
                {
                    const rr = this.fly.radius;
                    if (this.fly.x <= 5.0 + rr + 1.5 && Math.cos(this.fly.heading) < 0) {
                        this.fly.heading = 0.0;
                    } else if (this.fly.x >= 60.0 - rr - 1.5 && Math.cos(this.fly.heading) > 0) {
                        this.fly.heading = Math.PI;
                    }
                }
                if (this.paradigmElapsedSec >= 180.0) {
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
                if (this.paradigmElapsedSec >= 60.0) {
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

                // When the daemon streams this paradigm its kinematics and benchmark metrics
                // are mirrored into paradigmState by DaemonBridgeClient; do not overwrite them.
                if (this.remoteDriven) {
                    this.paradigmStatus = `LIVE DAEMON BENCHMARK: ${(p.compositeScore || 0).toFixed(1)} / 100`;
                    break;
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
                p.totalDistance += Math.abs(this.fly.speed) * dt;
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

        // When a live daemon streams this paradigm, the daemon owns the fly position and
        // heading (see DaemonBridgeClient); the local engine still runs the brain models
        // for the HUD but must not integrate locomotion on top of the streamed pose.
        if (this.activeParadigmId !== 'visual-operant' && this.activeParadigmId !== 'optomotor' && !this.remoteDriven) {
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

            // Descending-level wall reflex (mirror of arena.py wall_avoidance_turn): applied to
            // the yaw command after the brain/DN output and before heading integration.
            // fly.yawRate stays the brain's (filtered) command; the reflex output is not fed
            // back into that state, otherwise it accumulates step after step.
            // Engineered assist, gated by the labelled "Preview wall-reflex assist" toggle.
            const yawCmd = this.previewWallAssist ? this.wallAvoidanceTurn(this.fly.yawRate, dt) : this.fly.yawRate;
            this.fly.yawCommand = yawCmd;
            this.fly.heading = ((this.fly.heading + yawCmd * dt + Math.PI) % (2 * Math.PI) + 2 * Math.PI) % (2 * Math.PI) - Math.PI;
            this.cx.updateBump(this.fly.heading);

            let vx = this.fly.speed * Math.cos(this.fly.heading);
            let vy = this.fly.speed * Math.sin(this.fly.heading);
            let proposedX = this.fly.x + vx * dt;
            let proposedY = this.fly.y + vy * dt;
            const prevX = this.fly.x;
            const prevY = this.fly.y;

            if (this.currentWalls && this.currentWalls.length > 0) {
                const p0 = [prevX, prevY];
                const p1 = [proposedX, proposedY];
                let earliestToi = 1.0;
                let hitAny = false;

                for (const wall of this.currentWalls) {
                    const swept = wall.sweptCircleToi(p0, p1, this.fly.radius);
                    if (swept.hit) {
                        hitAny = true;
                        if (swept.toi < earliestToi) {
                            earliestToi = swept.toi;
                        }
                    }
                }

                if (!hitAny) {
                    for (const wall of this.currentWalls) {
                        if (wall.distanceToPoint(p1[0], p1[1]) < this.fly.radius) {
                            hitAny = true;
                            earliestToi = 0.0;
                            break;
                        }
                    }
                }

                if (hitAny) {
                    const s = Math.max(0.0, Math.min(1.0, earliestToi));
                    const cx = p0[0] + s * (p1[0] - p0[0]);
                    const cy = p0[1] + s * (p1[1] - p0[1]);

                    // Gather every wall actually touching the fly at the time of impact.
                    // Normals always point from the wall towards the fly centre, so walls
                    // are two-sided and a far wall the fly happens to be "behind" is never
                    // treated as a contact (that was the cause of through-wall teleports).
                    const contacts = [];
                    for (const wall of this.currentWalls) {
                        const proj = wall.projectPoint(cx, cy);
                        const d = Math.hypot(cx - proj[0], cy - proj[1]);
                        if (d <= this.fly.radius + 0.05) {
                            let nx, ny;
                            if (d > 1e-8) {
                                nx = (cx - proj[0]) / d;
                                ny = (cy - proj[1]) / d;
                            } else {
                                const sidePrev = (p0[0] - proj[0]) * wall.nx + (p0[1] - proj[1]) * wall.ny;
                                const side = sidePrev >= 0.0 ? 1.0 : -1.0;
                                nx = side * wall.nx;
                                ny = side * wall.ny;
                            }
                            contacts.push({ wall, cp: [proj[0], proj[1]], n: [nx, ny], d });
                        }
                    }

                    if (contacts.length > 0) {
                        let resX, resY, resVx, resVy;

                        if (contacts.length === 1) {
                            const c1 = contacts[0];
                            resX = c1.cp[0] + c1.n[0] * (this.fly.radius + 0.001);
                            resY = c1.cp[1] + c1.n[1] * (this.fly.radius + 0.001);

                            const vd = vx * c1.n[0] + vy * c1.n[1];
                            if (vd < 0.0) {
                                const slid = this.coulombSlide(vx, vy, c1.n, vd, c1.wall.friction);
                                resVx = -c1.wall.restitution * vd * c1.n[0] + slid[0];
                                resVy = -c1.wall.restitution * vd * c1.n[1] + slid[1];
                            } else {
                                resVx = vx;
                                resVy = vy;
                            }
                        } else {
                            // Simultaneous 2x2 corner wedge solver
                            let maxCross = -1.0;
                            let pair = [contacts[0], contacts[1]];
                            for (let i = 0; i < contacts.length; i++) {
                                for (let j = i + 1; j < contacts.length; j++) {
                                    const cr = Math.abs(contacts[i].n[0] * contacts[j].n[1] - contacts[i].n[1] * contacts[j].n[0]);
                                    if (cr > maxCross) {
                                        maxCross = cr;
                                        pair = [contacts[i], contacts[j]];
                                    }
                                }
                            }
                            const c1 = pair[0];
                            const c2 = pair[1];
                            const dotN = c1.n[0] * c2.n[0] + c1.n[1] * c2.n[1];

                            if (dotN > 0.6) {
                                // Nearly parallel normals (adjacent facets of a polygonal rim, or a
                                // wall plus the endpoint of a wall abutting it): this is one surface,
                                // not a wedge. Slide along the mean normal; zeroing the velocity here
                                // pinned the fly to the Buridan rim and labyrinth T-junctions.
                                let mx = c1.n[0] + c2.n[0], my = c1.n[1] + c2.n[1];
                                const mm = Math.hypot(mx, my) || 1.0;
                                mx /= mm; my /= mm;
                                const deeper = c1.d <= c2.d ? c1 : c2;
                                // Correct only penetration at the impact point. Anchoring the
                                // averaged normal at either wall's closest point adds a spurious
                                // tangential jump at endpoints, repeatedly undoing the slide.
                                resX = cx;
                                resY = cy;
                                for (let pass = 0; pass < 4; pass++) {
                                    for (const c of contacts) {
                                        const deficit = this.fly.radius + 0.001
                                            - ((resX - c.cp[0]) * c.n[0] + (resY - c.cp[1]) * c.n[1]);
                                        if (deficit > 0.0) {
                                            resX += c.n[0] * deficit;
                                            resY += c.n[1] * deficit;
                                        }
                                    }
                                }
                                const vdm = vx * mx + vy * my;
                                if (vdm < 0.0) {
                                    const slid = this.coulombSlide(vx, vy, [mx, my], vdm, deeper.wall.friction);
                                    resVx = -deeper.wall.restitution * vdm * mx + slid[0];
                                    resVy = -deeper.wall.restitution * vdm * my + slid[1];
                                } else {
                                    resVx = vx;
                                    resVy = vy;
                                }
                            } else {

                            // Contact constraints are inequalities. Keep separating motion;
                            // solving both walls as equalities snaps a departing fly back into
                            // the corner every tick. Project only actual overlap instead.
                            resX = cx;
                            resY = cy;
                            for (let pass = 0; pass < 4; pass++) {
                                for (const c of contacts) {
                                    const deficit = this.fly.radius + 0.001
                                        - ((resX - c.cp[0]) * c.n[0] + (resY - c.cp[1]) * c.n[1]);
                                    if (deficit > 0.0) {
                                        resX += c.n[0] * deficit;
                                        resY += c.n[1] * deficit;
                                    }
                                }
                            }

                            const vd1 = vx * c1.n[0] + vy * c1.n[1];
                            const vd2 = vx * c2.n[0] + vy * c2.n[1];
                            if (vd1 <= 0.0 && vd2 <= 0.0) {
                                resVx = 0.0;
                                resVy = 0.0;
                            } else if (vd1 < 0.0) {
                                const [vtx, vty] = this.coulombSlide(vx, vy, c1.n, vd1, c1.wall.friction);
                                resVx = (vtx * c2.n[0] + vty * c2.n[1] >= -1e-5) ? vtx : 0.0;
                                resVy = (vtx * c2.n[0] + vty * c2.n[1] >= -1e-5) ? vty : 0.0;
                            } else if (vd2 < 0.0) {
                                const [vtx, vty] = this.coulombSlide(vx, vy, c2.n, vd2, c2.wall.friction);
                                resVx = (vtx * c1.n[0] + vty * c1.n[1] >= -1e-5) ? vtx : 0.0;
                                resVy = (vtx * c1.n[0] + vty * c1.n[1] >= -1e-5) ? vty : 0.0;
                            } else {
                                resVx = vx;
                                resVy = vy;
                            }
                            }
                        }

                        // Residual timestep sliding
                        const remDt = (1.0 - s) * dt;
                        if (remDt > 1e-5 && Math.hypot(resVx, resVy) > 1e-5) {
                            let slideX = resX + resVx * remDt;
                            let slideY = resY + resVy * remDt;
                            for (const c of contacts) {
                                const cp = c.wall.projectPoint(slideX, slideY);
                                const d = Math.hypot(slideX - cp[0], slideY - cp[1]);
                                if (d < this.fly.radius) {
                                    slideX = cp[0] + c.n[0] * (this.fly.radius + 0.001);
                                    slideY = cp[1] + c.n[1] * (this.fly.radius + 0.001);
                                }
                            }
                            resX = slideX;
                            resY = slideY;
                        }

                        proposedX = resX;
                        proposedY = resY;
                        vx = resVx;
                        vy = resVy;

                        for (const c of contacts) {
                            this.collisionNormals.push({
                                x: proposedX,
                                y: proposedY,
                                nx: c.n[0],
                                ny: c.n[1]
                            });
                        }
                        if (this.paradigmState && this.paradigmState.wallCollisions !== undefined) {
                            this.paradigmState.wallCollisions += 1;
                        }

                        // Collision impulses change translation, not body heading. The
                        // tactile reflex above supplies turning. Aligning heading to a tiny
                        // frictional slide can cancel that reflex and pin a fly in a corner.

                    }
                }
            }

            // Universal hard containment (safety net only; the wall physics above should
            // keep the fly inside). Bounce: reflect the velocity AND the heading so the fly
            // walks away from the boundary instead of being pinned against it.
            if (this.worldBounds) {
                const pad = this.fly.radius + 0.1;
                const minX = this.worldBounds.minX + pad;
                const maxX = this.worldBounds.maxX - pad;
                const minY = this.worldBounds.minY + pad;
                const maxY = this.worldBounds.maxY - pad;
                let bounced = false;
                if (proposedX < minX) { proposedX = minX; vx = Math.abs(vx); bounced = true; }
                if (proposedX > maxX) { proposedX = maxX; vx = -Math.abs(vx); bounced = true; }
                if (proposedY < minY) { proposedY = minY; vy = Math.abs(vy); bounced = true; }
                if (proposedY > maxY) { proposedY = maxY; vy = -Math.abs(vy); bounced = true; }
                if (bounced && Math.hypot(vx, vy) > 1e-6) {
                    this.fly.heading = Math.atan2(vy, vx);
                }
            }

            this.fly.x = proposedX;
            this.fly.y = proposedY;
            // Keep the sign of the speed: MDN "moonwalking" is backward motion and must
            // not be folded into forward motion by the magnitude.
            this.fly.speed = (isReversing ? -1.0 : 1.0) * Math.hypot(vx, vy);

            // Hard geometric failsafe (region based, mirrors arena.py enforce_containment)
            this.enforceContainment(dt);
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
                source: 'local_preview',
                segment: `${this.activeParadigmId}:${this.currentTrial}`,
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

    // Wall-avoidance reflex, mirrored from arena.py (WALL_PERCEPTION_MM / WALL_AVOID_YAW_RAD_S):
    // boundaries closer than 4 mm to the body edge are perceived, and the reflex commands
    // a yaw away from them. arena.py clips the brain's yaw command to 0.45 rad/s and uses a
    // 4 rad/s reflex; this engine's brain saturates at ~2.1 rad/s (0.14 * 67.5 / 4.5), so the
    // reflex ceiling is 8 rad/s here to stay dominant over the goal drive the same way.
    // The "Boundary Repulsion" slider scales it (1.0 = default).
    static get WALL_PERCEPTION_MM() { return 4.0; }
    static get WALL_AVOID_YAW_RAD_S() { return 8.0; }

    /**
     * Boundaries within `perception` mm of the body edge as [gap, nx, ny]: gap is the
     * free space between body edge and boundary (negative when penetrating), (nx, ny)
     * the unit normal pointing away from that boundary. Wall segments and the
     * containment region are both sensed (mirror of arena.py sense_boundaries).
     */
    senseBoundaries(x, y, radius, perception) {
        const reach = perception === undefined ? ScientificBioArena.WALL_PERCEPTION_MM : perception;
        const found = [];
        if (this.containment) {
            const g = this.containment.signedGap(x, y) - radius;
            if (g < reach) {
                const n = this.containment.inwardNormal(x, y);
                found.push([g, n[0], n[1]]);
            }
        }
        for (const wall of (this.currentWalls || [])) {
            const [px, py] = wall.projectPoint(x, y);
            const dx = x - px, dy = y - py;
            const d = Math.hypot(dx, dy);
            const g = d - radius;
            if (g < reach) {
                if (d > 1e-8) found.push([g, dx / d, dy / d]);
                else found.push([g, wall.nx, wall.ny]);
            }
        }
        return found;
    }

    /**
     * Descending-level reflex (mirror of arena.py wall_avoidance_turn): each perceived
     * boundary contributes its away-normal weighted by proximity (1 at contact, 0 at the
     * perception range). If the fly is travelling into the net normal, an extra yaw of
     * up to WALL_AVOID_YAW_RAD_S is added in the direction that rotates the travel
     * direction away from the wall; the side is remembered in fly.wallTurnDir so a
     * head-on approach does not dither. Sliding parallel to a wall is untouched.
     * Returns the modified yaw command (rad/s).
     */
    wallAvoidanceTurn(dheading, dt) {
        const fly = this.fly;
        const sensed = this.senseBoundaries(fly.x, fly.y, fly.radius);
        if (sensed.length === 0) return dheading;

        const reach = ScientificBioArena.WALL_PERCEPTION_MM;
        const travel = fly.speed >= 0.0 ? fly.heading : fly.heading + Math.PI;
        const hx = Math.cos(travel), hy = Math.sin(travel);
        let netX = 0.0, netY = 0.0, proximity = 0.0;
        for (const [gap, nx, ny] of sensed) {
            const wgt = 1.0 - Math.max(0.0, gap) / reach;
            // A wall we are departing must not cancel the wall we are approaching
            // at a junction. Weight each normal by its own closing direction.
            const closing = Math.max(0.0, -(hx * nx + hy * ny));
            netX += wgt * closing * nx;
            netY += wgt * closing * ny;
            proximity = Math.max(proximity, wgt);
        }
        const nMag = Math.hypot(netX, netY);
        if (nMag < 1e-9 || proximity <= 0.0) return dheading;
        netX /= nMag;
        netY /= nMag;

        // Direction of travel, not the heading: a fly walking backwards (MDN reverse)
        // can back into a wall while facing away from it.
        const approach = -(hx * netX + hy * netY);   // > 0 when moving into the boundary
        if (approach <= 0.0) return dheading;

        const cross = hx * netY - hy * netX;          // > 0: CCW turn moves travel toward the normal
        if (Math.abs(cross) > 0.1) fly.wallTurnDir = cross > 0.0 ? 1.0 : -1.0;
        const gain = (this.wallRepulsion !== undefined) ? this.wallRepulsion : 1.0;
        const avoid = fly.wallTurnDir * ScientificBioArena.WALL_AVOID_YAW_RAD_S * gain * proximity * approach;
        const limit = Math.min(ScientificBioArena.WALL_AVOID_YAW_RAD_S * Math.max(1.0, gain), (Math.PI / 2.0) / Math.max(dt, 1e-6));
        return Math.max(-limit, Math.min(limit, dheading + avoid));
    }

    /**
     * Coulomb sliding: the tangential velocity that survives a contact. The friction
     * impulse is bounded by mu times the normal impulse (|vd| removed), so a glancing
     * contact barely slows the fly while a head-on one stops it. (A fixed percentage cut
     * per 20 ms step made walls "sticky": the fly crawled along them at < 1 mm/s.)
     */
    coulombSlide(vx, vy, n, vd, mu) {
        const vtx = vx - vd * n[0];
        const vty = vy - vd * n[1];
        const vtMag = Math.hypot(vtx, vty);
        if (vtMag < 1e-9) return [0.0, 0.0];
        const cut = Math.min(vtMag, Math.max(0.0, mu) * (-vd));
        const k = (vtMag - cut) / vtMag;
        return [vtx * k, vty * k];
    }

    /** Drops a food pellet a little ahead of the fly (Open Arena / Labyrinth action button). */
    spawnFoodNearFly() {
        const ahead = 18.0;
        this.foodItems.push({
            x: this.fly.x + ahead * Math.cos(this.fly.heading),
            y: this.fly.y + ahead * Math.sin(this.fly.heading),
            radius: 12.0, odorStrength: 1.0
        });
        if (this.foodItems.length > 12) this.foodItems.shift();
    }

    /** Launches a looming predator toward the fly (Open Arena action button). */
    triggerLooming() {
        const dist = 70.0;
        const ang = this.fly.heading + Math.PI * 0.75;
        const px = this.fly.x + dist * Math.cos(ang);
        const py = this.fly.y + dist * Math.sin(ang);
        const toFly = Math.atan2(this.fly.y - py, this.fly.x - px);
        this.predators.push({ x: px, y: py, vx: Math.cos(toFly) * 28.0, vy: Math.sin(toFly) * 28.0, radius: 8.0, active: true });
        if (this.predators.length > 6) this.predators.shift();
    }

    /**
     * Hard geometric failsafe (mirror of arena.py enforce_containment): keep the body
     * inside the paradigm's legal region. Runs after collision resolution and the wall
     * reflex, so it only fires when those have already failed. It behaves like a wall
     * contact rather than a silent clamp: the position is projected back inside, the
     * speed component driving into the boundary is removed, and the heading receives
     * the same away-from-wall torque as a real collision, so the fly is never left
     * pushing into an invisible boundary step after step. Returns true when corrected.
     */
    enforceContainment(dt = 0.02) {
        const fly = this.fly;
        if (!this.containment) return false;
        const r = fly.radius;
        const x = fly.x, y = fly.y;
        if (this.containment.signedGap(x, y) >= r) return false;

        const [nx, ny] = this.containment.inwardNormal(x, y);
        [fly.x, fly.y] = this.containment.clamp(x, y, r);

        const travel = fly.speed >= 0.0 ? fly.heading : fly.heading + Math.PI;
        const hx = Math.cos(travel), hy = Math.sin(travel);
        const into = -(hx * nx + hy * ny);
        if (into > 0.0) {
            // Keep only the tangential share of the commanded speed (no bounce), sign kept.
            fly.speed = fly.speed * Math.sqrt(Math.max(0.0, 1.0 - into * into));
            const cross = hx * ny - hy * nx;
            if (Math.abs(cross) > 0.1) fly.wallTurnDir = cross > 0.0 ? 1.0 : -1.0;
            const turn = Math.min(ScientificBioArena.WALL_AVOID_YAW_RAD_S * dt, Math.PI / 2.0);
            fly.heading = ((fly.heading + fly.wallTurnDir * turn + Math.PI) % (2 * Math.PI) + 2 * Math.PI) % (2 * Math.PI) - Math.PI;
        }
        return true;
    }

    getCanonicalMetricInfo() {
        if (this.remoteDriven && this.remotePacket) {
            const keys = {'t-maze':'performance_index','y-maze':'spontaneous_alternation_rate',
                'heat-maze':'escape_latency_ms','buridan':'centrophobism_index','visual-operant':'operant_learning_index',
                'wind-tunnel':'upwind_progress_mm','looming-escape':'time_to_collision_at_jump_ms','optomotor':'optomotor_gain',
                'gap-crossing':'crossing_success','circadian-dam':'total_sleep_minutes','courtship':'courtship_index',
                'labyrinth':'path_tortuosity','multisensory-sandbox':'composite_benchmark_score'};
            const key=keys[this.activeParadigmId],value=key ? this.remotePacket.metrics?.[key] : this.remotePacket.neural?.net_valence;
            const units={'escape_latency_ms':' ms','time_to_collision_at_jump_ms':' ms','upwind_progress_mm':' mm','total_sleep_minutes':' min','composite_benchmark_score':' /100'};
            const rawValue=typeof value==='boolean'?Number(value):Number.isFinite(value)?value:null;
            if(this.activeParadigmId==='gap-crossing') return {label:'Gap width / outcome',rawValue,unit:'crossed 0/1',value:`${this.remotePacket.metrics?.gap_width_mm ?? '—'} mm · ${value?'CROSSED':this.remotePacket.metrics?.decision_outcome||'APPROACH'}`,sub:'Daemon measurement'};
            return {label:key ? key.replace(/_/g,' ') : 'Odor value',rawValue,unit:units[key]||'',value:rawValue!==null?rawValue.toFixed(2)+(units[key]||''):'Not observed',sub:'Daemon measurement'};
        }
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
        if (this.remoteDriven && this.remotePacket?.world_bounds) {
            const b=this.remotePacket.world_bounds,o=daemonFrameOffset(this.remotePacket);
            const a=this.worldToScreen(b[0]+o[0],b[3]+o[1]),z=this.worldToScreen(b[2]+o[0],b[1]+o[1]);
            ctx.strokeStyle='#38bdf8';ctx.lineWidth=1.2;ctx.strokeRect(a.x,a.y,z.x-a.x,z.y-a.y);
        }
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

        const reversed = this.remotePacket?.scene?.cs_plus_arm === 'arm_b';
        const sa = this.worldToScreen(reversed ? 125.0 : 15.0, 50.0);
        const gradA = ctx.createRadialGradient(sa.x, sa.y, 2, sa.x, sa.y, 70);
        gradA.addColorStop(0, 'rgba(34, 197, 94, 0.5)');
        gradA.addColorStop(1, 'rgba(34, 197, 94, 0.0)');
        ctx.fillStyle = gradA;
        ctx.beginPath(); ctx.arc(sa.x, sa.y, 70, 0, 2 * Math.PI); ctx.fill();

        ctx.fillStyle = '#22c55e';
        ctx.beginPath(); ctx.arc(sa.x, sa.y, 8, 0, 2 * Math.PI); ctx.fill();
        ctx.fillStyle = '#86efac';
        ctx.beginPath(); ctx.arc(sa.x - 2, sa.y - 2, 3, 0, 2 * Math.PI); ctx.fill();

        const sb = this.worldToScreen(reversed ? 15.0 : 125.0, 50.0);
        const gradB = ctx.createRadialGradient(sb.x, sb.y, 2, sb.x, sb.y, 70);
        gradB.addColorStop(0, 'rgba(244, 63, 94, 0.5)');
        gradB.addColorStop(1, 'rgba(244, 63, 94, 0.0)');
        ctx.fillStyle = gradB;
        ctx.beginPath(); ctx.arc(sb.x, sb.y, 70, 0, 2 * Math.PI); ctx.fill();

        ctx.strokeStyle = this.paradigmState.shockPulse > 0.1 ? '#fbbf24' : 'rgba(244, 63, 94, 0.35)';
        ctx.lineWidth = 1.0;
        for (let gx = reversed ? 12 : 82; gx < (reversed ? 58 : 128); gx += 4) {
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

        const streamedStripes=this.remotePacket?.paradigm==='buridan' ? this.remotePacket.scene?.landmarks : null;
        const stripes=Array.isArray(streamedStripes) && streamedStripes.length>=2
            && streamedStripes.slice(0,2).every(p=>Array.isArray(p)&&p.length===2&&p.every(Number.isFinite))
            ? streamedStripes : [[110,60],[10,60]];
        const s1 = this.worldToScreen(...stripes[0]);
        const s2 = this.worldToScreen(...stripes[1]);
        ctx.globalAlpha=this.remotePacket?.scene?.stripe_contrast === 0 ? 0 : 1;
        ctx.fillStyle = '#000000';
        ctx.strokeStyle = '#ffffff';
        ctx.lineWidth = 1.5;
        ctx.fillRect(s1.x - 5, s1.y - 18, 10, 36); ctx.strokeRect(s1.x - 5, s1.y - 18, 10, 36);
        ctx.fillRect(s2.x - 5, s2.y - 18, 10, 36); ctx.strokeRect(s2.x - 5, s2.y - 18, 10, 36);
        ctx.globalAlpha=1;

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
            const isPunished = (q % 2 === 1) !== !!this.remotePacket?.scene?.invert_sectors;
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

        // Show the Gaussian field actually sampled by the daemon across the tunnel.
        const sigma=this.remotePacket?.scene?.filament_sigma || p.filamentSigma || 3.5;
        for(let y=0;y<60;y+=2){
            const c=Math.exp(-((y+1-p.nozzlePos[1])**2)/(2*sigma*sigma));
            if(c<.005)continue;
            const left=this.worldToScreen(0,y+2),right=this.worldToScreen(p.nozzlePos[0]+5,y);
            const grad=ctx.createLinearGradient(left.x,0,right.x,0);
            grad.addColorStop(0,`rgba(34,197,94,${.45*c*Math.exp(-p.nozzlePos[0]/250)})`);
            grad.addColorStop(1,`rgba(34,197,94,${.45*c})`);
            ctx.fillStyle=grad;ctx.fillRect(left.x,left.y,right.x-left.x,right.y-left.y);
        }

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
        if (p.contrast===0) {
            ctx.fillStyle='#64748b';ctx.beginPath();ctx.arc(sc.x,sc.y,85,0,Math.PI*2);ctx.fill();
        }
        for (let i = 0; p.contrast!==0 && i < numStripes; i++) {
            const a1 = ((p.drumAngleDeg + i * (360 / numStripes)) * Math.PI) / 180;
            const a2 = a1 + (Math.PI / numStripes);
            ctx.fillStyle = p.contrast === 0 ? '#64748b' : (i % 2 === 0 ? '#020617' : '#e2e8f0');
            ctx.beginPath(); ctx.moveTo(sc.x, sc.y); ctx.arc(sc.x, sc.y, 85, a1, a2); ctx.fill();
        }

        ctx.fillStyle = '#0f172a';
        ctx.beginPath(); ctx.arc(sc.x, sc.y, 65, 0, 2 * Math.PI); ctx.fill();

        ctx.font = '10px monospace';
        ctx.fillStyle = '#38bdf8';
        ctx.fillText(`HS proxy: ${p.hsFiringRate.toFixed(1)} Hz | Shunt model: 85%`, sc.x - 75, sc.y + 95);
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
        const label = this.worldToScreen(0, 165);
        ctx.fillStyle = '#94a3b8';
        ctx.font = '10px monospace';
        ctx.fillText('Tube 1 active · 1 simulated fly', label.x, label.y);
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
            { key: 'predatorSpeed', label: 'Predator Speed', min: 10, max: 60, step: 5, val: 25, unit: 'mm/s', desc: 'Linear velocity of approaching predatory mantids. Faster speeds challenge Giant Fiber optical expansion detection.', apply: (a, v) => { a.predators.forEach(p => { const sp = Math.hypot(p.vx, p.vy) || 1; p.vx = (p.vx / sp) * v; p.vy = (p.vy / sp) * v; }); } },
            { key: 'windVelocity', label: 'Wind Velocity', min: 0, max: 40, step: 5, val: 15, unit: 'mm/s', desc: 'Ambient airflow velocity sensed by Johnston’s organ, driving upwind anemotactic course correction.', apply: (a, v) => { a.windVector = [-v, 0]; } }
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
            { key: 'shockPulse', label: 'Shock Voltage', min: 0, max: 100, step: 10, val: 60, unit: 'V', desc: 'Aversive electric grid voltage in CS- arm. Regulates PPL1 dopaminergic punishment spike rate and rate of learning.', apply: (a, v) => { a.paradigmState.shockPulse = v / 100; } }
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
            { key: 'turnBias', label: 'Premotor Turn Bias', min: -20, max: 20, step: 2, val: 0, unit: 'Hz', desc: 'Injected bilateral current asymmetry between left and right DNa02 descending neurons, inducing turn handedness.', apply: (a, v) => { a.dn.dna02Diff += v; } }
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
            { key: 'floorTemp', label: 'Floor Temperature', min: 28, max: 42, step: 0.5, val: 36.5, unit: '°C', desc: 'Floor bath temperature. Above 36°C, TrpA1 heat receptors drive intense escape toward cool refuge.', apply: (a, v) => { a.paradigmState.temp = v; } },
            { key: 'refugeRadius', label: 'Refuge Radius', min: 6, max: 15, step: 1, val: 9, unit: 'mm', desc: 'Target cool tile radius. Smaller targets require tighter landmark triangulation by the Central Complex.', apply: (a, v) => { a.paradigmState.refugeRadius = v; } }
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
            { key: 'platformRadius', label: 'Platform Radius', min: 35, max: 60, step: 5, val: 50, unit: 'mm', desc: 'Radius of illuminated circular stage. Regulates centrophobism arena area and travel distance.', apply: (a, v) => { a.paradigmState.platformRadius = v; } }
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
            { key: 'couplingGain', label: 'Yaw Coupling Gain', min: 50, max: 200, step: 10, val: 120, unit: '°/s', desc: 'Closed-loop coupling gain between flight yaw torque and drum rotation. Higher values make steering more responsive.', apply: (a, v) => { a.paradigmState.couplingGain = v; } }
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
            { key: 'windVelocity', label: 'Wind Velocity', min: 10, max: 50, step: 5, val: 25, unit: 'mm/s', desc: 'Downwind carrier velocity. Sensed by Johnston’s organ, dictating surge vs cast transitions.', apply: (a, v) => { a.paradigmState.windFlow = [-v, 0]; } }
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
            { key: 'rOverV', label: 'Looming r/v Ratio', min: 10, max: 50, step: 5, val: 25, unit: 'ms', desc: 'Threat size-to-approach speed ratio (r/v). Smaller values model faster looming predator swoops.', apply: (a, v) => { a.paradigmState.rOverVS = v / 1000; } }
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
            { key: 'drumSpeed', label: 'Grating Velocity', min: -60, max: 60, step: 5, val: 30, unit: '°/s', desc: 'Rotational speed of surrounding drum. Drives wide-field optic flow in Lobula Plate Tangential Cells.', apply: (a, v) => { a.paradigmState.drumSpeedDegS = v; } }
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
            { key: 'gapWidth', label: 'Chasm Width', min: 2.0, max: 5.5, step: 0.2, val: 3.5, unit: 'mm', desc: 'Abyss chasm width. Fly will attempt step-over if <= 3.8mm, but execute 180° abort turn if > 4.2mm.', apply: (a, v) => { a.paradigmState.gapWidthMm = v; } }
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
            { key: 'dayNight', label: 'Day / Night Light', min: 0, max: 1, step: 1, val: 1, unit: ' (0=DD, 1=LD)', desc: '12:12 Light:Dark (LD) photoperiod vs Constant Darkness (DD). Lights-on triggers circadian morning peak.', apply: (a, v) => { a.paradigmState.isLightsOn = (v === 1); } }
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
            { key: 'femaleMated', label: 'Female Mated State', min: 0, max: 1, step: 1, val: 0, unit: ' (0=Virgin, 1=Mated)', desc: 'Female state: 0=Virgin (receptive, aphrodisiac), 1=Mated (aversive cVA, rejection kicks training male to suppress song).', apply: (a, v) => { a.paradigmState.isFemaleVirgin = (v === 0); } }
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
            { key: 'friction', label: 'Wall Friction', min: 0.1, max: 0.9, step: 0.1, val: 0.5, unit: 'mu', desc: 'Coulomb crawling contact friction against corridor walls. Higher friction dampens sliding velocity.', apply: (a, v) => { a.currentWalls.forEach(w => w.friction = v); } }
        ]
    },
    'multisensory-sandbox': {
        title: "Multisensory Ingress & Limb Biomechanics Sandbox",
        ref: "Project NeuroFly compact modular sensorimotor model",
        whatToWatch: [
            "Full multi-sensory cue integration: Food Odor A, Repellent Odor B, cVA Pheromone, Thermal Gradient, and Vector Wind.",
            "Inspect 6 articulated tripod legs with real-time Coxa, Femur, and Tibia joint angle flexions.",
            "Toggle between Autonomous Connectome Mode and Direct Neuro-Stimulation / Limb Override Deck.",
            "Evaluate composite benchmark score across Coordination, Sensory Integration, Smoothness, and Efficiency."
        ],
        params: [
            { key: 'windMagnitude', label: 'Wind Velocity', min: 0, max: 40, step: 5, val: 15, unit: 'mm/s', desc: 'Continuous vector wind speed modulating Johnston’s organ antennal load and upwind anemotaxis drive.', apply: (a, v) => { a.windVector = [-v, 0]; } },
            { key: 'hotspotTemp', label: 'Hotspot Temp', min: 28, max: 45, step: 1, val: 38.5, unit: '°C', desc: 'Peak temperature of the localized thermal emitter. Tests thermotactic avoidance vs food attraction.', apply: (a, v) => { if (a.paradigmState) a.paradigmState.hotspotTemp = v; } },
            { key: 'cpgBaseFreq', label: 'CPG Cadence', min: 3, max: 14, step: 0.5, val: 8.5, unit: 'Hz', desc: 'Kuramoto Central Pattern Generator base tripod cadence modulating 6-leg stepping frequency.', apply: (a, v) => { a.cpg.baseFreq = v; } }
        ]
    }
};


// =============================================================================
// 7B. CONTINUOUS LEARNING CLUSTER DAEMON BRIDGE (RYZEN & LOCAL CLUSTER)
// =============================================================================

// The daemon reports fly positions in each paradigm's own arena frame ([0, width] x
// [0, height], see maze.py `dimensions`). Most browser arenas use the same frame; the
// ones that do not are listed here as (offset x, offset y) to add to daemon coordinates.
const DAEMON_FRAME_OFFSET = {
    'multisensory-sandbox': [-80.0, -80.0]   // daemon 160x160 corner-origin -> browser centre-origin
};

function daemonFrameOffset(pkt) {
    const bounds = pkt.world_bounds;
    if (pkt.paradigm === 'multisensory-sandbox' && bounds?.[0] < 0) return [0, 0];
    if (pkt.paradigm === 'open-arena' && bounds?.length === 4) {
        return [-(bounds[0]+bounds[2])/2, -(bounds[1]+bounds[3])/2];
    }
    return DAEMON_FRAME_OFFSET[pkt.paradigm] || [0, 0];
}

function isStaleDaemonPacket(pkt, previous) {
    // A restarted daemon begins a new run at step zero, even in the same assay.
    if (!previous || pkt.run_id !== previous.run_id || pkt.paradigm !== previous.paradigm) return false;
    return (Number.isFinite(pkt.step) && pkt.step < previous.step)
        || (Number.isFinite(pkt.timestamp) && pkt.timestamp < previous.timestamp);
}

/**
 * Why a packet must be rejected after this tab's last acknowledged switch, or null.
 * Mirror of neurofly_daemon.identity_rejection. The ack names the activated run and
 * instance and its activation counter: from the same daemon process, an older
 * activation is stale and the same activation must match run_id and instance_id.
 * A newer activation (another tab switched later) or another daemon process is accepted.
 */
function identityRejection(pkt, ack) {
    const want = ack?.identity, got = pkt?.identity;
    if (!want || !got) return null;
    if (pkt.run_id !== want.daemon_run_id) return null;
    if (!Number.isInteger(got.activation) || !Number.isInteger(want.activation)) return 'packet identity has no activation counter';
    if (got.activation < want.activation) return `stale: activation ${got.activation} precedes acknowledged switch ${want.activation}`;
    if (got.activation === want.activation && (got.run_id !== want.run_id || got.instance_id !== want.instance_id)) {
        return 'run_id/instance_id differ from the acknowledged switch';
    }
    return null;
}
window.neuroflyIdentityRejection = identityRejection;

// motor_source values that mean the named controller drove the step (arena.py
// Arena.MOTOR_SOURCES_NORMAL); anything else is shown in the identity banner.
const MOTOR_SOURCE_NOTES = {
    'halted-rpc-fault': 'graph RPC unavailable: the fly is halted, no motor command',
    'surrogate-fallback-TEST': 'TEST fallback: the hand-built surrogate drives the fly, not the graph',
    'graph-unmapped-io': 'no verified sensory/motor mapping for this assay yet: the graph gets no input and commands no motion',
    'halted-no-instance': 'no active graph instance: the fly is halted',
};

/** Identity bar + conspicuous banner for synthetic/test runs, abnormal motor sources and faults. */
function renderIdentity(pkt) {
    const id = pkt?.identity || {}, motor = pkt?.motor || {};
    const set = (elId, text, title) => { const el = document.getElementById(elId); if (el) { el.textContent = text; if (title !== undefined) el.title = title; } };
    set('identBackend', id.backend || 'unknown');
    set('identLabel', id.label || '', id.label || '');
    const assists = motor.motor_assists || {};
    const on = Object.keys(assists).filter(k => assists[k]);
    set('identAssists', motor.motor_assists_enabled ? `ON (${on.join(', ')})` : 'OFF', 'Engineered motor assists: they turn the fly away from walls on the controller\'s behalf and are not credited to any brain.');
    set('identMotor', motor.motor_source || '--');
    const fault = pkt?.controller_fault ?? motor.controller_fault ?? null;
    set('identFault', fault || 'none');
    set('identRun', `run ${(id.run_id || '--').slice(-12)}`, `run_id ${id.run_id || '?'}\ninstance_id ${id.instance_id || '?'}\ngraph_sha256 ${id.graph_sha256 || 'none (modular)'}\nactivation ${id.activation ?? '?'}`);
    const faultEl = document.getElementById('identFault');
    if (faultEl) faultEl.style.color = fault ? '#f87171' : '';
    const lines = [];
    if (id.synthetic) lines.push(`SYNTHETIC TEST GRAPH · backend ${id.backend || '?'} · not a scientific result`);
    else if (id.test_mode) lines.push(`TEST MODE · ${id.label || id.backend} · not a scientific result`);
    if (motor.motor_source && motor.motor_source_normal === false) {
        lines.push(`MOTOR SOURCE ${motor.motor_source}: ${MOTOR_SOURCE_NOTES[motor.motor_source] || 'not a normal controller source'}`);
    }
    if (fault) lines.push(`CONTROLLER FAULT: ${fault}`);
    const banner = document.getElementById('identityBanner');
    if (banner) {
        const text = lines.join(' · ');
        if (banner.textContent !== text) banner.textContent = text;
        banner.style.display = text ? 'block' : 'none';
    }
}

// -----------------------------------------------------------------------------
// Structured error reporting. Every failure names its phase and the assay, run and
// step it hit; repeats are counted, not re-logged, so no error loop floods the
// console. Nothing here resets the brain or invents data: a failing view phase is
// suspended (the last measured frame stays on screen) until the user resumes it.
// Diagnostics: window.neuroflyDiagnostics. Test-build fault injection:
// ?inject=malformed | apply | render (one-shot, see DaemonBridgeClient / main loop).
// -----------------------------------------------------------------------------
const NEUROFLY_INJECT = (() => {
    try { return new URLSearchParams(window.location.search || '').get('inject') || ''; } catch (e) { return ''; }
})();

const NeuroflyErrors = {
    log: [],
    counts: new Map(),
    suspended: new Set(),
    report(phase, err, ctx = {}) {
        const pkt = window.arena?.remotePacket;
        const entry = {
            phase,
            message: String(err?.message ?? err),
            assay: ctx.assay ?? window.arena?.activeParadigmId ?? null,
            run_id: ctx.run_id ?? pkt?.run_id ?? null,
            step: ctx.step ?? pkt?.step ?? null,
            time: new Date().toISOString(),
            stack: typeof err?.stack === 'string' ? err.stack.split('\n').slice(0, 4).join(' | ') : ''
        };
        const key = `${phase}:${entry.message}`;
        const count = (this.counts.get(key) || 0) + 1;
        this.counts.set(key, count);
        entry.count = count;
        if (count === 1 || count % 100 === 0) console.error('[NeuroFly]', phase, entry);
        this.log.push(entry);
        if (this.log.length > 50) this.log.shift();
        this.show(entry);
        return entry;
    },
    suspend(phase) { this.suspended.add(phase); },
    isSuspended(phase) { return this.suspended.has(phase); },
    resume() { this.suspended.clear(); this.hide(); },
    show(entry) {
        const banner = document.getElementById('errorBanner');
        const text = document.getElementById('errorBannerText');
        if (!banner || !text) return;
        const where = `assay ${entry.assay || '?'} · run ${(entry.run_id || '?').slice(0, 8)} · step ${entry.step ?? '?'}`;
        const repeat = entry.count > 1 ? ` (×${entry.count})` : '';
        const frozen = this.suspended.size ? ` · view suspended (${[...this.suspended].join(', ')}); last measured frame kept` : '';
        text.textContent = `${entry.phase.toUpperCase()} ERROR${repeat}: ${entry.message} — ${where}${frozen}`;
        banner.style.display = 'flex';
        const resume = document.getElementById('btnErrorResume');
        if (resume) resume.hidden = this.suspended.size === 0;
    },
    hide() {
        const banner = document.getElementById('errorBanner');
        if (banner) banner.style.display = 'none';
    }
};
window.neuroflyErrors = NeuroflyErrors;

/** Why a parsed daemon packet cannot be applied, or null when it can. */
function validateDaemonPacket(pkt) {
    if (!pkt || typeof pkt !== 'object' || Array.isArray(pkt)) return 'packet is not an object';
    if (pkt.type !== 'telemetry') return `unexpected packet type ${JSON.stringify(pkt.type)}`;
    if (!Number.isFinite(pkt.step)) return 'step is missing or not a finite number';
    if (typeof pkt.paradigm !== 'string' || !pkt.paradigm) return 'paradigm is missing';
    if (!pkt.fly || !Number.isFinite(pkt.fly.x) || !Number.isFinite(pkt.fly.y)) return 'fly position is missing or not finite';
    if (pkt.path !== undefined && !Array.isArray(pkt.path)) return 'path is not an array';
    return null;
}

class DaemonBridgeClient {
    constructor(arena, hud) {
        this.arena = arena;
        this.hud = hud;
        this.statusPill = document.getElementById('clusterStatusPill');
        this.connected = false;
        this.daemonPort = window.location.port === '8780' ? 8781 : 8769;
        this.eventSource = null;
        this.reconnectTimer = null;
        this.reconnectDelayMs = 4000;
        this.activeUrl = null;
        this.lastPacketTime = 0;      // last SSE event of any kind (data or heartbeat)
        this.lastValidDataTime = 0;   // last frame that passed validation and was applied
        this.lastOrderedPacket = null;
        // Old data is shown as STALE while the stream is open; only a stream that is
        // completely silent (no frame, no heartbeat) for deadAfterMs is torn down.
        // Tearing down a slow-but-alive stream froze the view at 100x (rethink audit).
        this.staleAfterMs = 3000;
        this.deadAfterMs = 20000;
        this.rejectedPackets = 0;
        this.streamStats = null;
        this.lastAck = null;
        this.lastSwitchAck = null;          // ack of this tab's last successful switch (identity)
        this.rejectedIdentityPackets = 0;
        this.lastIdentityRejection = null;
        this.manifest = null;               // full run manifest (GET /api/manifest), for exports
        this.manifestRunId = null;
        this.injectPending = NEUROFLY_INJECT;

        if (this.statusPill) {
            this.statusPill.addEventListener('click', () => {
                this.initConnection(true);
            });
        }

        // Watchdog: an EventSource that silently stops delivering (proxy timeout, daemon
        // restart) never fires onerror, so treat a fully silent stream as disconnected.
        this.watchdogTimer = setInterval(() => {
            const now = performance.now();
            if (this.connected && this.lastPacketTime > 0 && (now - this.lastPacketTime) > this.deadAfterMs) {
                this.onDaemonDisconnected(`stream silent for ${Math.round((now - this.lastPacketTime) / 1000)} s`);
                this.scheduleReconnect();
            }
            this.updateFreshness();
        }, 250);

        window.neuroflyDiagnostics = {
            errors: NeuroflyErrors.log,
            bridge: this,
            dataAgeMs: () => this.lastValidDataTime ? performance.now() - this.lastValidDataTime : null,
            lastStep: () => this.lastOrderedPacket?.step ?? null,
        };

        const researchLink = document.getElementById('researchLink');
        if (researchLink) researchLink.addEventListener('click', () => document.getElementById('tabTraining').click());
        this.initConnection(false);
    }

    /**
     * Daemon base URLs to probe, in order:
     *   1. `?daemon=http://host:port` query parameter (explicit),
     *   2. the page's own origin (dashboard served by the daemon or behind one proxy),
     *   3. the page's host on the daemon port,
     *   4. localhost / 127.0.0.1 on the daemon port.
     */
    get candidateUrls() {
        const list = [];
        const port = this.daemonPort;
        const add = (u) => { if (u && !list.includes(u)) list.push(u); };
        if (typeof window !== 'undefined' && window.location) {
            const loc = window.location;
            try {
                const param = new URLSearchParams(loc.search || '').get('daemon');
                if (param === 'off') return [];
                // An explicit endpoint must never fall through to a different live lab.
                if (param) return [param.replace(/\/+$/, '')];
            } catch (e) {}
            const httpLike = loc.protocol === 'http:' || loc.protocol === 'https:';
            if (httpLike && loc.origin && loc.origin !== 'null') add(loc.origin);
            if (httpLike && loc.hostname) add(`${loc.protocol}//${loc.hostname}:${port}`);
        }
        add(`http://localhost:${port}`);
        add(`http://127.0.0.1:${port}`);
        return list;
    }

    scheduleReconnect() {
        if (this.reconnectTimer) clearTimeout(this.reconnectTimer);
        this.reconnectTimer = setTimeout(() => this.initConnection(false), this.reconnectDelayMs);
        this.reconnectDelayMs = Math.min(30000, Math.round(this.reconnectDelayMs * 1.5));
    }

    async initConnection(force = false) {
        if (this.reconnectTimer) clearTimeout(this.reconnectTimer);
        if (this.probing && !force) return;
        this.probing = true;
        if (force) this.reconnectDelayMs = 4000;
        let foundUrl = null;

        for (const url of this.candidateUrls) {
            try {
                const res = await fetch(`${url}/api/status`, {
                    method: 'GET',
                    signal: AbortSignal.timeout(4000)
                });
                if (res.ok) {
                    const status = await res.json();
                    if (status && status.status === 'online') {
                        foundUrl = url;
                        this.activeUrl = url;
                        this.onDaemonConnected(status);
                        break;
                    }
                }
            } catch (e) {}
        }
        this.probing = false;

        if (!foundUrl) {
            this.onDaemonDisconnected();
            this.scheduleReconnect();
        }
    }

    onDaemonConnected(status) {
        this.connected = true;
        this.readOnly = false;
        this.reconnectDelayMs = 4000;
        this.lastPacketTime = performance.now();
        this.lastOrderedPacket = null;
        this.lastTrailStep = -1;
        this.showingStale = false;
        if (this.statusPill) {
            this.statusPill.textContent = '● LIVE DAEMON';
            this.statusPill.style.background = 'rgba(34, 197, 94, 0.25)';
            this.statusPill.style.border = '1px solid #22c55e';
            this.statusPill.style.color = '#4ade80';
            this.statusPill.title = `Connected to the learning daemon at ${this.activeUrl} (assay: ${status.active_paradigm}, ${status.total_steps} steps, ${status.uptime_sec}s uptime)`;
        }
        // Start the instrument on the experiment actually running on the daemon.
        // Updating the view must not send a switch command or create a new brain.
        const pid = (status.active_paradigm || '').replace(/_/g, '-');
        if (Object.prototype.hasOwnProperty.call(EXPERIMENT_GUIDES, pid)
                && pid !== this.arena.activeParadigmId) {
            this.arena.initParadigm(pid);
            this.hud.updateActiveCard(pid);
            this.hud.renderExperimentGuide(pid);
            this.hud.renderAssayTools(pid);
            const badge = document.getElementById('navbarParadigmBadge');
            if (badge) badge.textContent = pid.toUpperCase().replace(/-/g, ' ');
        }
        if (status.public === true || status.read_only === true
                || status.stream?.read_only || status.stream?.commands_require_token) this.markReadOnly();
        this.arena.awaitingDaemon = true; // First telemetry frame supplies the authoritative pose.
        this.startStreaming();
    }

    onDaemonDisconnected(reason = '') {
        this.connected = false;
        this.readOnly = false;
        this.disconnectReason = reason;
        // Keep the last scientific frame still during an outage; never silently
        // substitute a locally generated trajectory into a daemon recording.
        // remoteDriven stays as it was, so labels, units and measured metrics of the
        // last frame remain on screen instead of switching to local preview values.
        this.arena.awaitingDaemon = !!this.activeUrl;
        if (!this.activeUrl) this.arena.remoteDriven = false;
        if (this.statusPill) {
            this.statusPill.textContent = this.arena.awaitingDaemon ? '○ DISCONNECTED · FROZEN VIEW' : '○ LOCAL ENGINE';
            this.statusPill.style.background = 'rgba(148, 163, 184, 0.15)';
            this.statusPill.style.border = '1px solid #64748b';
            this.statusPill.style.color = '#94a3b8';
            this.statusPill.title = this.arena.awaitingDaemon
                ? `Daemon connection lost${reason ? ' (' + reason + ')' : ''}. The last measured frame (step ${this.lastOrderedPacket?.step ?? '?'}) is kept unchanged; no local data replaces it. Click to retry now.`
                : `In-browser simulation engine active (offline/standalone mode${reason ? ': ' + reason : ''}). Click to retry the daemon connection, or open the page with ?daemon=http://host:${this.daemonPort}.`;
        }
        if (this.eventSource) {
            this.eventSource.onerror = null;
            this.eventSource.onmessage = null;
            this.eventSource.close();
            this.eventSource = null;
        }
        this.updateFreshness();
    }

    /** Data-age indicator, achieved speed and the LIVE / STALE pill state (4 Hz). */
    updateFreshness() {
        const ageEl = document.getElementById('statDataAge');
        const age = this.lastValidDataTime ? (performance.now() - this.lastValidDataTime) / 1000 : null;
        if (ageEl) {
            ageEl.textContent = age === null ? '--' : `${age < 10 ? age.toFixed(1) : Math.round(age)}s${this.connected ? '' : ' (frozen)'}`;
            ageEl.style.color = age === null ? '' : (!this.connected ? '#f87171' : age * 1000 > this.staleAfterMs ? '#fbbf24' : '#4ade80');
        }
        if (!this.connected || !this.statusPill || age === null) return;
        const stale = age * 1000 > this.staleAfterMs;
        if (stale === !!this.showingStale) return;
        this.showingStale = stale;
        const ro = this.readOnly ? ' (READ-ONLY)' : '';
        this.statusPill.textContent = stale ? `● LIVE DAEMON${ro} · STALE DATA` : `● LIVE DAEMON${ro}`;
        this.statusPill.style.color = stale ? '#fbbf24' : '#4ade80';
        this.statusPill.style.border = stale ? '1px solid #f59e0b' : '1px solid #22c55e';
    }

    startStreaming() {
        if (!this.activeUrl) return;
        if (this.eventSource) this.eventSource.close();

        try {
            this.eventSource = new EventSource(`${this.activeUrl}/api/stream`);
            this.eventSource.onmessage = (event) => this.receive(event.data);
            // Liveness without new data (paused or slow daemon): keeps the stream open.
            this.eventSource.addEventListener('heartbeat', () => { this.lastPacketTime = performance.now(); });
            this.eventSource.addEventListener('stream', (event) => {
                this.lastPacketTime = performance.now();
                try { this.streamStats = JSON.parse(event.data); }
                catch (e) { NeuroflyErrors.report('packet-parse', e, {step: this.lastOrderedPacket?.step}); }
            });
            this.eventSource.onerror = () => {
                // EventSource retries on its own; close it and re-probe with backoff so a
                // dead daemon does not produce a tight reconnect loop.
                this.onDaemonDisconnected('stream error');
                this.scheduleReconnect();
            };
        } catch (e) {
            NeuroflyErrors.report('startup', e, {});
            this.onDaemonDisconnected('stream could not be opened');
        }
    }

    /** Parse, validate and apply one SSE frame. A bad frame is rejected whole and reported. */
    receive(raw) {
        this.lastPacketTime = performance.now();
        const last = this.lastOrderedPacket;
        let pkt;
        try {
            pkt = JSON.parse(raw);
        } catch (e) {
            this.rejectedPackets++;
            NeuroflyErrors.report('packet-parse', e, {run_id: last?.run_id, step: last?.step, assay: last?.paradigm});
            return false;
        }
        const problem = validateDaemonPacket(pkt);
        if (problem) {
            this.rejectedPackets++;
            NeuroflyErrors.report('packet-validate', new Error(problem),
                {run_id: pkt?.run_id ?? last?.run_id, step: Number.isFinite(pkt?.step) ? pkt.step : last?.step, assay: pkt?.paradigm ?? last?.paradigm});
            return false;
        }
        try {
            if (this.injectPending === 'apply') { this.injectPending = ''; throw new Error('Injected packet-apply fault (test build)'); }
            const applied = this.handleDaemonPacket(pkt);
            if (applied !== false) this.lastValidDataTime = performance.now();
        } catch (e) {
            NeuroflyErrors.report('packet-apply', e, {run_id: pkt.run_id, step: pkt.step, assay: pkt.paradigm});
            return false;
        }
        if (this.injectPending === 'malformed') {
            // One-shot test-build fault: a truncated frame and a structurally invalid one,
            // through the same path as real frames. Both must be rejected; the view keeps
            // the last valid frame and the next real frame is applied normally.
            this.injectPending = '';
            this.receive('{"type":"telemetry","step":');
            this.receive(JSON.stringify({type: 'telemetry', step: pkt.step + 1, paradigm: pkt.paradigm, run_id: pkt.run_id, fly: {x: 'NaN', y: null}}));
        }
        return true;
    }

    handleDaemonPacket(pkt) {
        if (!pkt || pkt.type !== 'telemetry') return false;

        // Drop stale / out-of-order packets (SSE reconnects can replay old frames).
        if (isStaleDaemonPacket(pkt, this.lastOrderedPacket)) return false;
        // Packets produced before this tab's acknowledged switch (older activation or a
        // different run/instance) must not be drawn under the new run's identity.
        const identityProblem = identityRejection(pkt, this.lastSwitchAck);
        if (identityProblem) {
            this.rejectedIdentityPackets++;
            this.lastIdentityRejection = identityProblem;
            return false;
        }
        const step = Number.isFinite(pkt.step) ? pkt.step : null;
        this.lastPacketTime = performance.now();
        this.lastOrderedPacket = pkt;
        renderIdentity(pkt);
        if (pkt.identity?.run_id && pkt.identity.run_id !== this.manifestRunId) this.fetchManifest(pkt.identity.run_id);

        // Synchronize fly pose from the daemon ONLY when the paradigms match; the daemon
        // then owns locomotion and the local engine stops integrating position.
        const daemonParadigm = (pkt.paradigm || '').toLowerCase().replace(/_/g, '-');
        if (daemonParadigm !== this.arena.activeParadigmId
                && Object.prototype.hasOwnProperty.call(EXPERIMENT_GUIDES, daemonParadigm)) {
            this.arena.initParadigm(daemonParadigm);
            this.hud.updateActiveCard(daemonParadigm);
            this.hud.renderExperimentGuide(daemonParadigm);
            this.hud.renderAssayTools(daemonParadigm);
            document.getElementById('navbarParadigmBadge').textContent=daemonParadigm.toUpperCase().replace(/-/g,' ');
        }
        const activeParadigm = (this.arena.activeParadigmId || '').toLowerCase().replace(/_/g, '-');
        const poseMatch = !!(pkt.fly && daemonParadigm === activeParadigm
            && Number.isFinite(pkt.fly.x) && Number.isFinite(pkt.fly.y));
        this.arena.remoteDriven = poseMatch;

        if (poseMatch) {
            this.arena.awaitingDaemon = false;
            const segment = pkt.segment_id || `${pkt.brain_id}:${pkt.trial}`;
            const boundary = this.arena.remoteSegment !== segment;
            if (boundary) { this.arena.fly.trail = []; this.hud.scopeHistory = []; }
            this.arena.remoteSegment = segment;
            this.arena.remotePacket = pkt;
            this.arena.currentTrial = pkt.trial;
            this.arena.paradigmElapsedSec = pkt.trial_elapsed_s;
            this.arena.simTime = pkt.sim_time_s ?? pkt.step * 0.02;
            this.arena.stepCount = pkt.step;
            this.arena.paradigmStatus = pkt.error ? `SIMULATION ERROR: ${pkt.error}` : pkt.paused ? 'PAUSED' : pkt.brain?.teaching ? 'CUE TEACHING · ARENA PAUSED' : `${pkt.fly.state} · ${pkt.continuous ? 'CONTINUOUS OBSERVATION' : 'TRIAL ' + pkt.trial}`;
            const phaseLabel = document.getElementById('arenaRunState');
            if (phaseLabel) phaseLabel.textContent = this.arena.paradigmStatus;
            this.hud.simSpeed = pkt.sim_speed;
            document.getElementById('statSpeed').textContent = `${pkt.sim_speed}x`;
            const achievedEl = document.getElementById('statAchieved');
            const timing = pkt.timing;
            if (achievedEl && timing && Number.isFinite(timing.achieved_speed)) {
                achievedEl.textContent = pkt.paused ? 'paused' : `${timing.achieved_speed.toFixed(1)}x`;
                achievedEl.style.color = timing.overloaded ? '#fbbf24' : '';
                const dropped = this.streamStats ? ` Display decimation: ${this.streamStats.decimated_snapshots} snapshots skipped in the last second (latest-value-wins).` : '';
                achievedEl.title = `Requested ${timing.requested_speed}x, measured ${timing.achieved_speed}x; fixed dt ${timing.integration_dt_s} s; ${timing.steps_in_frame ?? '?'} steps in this frame.`
                    + (timing.overloaded ? ' This computer cannot run the requested speed; the daemon runs slower instead of skipping steps.' : '') + dropped;
            } else if (achievedEl) {
                achievedEl.textContent = '--';
            }
            document.getElementById('btnSpeedToggle').textContent = `Speed: ${pkt.sim_speed}x`;
            document.getElementById('selectSpeed').value = String(pkt.sim_speed);
            const pause = document.getElementById('btnPauseToggle');
            pause.textContent = pkt.paused ? 'Resume' : 'Pause';
            this.arena.mb.kcFiring = pkt.neural?.kc_hz || this.arena.mb.kcFiring;
            if (Number.isFinite(pkt.neural?.net_valence)) this.arena.mb.netValence = pkt.neural.net_valence;
            this.arena.mb.pamRate = pkt.neural?.pam_trace || 0;
            this.arena.mb.ppl1Rate = pkt.neural?.ppl1_trace || 0;
            this.arena.cx.updateBump(pkt.neural?.compass_heading ?? pkt.fly.heading);
            this.arena.cpg.steppingFreq = pkt.paused || pkt.brain?.teaching ? 0 : (pkt.biomechanics?.cadence_hz || 0);
            this.arena.cpg.phaseA = this.arena.simTime * this.arena.cpg.steppingFreq * 2 * Math.PI;
            this.arena.cpg.phaseB = this.arena.cpg.phaseA + Math.PI;
            this.arena.dn.dna02Diff = pkt.descending?.dna02_yaw || 0;
            this.arena.fly.yawRate = pkt.descending?.dna02_yaw || 0;
            if (Number.isFinite(pkt.sensory?.wind_x) && Number.isFinite(pkt.sensory?.wind_y)) {
                this.arena.windVector = [pkt.sensory.wind_x, pkt.sensory.wind_y];
            }
            this.arena.dn.dnp09 = pkt.descending?.dnp09_thrust || 0;
            this.arena.dn.mdn = pkt.descending?.mdn_reverse || 0;
            this.arena.dn.escapeActive = !!pkt.descending?.gf_escape;
            // Export one measured row per distinct daemon tick. No synthetic FPS samples.
            if (this.lastRecordedSegment !== segment || this.lastRecordedStep !== pkt.step) {
                if (this.arena.telemetryBuffer[0]?.source === 'local_preview') this.arena.telemetryBuffer = [];
                this.arena.telemetryBuffer.push({source:'daemon', run_id:pkt.run_id || '', brain_id:pkt.brain_id, segment,
                    controller_run_id:pkt.identity?.run_id || '', instance_id:pkt.identity?.instance_id || '',
                    backend:pkt.identity?.backend || '', synthetic:!!pkt.identity?.synthetic,
                    motor_source:pkt.motor?.motor_source || '', assists:!!pkt.motor?.motor_assists_enabled,
                    controller_fault:pkt.controller_fault || '',
                    boundary:boundary, transition:boundary ? (pkt.transition?.reason || 'segment_start') : '',
                    step:pkt.step, simTime:pkt.sim_time_s ?? pkt.step*.02, paradigm:pkt.paradigm, trial:pkt.trial,
                    trialTime:pkt.trial_elapsed_s, x:pkt.fly.x, y:pkt.fly.y, heading:pkt.fly.heading, speed:pkt.fly.speed,
                    state:pkt.fly.state, paused:!!pkt.paused, teaching:!!pkt.brain?.teaching, error:pkt.error || ''});
                this.arena.telemetryBuffer = this.arena.telemetryBuffer.slice(-4500);
                this.lastRecordedStep = pkt.step; this.lastRecordedSegment = segment;
            }
            const off = daemonFrameOffset(pkt);
            const state = this.arena.paradigmState, m = pkt.metrics || {}, stimulus = pkt.stimuli || {}, assay = pkt.assay_state || {};
            if (state) {
                const fields = {performance_index:'performanceIndex',spontaneous_alternation_rate:'sar',
                    escape_latency_ms:'escapeLatencyMs',centrophobism_index:'centrophobism',operant_learning_index:'learningIndex',
                    surge_to_cast_ratio:'surgeCastRatio',time_to_source_ms:'timeToSourceMs',source_reached:'sourceReached',
                    optomotor_gain:'gain',mean_hs_firing_rate:'hsFiringRate',gap_width_mm:'gapWidthMm',
                    decision_outcome:'decisionOutcome',crossing_success:'crossingSuccess',total_sleep_minutes:'totalSleepMin',
                    total_beam_crossings:'beamCrossings',courtship_index:'courtshipIndex',rejection_kicks_count:'rejectionKicks',
                    time_to_goal_ms:'timeToGoalMs',path_tortuosity:'pathTortuosity',
                    mean_stripe_fixation:'meanFixation',stripe_crossings:'stripeCrossings',refuge_reached:'refugeReached',
                    goal_reached:'goalReached',total_distance_mm:'pathLength',probing_duration_ms:'probingDurationMs',
                    surge_steps:'surgeSteps',cast_steps:'castSteps',upwind_progress_mm:'upwindProgress',mean_retinal_slip:'effectiveSlip'};
                for (const [key,target] of Object.entries(fields)) if (m[key] !== undefined) state[target]=m[key];
                state.behavioralState=pkt.fly.state;
                if(activeParadigm==='y-maze' && typeof m.choice_sequence==='string') {
                    state.armCounts=['A','B','C'].map(letter=>Array.from(m.choice_sequence).filter(v=>v===letter).length);
                }
                if(activeParadigm==='t-maze') {
                    const reversed=pkt.scene?.cs_plus_arm==='arm_b';
                    state.choiceCounts={arm_a:reversed?m.cs_minus_choices:m.cs_plus_choices,arm_b:reversed?m.cs_plus_choices:m.cs_minus_choices};
                    state.shockPulse=assay.punishment||0;
                }
                if (pkt.scene?.nozzle_pos) state.nozzlePos=pkt.scene.nozzle_pos;
                if (pkt.scene?.filament_sigma!==undefined) state.filamentSigma=pkt.scene.filament_sigma;
                if (pkt.scene?.cs_plus_arm) state.csPlusArm=pkt.scene.cs_plus_arm;
                if (stimulus.drum_velocity_deg_s!==undefined) state.drumVelocityDegS=stimulus.drum_velocity_deg_s;
                if (stimulus.contrast!==undefined) state.contrast=stimulus.contrast;
                if (assay.wing_extension_angle_deg!==undefined) state.wingAngleDeg=assay.wing_extension_angle_deg;
                if (stimulus.temperature !== undefined) state.temp=stimulus.temperature;
                if (stimulus.laser_active !== undefined) state.laserActive=stimulus.laser_active;
                if (pkt.scene?.drum_angle_deg !== undefined) state.drumAngleDeg=pkt.scene.drum_angle_deg;
                if (assay.yaw_torque !== undefined) state.yawTorque=assay.yaw_torque;
                if (assay.hs_firing_rate !== undefined) state.hsFiringRate=assay.hs_firing_rate;
                if (stimulus.looming_angle_deg !== undefined) state.thetaDeg=stimulus.looming_angle_deg;
                if (stimulus.theta_deg !== undefined) state.thetaDeg=stimulus.theta_deg;
                if (activeParadigm==='optomotor') state.drumAngleDeg=(stimulus.drum_velocity_deg_s||0)*pkt.trial_elapsed_s%360;
                if (pkt.scene?.female_pos) state.femalePos=pkt.scene.female_pos.map((v,i)=>v+off[i]);
                if (pkt.scene?.female_type) state.femaleType=pkt.scene.female_type;
            }
            if (activeParadigm==='open-arena' && pkt.scene) {
                this.arena.foodItems=pkt.scene.food.map(([x,y])=>({x:x+off[0],y:y+off[1],radius:3,odorStrength:1}));
                this.arena.alarms=pkt.scene.hazards.map(([x,y])=>({x:x+off[0],y:y+off[1],strength:1,life:1}));
                this.arena.predators=pkt.scene.predators.map(([x,y,vx,vy])=>({x:x+off[0],y:y+off[1],vx,vy,radius:5,active:true}));
            }
            let fx = pkt.fly.x + off[0];
            let fy = pkt.fly.y + off[1];
            // The daemon's coordinates are authoritative: they are never re-clamped with
            // the browser's own geometry. Its body radius and world bounds (present in
            // newer telemetry) are adopted; the bounds only guard against a corrupt frame.
            if (Number.isFinite(pkt.fly.radius) && pkt.fly.radius > 0) this.arena.fly.radius = pkt.fly.radius;
            const wb = Array.isArray(pkt.world_bounds) && pkt.world_bounds.length === 4 ? pkt.world_bounds : null;
            if (wb && wb.every(Number.isFinite)) {
                const rr = this.arena.fly.radius;
                fx = Math.max(wb[0] + off[0] + rr, Math.min(wb[2] + off[0] - rr, fx));
                fy = Math.max(wb[1] + off[1] + rr, Math.min(wb[3] + off[1] - rr, fy));
            }
            this.arena.fly.x = fx;
            this.arena.fly.y = fy;
            if (Number.isFinite(pkt.fly.heading)) {
                // Daemon headings are 0..2pi; the browser keeps -pi..pi.
                this.arena.fly.heading = ((pkt.fly.heading + Math.PI) % (2 * Math.PI) + 2 * Math.PI) % (2 * Math.PI) - Math.PI;
            }
            if (Number.isFinite(pkt.fly.speed)) this.arena.fly.speed = pkt.fly.speed;
            if (this.arena.fly.trail) {
                // Measured positions of every step since earlier frames ([step, x, y]); a
                // decimated display still draws the path actually taken, every 5th step.
                if (boundary) this.lastTrailStep = -1;
                const escape = !!(pkt.descending && pkt.descending.gf_escape > 0.5);
                const points = Array.isArray(pkt.path) && pkt.path.length ? pkt.path : [[step, pkt.fly.x, pkt.fly.y]];
                for (const [s, x, y] of points) {
                    if (!Number.isFinite(s) || s <= (this.lastTrailStep ?? -1) || s % 5 !== 0) continue;
                    if (!Number.isFinite(x) || !Number.isFinite(y)) continue;
                    this.arena.fly.trail.push({ x: x + off[0], y: y + off[1], speed: this.arena.fly.speed, escape });
                    this.lastTrailStep = s;
                }
                if (this.arena.fly.trail.length > 200) this.arena.fly.trail = this.arena.fly.trail.slice(-150);
            }
        }

        // Synchronize learning curve from daemon (the daemon sends its last 30 points;
        // mirror them instead of appending, so the list stays bounded and in order).
        if (pkt.plasticity && Array.isArray(pkt.plasticity.learning_curve) ) {
            const curve = pkt.plasticity.learning_curve.filter(v => Number.isFinite(v));
            if (this.hud && this.hud.learningTrials && poseMatch) {
                const base = Math.max(0, (pkt.trial || curve.length) - curve.length);
                this.hud.learningTrials = curve.map((v, i) => ({ trial: base + i + 1, value: v, formatted: v.toFixed(2) }));
            }
        }

        // Synchronize 6-leg joint angles from daemon
        if (pkt.biomechanics && pkt.biomechanics.joint_angles && poseMatch) {
            const ja = pkt.biomechanics.joint_angles;
            const box = document.getElementById('jointAnglesBox');
            if (box) {
                const fmt = (v) => Number.isFinite(v) ? v.toFixed(0) : '--';
                box.innerHTML = Object.entries(ja).map(([leg, info]) => {
                    const ctrSign = (info && info.ctr >= 0) ? '+' : '';
                    return `<div>${leg}: CTr: ${ctrSign}${fmt(info && info.ctr)}° FTi: ${fmt(info && info.fti)}° [${(info && info.phase) || '--'}]</div>`;
                }).join('');
            }
            // Mirror the daemon's kinematics into the local state so hud.update() (which
            // redraws the same box every frame) shows the streamed values, not local ones.
            const p = this.arena.paradigmState;
            if (p && p.jointAngles) {
                for (const [leg, info] of Object.entries(ja)) {
                    if (p.jointAngles[leg] && info) {
                        p.jointAngles[leg].ctr = Number.isFinite(info.ctr) ? info.ctr : p.jointAngles[leg].ctr;
                        p.jointAngles[leg].fti = Number.isFinite(info.fti) ? info.fti : p.jointAngles[leg].fti;
                        this.arena.cpg.legStates[leg] = info.phase === 'STANCE';
                    }
                }
            }
        }

        // Synchronize composite benchmark scorecard (daemon schema: metrics.{composite_benchmark_score,
        // locomotor_coordination_index, multisensory_integration_score, biomechanical_efficiency,
        // kinematic_smoothness, wall_collisions, total_distance_mm, total_energy_atp}).
        if (pkt.metrics && poseMatch && this.arena.activeParadigmId === 'multisensory-sandbox') {
            const m = pkt.metrics;
            const p = this.arena.paradigmState;
            const num = (v, fallback) => Number.isFinite(v) ? v : fallback;
            if (p) {
                p.compositeScore = num(m.composite_benchmark_score, p.compositeScore);
                p.coordinationScore = num(m.locomotor_coordination_index, p.coordinationScore);
                p.sensoryIntegrationScore = num(m.multisensory_integration_score, p.sensoryIntegrationScore);
                p.efficiencyScore = num(m.biomechanical_efficiency, p.efficiencyScore);
                p.smoothnessScore = num(m.kinematic_smoothness, p.smoothnessScore);
                p.wallCollisions = num(m.wall_collisions, p.wallCollisions);
                p.totalDistance = num(m.total_distance_mm, p.totalDistance);
                p.totalEnergy = num(m.total_energy_atp, p.totalEnergy);
            }
        }
    }

    /** Marks the daemon as read-only (it runs with --public and refuses /api/command). */
    markReadOnly() {
        if (this.readOnly) return;
        this.readOnly = true;
        if (this.statusPill && this.connected) {
            this.statusPill.textContent = '● LIVE DAEMON (READ-ONLY)';
            this.statusPill.title = `Connected to a public learning daemon at ${this.activeUrl}: the stream is live, but commands (paradigm switches, speed, stimuli) are only applied to the in-browser engine.`;
        }
    }

    async requestParadigmSwitch(paradigm) {
        // Serialize requests; while one is in flight retain the latest click.
        // Only incoming telemetry changes the visible arena, never the click.
        this.queuedParadigm = paradigm;
        if (this.switchQueueRunning) return;
        this.switchQueueRunning = true;
        try {
            while (this.queuedParadigm) {
                const target = this.queuedParadigm;
                this.queuedParadigm = null;
                const result = await this.sendCommand('switch_paradigm', {paradigm:target});
                if (result?.status !== 'ok') {
                    const label=document.getElementById('arenaRunState');
                    if(label)label.textContent='Switch not applied: '+(result?.message||'daemon unavailable or read only');
                }
            }
        } finally { this.switchQueueRunning = false; }
    }

    /** Fetch the full manifest of the streamed run once per run_id (exports embed it). */
    async fetchManifest(runId) {
        this.manifestRunId = runId;
        if (!this.activeUrl) return;
        try {
            const res = await fetch(`${this.activeUrl}/api/manifest`, {signal: AbortSignal.timeout(3000)});
            if (!res.ok) return;
            const data = await res.json();
            if (data?.manifest?.run_id === runId) this.manifest = data.manifest;
            else this.manifestRunId = null;   // switched meanwhile; retry on the next packet
        } catch (e) {
            this.manifestRunId = null;
        }
    }

    async sendCommand(action, params = {}) {
        if (!this.connected || !this.activeUrl || this.readOnly) return null;
        if (action === 'switch_paradigm') this.switchPending = true;
        try {
            const res = await fetch(`${this.activeUrl}/api/command`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ action, params }),
                signal: AbortSignal.timeout(2000)
            });
            if (res.status === 403) {
                this.markReadOnly();
                return null;
            }
            if (res.ok) {
                const data=await res.json();
                if(data.status==='error') console.warn('[DaemonBridge] Command rejected:',data.message);
                // The daemon acknowledges the step at which the command took effect.
                if (data.ack) this.lastAck = {...data.ack, action};
                // A switch is acknowledged only after the target's brain and world are
                // ready; from now on older-identity packets are rejected.
                if (action === 'switch_paradigm' && data.status === 'ok' && data.ack?.identity) this.lastSwitchAck = data.ack;
                return data;
            }
        } catch (e) {
            console.warn('[DaemonBridge] sendCommand error:', e);
        } finally { if (action === 'switch_paradigm') this.switchPending = false; }
        return null;
    }
}


// =============================================================================
// 7C. SPECIALIZED SCIENTIFIC ASSAY SPECIFICATIONS (14 PARADIGMS)
// =============================================================================

const ASSAY_CONFIGS = {
    'open-arena': {
        title: 'Open Arena Multi-Modal Assay',
        badge: 'FORAGING & TAXIS',
        ref: 'Budick & Dickinson (2006) Animal Behaviour / Spatial Dispersal',
        sliders: [
            { key: 'wallRepulsion', label: 'Boundary Repulsion', min: 0.2, max: 3.0, step: 0.1, val: 1.0, unit: 'x', desc: 'Elastic boundary force pushing fly inward from perimeter boundaries to prevent boundary-sticking.', apply: (a, v) => { a.wallRepulsion = v; } },
            { key: 'windStrength', label: 'Wind Vector Speed', min: 0.0, max: 40.0, step: 2.0, val: 15.0, unit: ' mm/s', desc: 'Global environmental wind vector speed sensed by Johnston’s organ mechanoreceptors.', apply: (a, v) => { a.windVector[0] = -v; } }
        ],
        actions: [
            { label: 'Drop Food Pellet', class: 'primary', handler: (a, h) => { a.spawnFoodNearFly(); } },
            { label: 'Looming Shadow', class: 'danger', handler: (a, h) => { a.triggerLooming(); } },
            { label: 'Clear Trail', class: '', handler: (a, h) => { a.fly.trail = []; } }
        ],
        metrics: [
            { label: 'Ground Speed', get: (a) => `${a.fly.speed.toFixed(1)} mm/s` },
            { label: 'Dispersal Distance', get: (a) => `${Math.hypot(a.fly.x, a.fly.y).toFixed(1)} mm` },
            { label: 'Trail Length', get: (a) => `${a.fly.trail.length} pts` }
        ],
        drawChart: (ctx, w, h, a, hInst) => {
            ctx.fillStyle = '#38bdf8';
            ctx.font = '9px monospace';
            ctx.fillText('Radial Spatial Dispersal (mm from origin)', 8, 14);
            const pts = a.fly.trail.slice(-80);
            if (pts.length < 2) return;
            ctx.strokeStyle = '#0284c7';
            ctx.lineWidth = 1.2;
            ctx.beginPath();
            pts.forEach((p, i) => {
                const r = Math.hypot(p.x, p.y);
                const x = 20 + (i / 80) * (w - 30);
                const y = h - 10 - Math.min(h - 24, (r / 140.0) * (h - 24));
                if (i === 0) ctx.moveTo(x, y); else ctx.lineTo(x, y);
            });
            ctx.stroke();
        }
    },
    't-maze': {
        title: 'T-Maze Associative Conditioning',
        badge: 'TULLY-QUINN (1985)',
        ref: 'Tully & Quinn (1985) Science / Pavlovian Olfactory Memory',
        sliders: [
            { key: 'shockVoltage', label: 'Grid Shock Amplitude', min: 10, max: 100, step: 5, val: 60, unit: ' V', desc: 'Electric grid voltage in CS- arm. Sets PPL1 dopaminergic punishment spike rate during associative training.', apply: (a, v) => { if (a.paradigmState) a.paradigmState.shockVoltage = v; } },
            { key: 'vacuumAirflow', label: 'Vacuum Airflow Velocity', min: 2, max: 25, step: 1, val: 12, unit: ' cm/s', desc: 'Aspirated vacuum laminar flow pulling odors down each arm into the central elevator.', apply: (a, v) => { if (a.paradigmState) a.paradigmState.vacuumFlow = v; } }
        ],
        actions: [
            { label: 'Deliver Shock Pulse (PPL1)', class: 'danger', handler: (a, h) => {
                a.mb.stepPlasticity(0.0, 1.0, 5.0);
                if (a.paradigmState) a.paradigmState.shockPulse = 1.0;
                if (h.daemonBridge) h.daemonBridge.sendCommand('inject_stimulus', { type: 'thermal_flash', value: 40.0 });
            } },
            { label: 'Invert CS+/CS- Arms', class: 'primary', handler: (a, h) => {
                if (a.paradigmState) {
                    a.paradigmState.csPlusArm = (a.paradigmState.csPlusArm === 'arm_a' ? 'arm_b' : 'arm_a');
                }
            } },
            { label: 'Reset MB Synapses', class: '', handler: (a, h) => { a.mb.reset(false); } }
        ],
        metrics: [
            { label: 'Performance Index (PI)', get: (a) => `${(a.paradigmState && a.paradigmState.performanceIndex !== undefined ? a.paradigmState.performanceIndex : 0.0).toFixed(2)}` },
            { label: 'Shock Arm (CS+)', get: (a) => `${a.paradigmState ? (a.paradigmState.csPlusArm === 'arm_a' ? 'ARM A (OCT)' : 'ARM B (MCH)') : '--'}` },
            { label: 'Total Choices', get: (a) => `${a.paradigmState && a.paradigmState.choiceCounts ? (a.paradigmState.choiceCounts.arm_a + a.paradigmState.choiceCounts.arm_b) : 0}` }
        ],
        drawChart: (ctx, w, h, a, hInst) => {
            ctx.fillStyle = '#f8fafc';
            ctx.font = '9px monospace';
            ctx.fillText('Pavlovian Odor Avoidance (CS- vs CS+)', 8, 14);
            const pi = (a.paradigmState && a.paradigmState.performanceIndex !== undefined) ? a.paradigmState.performanceIndex : 0.0;
            const midY = h / 2 + 5;
            ctx.strokeStyle = 'rgba(255,255,255,0.1)';
            ctx.beginPath(); ctx.moveTo(20, midY); ctx.lineTo(w - 20, midY); ctx.stroke();
            const barW = 36;
            const barH = (pi * (h / 2 - 18));
            ctx.fillStyle = pi >= 0 ? '#38bdf8' : '#f43f5e';
            ctx.fillRect(w / 2 - barW / 2, midY - barH, barW, barH);
            ctx.fillStyle = '#94a3b8';
            ctx.fillText(`PI: ${pi >= 0 ? '+' : ''}${pi.toFixed(2)}`, w / 2 - 20, midY + 16);
        }
    },
    'y-maze': {
        title: 'Y-Maze Spontaneous Alternation & Handedness',
        badge: 'BUCHANAN ET AL. (NATURE 2015)',
        ref: 'Buchanan et al. (Nature 2015) Individual Idiosyncratic Handedness',
        sliders: [
            { key: 'handednessBias', label: 'Individual Handedness Perturbation', min: -1.0, max: 1.0, step: 0.1, val: 0.0, unit: ' bias', desc: 'Injected bilateral bias into DNa02 premotor steering circuit, skewing spontaneous left/right turning.', apply: (a, v) => { if (a.paradigmState) a.paradigmState.bias = v; } },
            { key: 'choiceHesitation', label: 'Hub Decision Delay', min: 0.1, max: 2.0, step: 0.1, val: 0.4, unit: ' s', desc: 'Bifurcation decision pause duration in the central choice hub before entering an arm.', apply: (a, v) => { if (a.paradigmState) a.paradigmState.delay = v; } }
        ],
        actions: [
            { label: 'Gate Left Arm', class: 'primary', handler: (a, h) => { if (a.paradigmState) a.paradigmState.gateLeft = !a.paradigmState.gateLeft; } },
            { label: 'Gate Right Arm', class: 'primary', handler: (a, h) => { if (a.paradigmState) a.paradigmState.gateRight = !a.paradigmState.gateRight; } },
            { label: 'Clear Turn History', class: '', handler: (a, h) => { if (a.paradigmState) { a.paradigmState.turnDirections = []; a.paradigmState.armSequence = []; } } }
        ],
        metrics: [
            { label: 'Alternation Rate (SAR)', get: (a) => `${((a.paradigmState && a.paradigmState.sar !== undefined ? a.paradigmState.sar : 0.67) * 100).toFixed(1)}%` },
            { label: 'Total Choices', get: (a) => `${a.paradigmState && a.paradigmState.turnDirections ? a.paradigmState.turnDirections.length : 0}` },
            { label: 'Left/Right Bias', get: (a) => {
                if (!a.paradigmState || !a.paradigmState.turnDirections || a.paradigmState.turnDirections.length === 0) return '0.00';
                const l = a.paradigmState.turnDirections.filter(t => t === 'L').length;
                const r = a.paradigmState.turnDirections.length - l;
                return ((r - l) / a.paradigmState.turnDirections.length).toFixed(2);
            } }
        ],
        drawChart: (ctx, w, h, a, hInst) => {
            ctx.fillStyle = '#f8fafc';
            ctx.font = '9px monospace';
            ctx.fillText('Spontaneous Alternation (Tri-Turn Sequences)', 8, 14);
            const sar = (a.paradigmState && a.paradigmState.sar !== undefined) ? a.paradigmState.sar : 0.67;
            const barW = (w - 60) * Math.min(1.0, sar);
            ctx.fillStyle = '#4ade80';
            ctx.fillRect(30, h / 2 - 10, barW, 20);
            ctx.strokeStyle = '#22c55e';
            ctx.strokeRect(30, h / 2 - 10, w - 60, 20);
            ctx.fillStyle = '#ffffff';
            ctx.fillText(`${(sar * 100).toFixed(1)}% Alternation`, 36, h / 2 + 4);
        }
    },
    'heat-maze': {
        title: 'Thermal Place Learning (Heat-Maze)',
        badge: 'OFSTAD ET AL. (NATURE 2011)',
        ref: 'Ofstad, Zuker & Reiser (Nature 2011) Visual Place Learning',
        sliders: [
            { key: 'floorTemp', label: 'Arena Floor Temperature', min: 32, max: 44, step: 1, val: 36.5, unit: ' °C', desc: 'Aversive heated floor temperature (°C) driving thermotactic escape search toward cool refuge.', apply: (a, v) => { if (a.paradigmState) a.paradigmState.hotTemp = v; } },
            { key: 'refugeRadius', label: 'Cool Refuge Radius', min: 6, max: 20, step: 1, val: 9, unit: ' mm', desc: 'Target cool tile radius (mm). Smaller radius requires tighter landmark triangulation by Central Complex.', apply: (a, v) => { if (a.paradigmState) a.paradigmState.refugeRadius = v; } }
        ],
        actions: [
            { label: 'Relocate Cool Refuge', class: 'primary', handler: (a, h) => {
                if (a.paradigmState) {
                    const angles = [Math.PI / 4, 3 * Math.PI / 4, 5 * Math.PI / 4, 7 * Math.PI / 4];
                    const choice = angles[Math.floor(Math.random() * angles.length)];
                    a.paradigmState.refugePos = [60.0 + Math.cos(choice) * 28.0, 60.0 + Math.sin(choice) * 28.0];
                }
            } },
            { label: 'Thermal Shock Flash (42°C)', class: 'danger', handler: (a, h) => {
                a.mb.stepPlasticity(0.0, 1.0, 6.0);
                if (h.daemonBridge) h.daemonBridge.sendCommand('inject_stimulus', { type: 'thermal_flash', value: 42.0 });
            } },
            { label: 'Reset Spatial Map', class: '', handler: (a, h) => { a.cx.headingBump = 0; } }
        ],
        metrics: [
            { label: 'Escape Latency', get: (a) => `${(a.paradigmState && a.paradigmState.escapeLatencyMs ? a.paradigmState.escapeLatencyMs / 1000 : a.paradigmElapsedSec).toFixed(1)}s` },
            { label: 'Current Surface Temp', get: (a) => `${(a.paradigmState && Number.isFinite(a.paradigmState.temp) ? a.paradigmState.temp : 36.5).toFixed(1)}°C` },
            { label: 'Cool Spot Reached', get: (a) => `${a.paradigmState && a.paradigmState.refugeReached ? 'YES' : 'SEARCHING'}` }
        ],
        drawChart: (ctx, w, h, a, hInst) => {
            ctx.fillStyle = '#fbbf24';
            ctx.font = '9px monospace';
            ctx.fillText('Escape Latency Progression (Trials 1..N)', 8, 14);
            const trials = (hInst && hInst.learningTrials) ? hInst.learningTrials : [];
            if (trials.length < 2) {
                ctx.fillStyle = '#64748b';
                ctx.fillText('Accumulating trial escape times...', 24, h / 2 + 4);
                return;
            }
            ctx.strokeStyle = '#fbbf24';
            ctx.lineWidth = 1.8;
            ctx.beginPath();
            trials.forEach((t, i) => {
                const x = 20 + (i / (trials.length - 1)) * (w - 40);
                const y = h - 12 - Math.min(h - 26, (t.value / 30.0) * (h - 26));
                if (i === 0) ctx.moveTo(x, y); else ctx.lineTo(x, y);
            });
            ctx.stroke();
        }
    },
    'buridan': {
        title: 'Buridan Landmark Fixation & Centrophobism',
        badge: 'GÖTZ (1980)',
        ref: 'Götz (1980) / Strauss (1997) Stripe Fixation & Water Moat',
        sliders: [
            { key: 'stripeWidth', label: 'Stripe Angular Width', min: 5, max: 30, step: 1, val: 12, unit: ' °', desc: 'Angular visual width of opposing high-contrast vertical black stripes on illuminated arena wall.', apply: (a, v) => { if (a.paradigmState) a.paradigmState.stripeWidth = v; } },
            { key: 'moatRepulsion', label: 'Moat Barrier Repulsion', min: 0.5, max: 3.0, step: 0.2, val: 1.5, unit: ' x', desc: 'Aversive water moat boundary repulsion force preventing fly from tumbling into surrounding liquid.', apply: (a, v) => { if (a.paradigmState) a.paradigmState.moatRepulsion = v; } }
        ],
        actions: [
            { label: 'Invert Contrast (Dark/Light)', class: 'primary', handler: (a, h) => { if (a.paradigmState) a.paradigmState.inverted = !a.paradigmState.inverted; } },
            { label: 'Rotate Stripes (90°)', class: '', handler: (a, h) => { if (a.paradigmState) a.paradigmState.stripeAngle = (a.paradigmState.stripeAngle || 0) + Math.PI / 2; } },
            { label: 'Reset Platform Transits', class: '', handler: (a, h) => { if (a.paradigmState) a.paradigmState.stripeCrossings = 0; } }
        ],
        metrics: [
            { label: 'Stripe Fixation Index', get: (a) => `${(a.paradigmState && Number.isFinite(a.paradigmState.meanFixation) ? a.paradigmState.meanFixation : 0).toFixed(2)}` },
            { label: 'Platform Transitions', get: (a) => `${a.paradigmState && a.paradigmState.stripeCrossings !== undefined ? a.paradigmState.stripeCrossings : 0}` },
            { label: 'Centrophobism Ratio', get: (a) => `${((a.paradigmState && Number.isFinite(a.paradigmState.centrophobism) ? a.paradigmState.centrophobism : 1.0) * 100).toFixed(1)}%` }
        ],
        drawChart: (ctx, w, h, a, hInst) => {
            ctx.fillStyle = '#38bdf8';
            ctx.font = '9px monospace';
            ctx.fillText('Stripe Heading Polar Distribution (0° & 180°)', 8, 14);
            const cx = w / 2, cy = h / 2 + 6, r = 38;
            ctx.strokeStyle = 'rgba(255,255,255,0.15)';
            ctx.beginPath(); ctx.arc(cx, cy, r, 0, Math.PI * 2); ctx.stroke();
            ctx.fillStyle = '#ffffff';
            ctx.fillRect(cx - r - 6, cy - 8, 4, 16);
            ctx.fillRect(cx + r + 2, cy - 8, 4, 16);
            ctx.strokeStyle = '#38bdf8';
            ctx.lineWidth = 2.0;
            ctx.beginPath(); ctx.moveTo(cx, cy);
            ctx.lineTo(cx + Math.cos(a.fly.heading) * (r - 4), cy - Math.sin(a.fly.heading) * (r - 4));
            ctx.stroke();
        }
    },
    'visual-operant': {
        title: 'Operant Flight Simulator (Drum & Laser)',
        badge: 'WOLF & HEISENBERG (1991)',
        ref: 'Wolf & Heisenberg (1991) J. Comp. Physiol. Operant Conditioning',
        sliders: [
            { key: 'laserPower', label: 'Laser Punishment Power', min: 10, max: 100, step: 5, val: 50, unit: ' mW', desc: 'Infrared heating laser punishment power (mW) aimed at tethered thorax when facing conditioned pattern.', apply: (a, v) => { if (a.paradigmState) a.paradigmState.laserPower = v; } },
            { key: 'drumFriction', label: 'Virtual Yaw Inertia', min: 0.5, max: 2.5, step: 0.1, val: 1.0, unit: ' x', desc: 'Virtual yaw inertia / rotational damping of surrounding 360° visual pattern drum.', apply: (a, v) => { if (a.paradigmState) a.paradigmState.torqueGain = v; } }
        ],
        actions: [
            { label: 'Invert Heat Sectors (Reversal)', class: 'primary', handler: (a, h) => { if (a.paradigmState) a.paradigmState.invertSectors = !a.paradigmState.invertSectors; } },
            { label: 'Laser Beam Pulse', class: 'danger', handler: (a, h) => { a.mb.stepPlasticity(0.0, 1.0, 4.0); } },
            { label: 'Reset Quadrant Timers', class: '', handler: (a, h) => { if (a.paradigmState) { a.paradigmState.timeSafeMs = 0; a.paradigmState.timePunishedMs = 0; } } }
        ],
        metrics: [
            { label: 'Operant PI', get: (a) => `${(a.paradigmState && Number.isFinite(a.paradigmState.learningIndex) ? a.paradigmState.learningIndex : 0).toFixed(2)}` },
            { label: 'Current Sector', get: (a) => `${a.paradigmState && a.paradigmState.laserActive ? 'PUNISHED (LASER ON)' : 'SAFE SECTOR'}` },
            { label: 'Laser Cumulative', get: (a) => `${((a.paradigmState && a.paradigmState.timePunishedMs) ? a.paradigmState.timePunishedMs / 1000 : 0.0).toFixed(1)}s` }
        ],
        drawChart: (ctx, w, h, a, hInst) => {
            ctx.fillStyle = '#f43f5e';
            ctx.font = '9px monospace';
            ctx.fillText('Operant Quadrants (4 Sectors: Safe vs Laser)', 8, 14);
            const cx = w / 2, cy = h / 2 + 6, r = 36;
            ctx.fillStyle = 'rgba(244, 63, 94, 0.3)';
            ctx.beginPath(); ctx.moveTo(cx, cy); ctx.arc(cx, cy, r, 0, Math.PI / 2); ctx.fill();
            ctx.beginPath(); ctx.moveTo(cx, cy); ctx.arc(cx, cy, r, Math.PI, 3 * Math.PI / 2); ctx.fill();
            ctx.fillStyle = 'rgba(56, 189, 248, 0.3)';
            ctx.beginPath(); ctx.moveTo(cx, cy); ctx.arc(cx, cy, r, Math.PI / 2, Math.PI); ctx.fill();
            ctx.beginPath(); ctx.moveTo(cx, cy); ctx.arc(cx, cy, r, 3 * Math.PI / 2, 2 * Math.PI); ctx.fill();
            ctx.strokeStyle = '#ffffff'; ctx.strokeRect(cx - 2, cy - 2, 4, 4);
        }
    },
    'wind-tunnel': {
        title: 'Anemotaxic Plume Tracking (Surge & Cast)',
        badge: 'ALVAREZ-SALVADO & DEMIR',
        ref: 'Alvarez-Salvado (2018) / Demir (2020) Odor Plume Navigation',
        sliders: [
            { key: 'windVelocity', label: 'Laminar Airflow Speed', min: 5, max: 40, step: 1, val: 18, unit: ' cm/s', desc: 'Laminar carrier airflow speed channeling upstream odor plume pulses toward downwind fly.', apply: (a, v) => { a.windVector[0] = -v; } },
            { key: 'plumeWidth', label: 'Gaussian Plume Width', min: 6, max: 30, step: 1, val: 14, unit: ' mm', desc: 'Gaussian width of intermittent odor plume filaments dictating surge vs casting transitions.', apply: (a, v) => { if (a.paradigmState) a.paradigmState.filamentSigma = v / 4.0; } }
        ],
        actions: [
            { label: 'Shift Plume Source', class: 'primary', handler: (a, h) => { if (a.paradigmState) a.paradigmState.nozzlePos[1] = 30.0 + (Math.random() - 0.5) * 30.0; } },
            { label: 'Turbulent Crosswind Gust', class: 'danger', handler: (a, h) => { a.windVector[1] = (Math.random() - 0.5) * 30.0; setTimeout(() => { a.windVector[1] = 0.0; }, 2500); } },
            { label: 'Reset Surge/Cast Filters', class: '', handler: (a, h) => { if (a.paradigmState) { a.paradigmState.surgeSteps = 0; a.paradigmState.castSteps = 0; } } }
        ],
        metrics: [
            { label: 'Surge/Cast Ratio', get: (a) => { const p = a.paradigmState || {}; return p.castSteps > 0 ? `${(p.surgeSteps / p.castSteps).toFixed(2)}x` : '0.00x'; } },
            { label: 'Upwind Progress', get: (a) => `${(a.paradigmState && Number.isFinite(a.paradigmState.upwindProgress) ? a.paradigmState.upwindProgress : 0).toFixed(1)} mm` },
            { label: 'Antenna Wind Deflection', get: (a) => `${(Math.hypot(a.windVector[0], a.windVector[1]) * 0.12).toFixed(1)} μN` }
        ],
        drawChart: (ctx, w, h, a, hInst) => {
            ctx.fillStyle = '#38bdf8';
            ctx.font = '9px monospace';
            ctx.fillText('Surge (Upwind) vs Cast (Crosswind) Vectors', 8, 14);
            const cx = w / 2, cy = h / 2 + 6;
            ctx.strokeStyle = '#0284c7'; ctx.lineWidth = 1.5;
            ctx.beginPath(); ctx.moveTo(cx - 50, cy); ctx.lineTo(cx + 50, cy); ctx.stroke();
            ctx.beginPath(); ctx.moveTo(cx, cy - 30); ctx.lineTo(cx, cy + 30); ctx.stroke();
            ctx.fillStyle = '#4ade80';
            ctx.fillRect(cx - 35, cy - 8, 16, 16);
            ctx.fillStyle = '#ffffff';
            ctx.fillText('Upwind Surge', cx - 48, cy + 24);
        }
    },
    'looming-escape': {
        title: 'Predator Looming Escape (Giant Fiber)',
        badge: 'CARD & DICKINSON (2008)',
        ref: 'Card & Dickinson (2008) PNAS Looming Visual Escape & Takeoff',
        sliders: [
            { key: 'lvRatio', label: 'Looming Size/Speed (l/v ratio)', min: 10, max: 100, step: 5, val: 40, unit: ' ms', desc: 'Size-to-approach speed ratio (l/v in ms). Smaller values model high-velocity predatory strikes.', apply: (a, v) => { if (a.paradigmState) a.paradigmState.lv = v; } },
            { key: 'gfThreshold', label: 'Giant Fiber Spike Threshold', min: 0.4, max: 0.95, step: 0.05, val: 0.65, unit: ' Vm', desc: 'Giant Fiber axon threshold membrane potential triggering all-or-none escape jump takeoff.', apply: (a, v) => { if (a.paradigmState) a.paradigmState.gfThresh = v; } }
        ],
        actions: [
            { label: 'Trigger Looming Disc', class: 'danger', handler: (a, h) => {
                a.dn.escapeActive = true;
                a.fly.speed = 35.0;
                if (h.daemonBridge) h.daemonBridge.sendCommand('inject_stimulus', { type: 'gf_looming', value: 1.0 });
            } },
            { label: 'Direct Giant Fiber Spike', class: 'primary', handler: (a, h) => { a.dn.escapeActive = true; } },
            { label: 'Reset Escape Posture', class: '', handler: (a, h) => { a.dn.escapeActive = false; } }
        ],
        metrics: [
            { label: 'GF Depolarization', get: (a) => `${a.dn.escapeActive ? '100% [SPIKE]' : '12% [SUBTHRESHOLD]'}` },
            { label: 'Escape Jump Angle', get: (a) => `${((a.fly.heading * 180) / Math.PI).toFixed(0)}°` },
            { label: 'Takeoff Velocity', get: (a) => `${a.fly.speed.toFixed(1)} mm/s` }
        ],
        drawChart: (ctx, w, h, a, hInst) => {
            ctx.fillStyle = '#f43f5e';
            ctx.font = '9px monospace';
            ctx.fillText('Angular Subtense theta(t) & GF Spike Threshold', 8, 14);
            ctx.strokeStyle = '#f43f5e'; ctx.lineWidth = 2.0;
            ctx.beginPath();
            for (let i = 0; i < 60; i++) {
                const x = 20 + (i / 60) * (w - 40);
                const theta = Math.tan((i / 60) * 1.4);
                const y = h - 10 - Math.min(h - 24, theta * 25);
                if (i === 0) ctx.moveTo(x, y); else ctx.lineTo(x, y);
            }
            ctx.stroke();
            ctx.strokeStyle = '#fbbf24'; ctx.setLineDash([3, 3]);
            ctx.beginPath(); ctx.moveTo(20, h / 2 - 5); ctx.lineTo(w - 20, h / 2 - 5); ctx.stroke();
            ctx.setLineDash([]);
        }
    },
    'optomotor': {
        title: 'Optomotor Gaze Stabilization & Saccades',
        badge: 'GÖTZ (1964) / KIM (2017)',
        ref: 'Götz (1964) Kybernetik / Kim et al. (Cell 2017) Saccadic Efference Copy',
        sliders: [
            { key: 'patternSpeed', label: 'Grating Velocity (omega)', min: -120, max: 120, step: 10, val: 30, unit: ' °/s', desc: 'Angular velocity of surrounding high-contrast vertical grating drum driving wide-field optic flow.', apply: (a, v) => { if (a.paradigmState) a.paradigmState.drumVelocityDegS = v; } },
            { key: 'spatialPeriod', label: 'Grating Spatial Wavelength', min: 10, max: 60, step: 5, val: 30, unit: ' °', desc: 'Grating angular wavelength. Shorter periods test spatial resolution limits of compound eyes.', apply: (a, v) => { if (a.paradigmState) a.paradigmState.wavelength = v; } }
        ],
        actions: [
            { label: 'Invert Grating Direction', class: 'primary', handler: (a, h) => { if (a.paradigmState) a.paradigmState.drumVelocityDegS = -(a.paradigmState.drumVelocityDegS || 30); } },
            { label: 'Toggle Efference Copy', class: '', handler: (a, h) => { if (a.paradigmState) a.paradigmState.efferenceCopy = !a.paradigmState.efferenceCopy; } },
            { label: 'Zero Contrast (Uniform)', class: '', handler: (a, h) => { if (a.paradigmState) a.paradigmState.contrast = (a.paradigmState.contrast === 0 ? 0.9 : 0); } }
        ],
        metrics: [
            { label: 'Optomotor Yaw Torque', get: (a) => `${(a.dn.dna02Diff).toFixed(2)} rad/s` },
            { label: 'Retinal Slip Rate', get: (a) => `${(a.paradigmState && Number.isFinite(a.paradigmState.effectiveSlip) ? a.paradigmState.effectiveSlip : 0).toFixed(1)} °/s` },
            { label: 'Closed-Loop Gain', get: (a) => `${(a.paradigmState && Number.isFinite(a.paradigmState.gain) ? a.paradigmState.gain : 0.88).toFixed(2)}` }
        ],
        drawChart: (ctx, w, h, a, hInst) => {
            ctx.fillStyle = '#38bdf8';
            ctx.font = '9px monospace';
            ctx.fillText('Optomotor Yaw Response Curve vs Grating Speed', 8, 14);
            const cx = w / 2, cy = h / 2 + 6;
            ctx.strokeStyle = 'rgba(255,255,255,0.1)';
            ctx.beginPath(); ctx.moveTo(20, cy); ctx.lineTo(w - 20, cy); ctx.stroke();
            ctx.strokeStyle = '#38bdf8'; ctx.lineWidth = 2.0;
            ctx.beginPath();
            for (let i = 0; i < 50; i++) {
                const norm = (i - 25) / 25;
                const resp = Math.tanh(norm * 2.0);
                const x = cx + norm * (w / 2 - 30);
                const y = cy - resp * (h / 2 - 18);
                if (i === 0) ctx.moveTo(x, y); else ctx.lineTo(x, y);
            }
            ctx.stroke();
        }
    },
    'gap-crossing': {
        title: 'Spatial Planning & Gap Crossing',
        badge: 'PICK & STRAUSS (2005)',
        ref: 'Pick & Strauss (Nature 2005) / Triphan (2010) Gap Crossing Spatial Planning',
        sliders: [
            { key: 'gapWidth', label: 'Chasm Void Width', min: 1.5, max: 5.0, step: 0.25, val: 3.5, unit: ' mm', desc: 'Physical chasm width. Fly will probe with forelegs and attempt crossing if <= 3.8mm.', apply: (a, v) => { if (a.paradigmState) a.paradigmState.gapWidthMm = v; } },
            { key: 'clawAdhesion', label: 'Tarsal Claw Friction', min: 0.5, max: 2.0, step: 0.1, val: 1.2, unit: ' x', desc: 'Tarsal claw cuticular friction coefficient against runway substrate edge.', apply: (a, v) => { if (a.paradigmState) a.paradigmState.friction = v; } }
        ],
        actions: [
            { label: 'Extend Foreleg Probe', class: 'primary', handler: (a, h) => { if (a.cpg) a.cpg.steppingFreq = 4.0; } },
            { label: 'Widen Gap (+0.5mm)', class: '', handler: (a, h) => { if (a.paradigmState) a.paradigmState.gapWidthMm = Math.min(5.0, (a.paradigmState.gapWidthMm || 3.5) + 0.5); } },
            { label: 'Narrow Gap (-0.5mm)', class: '', handler: (a, h) => { if (a.paradigmState) a.paradigmState.gapWidthMm = Math.max(1.5, (a.paradigmState.gapWidthMm || 3.5) - 0.5); } }
        ],
        metrics: [
            { label: 'Crossing Outcome', get: (a) => { const p = a.paradigmState || {}; return p.crossingSuccess ? 'CROSSED' : (p.decisionOutcome || (p.isProbing ? 'PROBING' : 'APPROACH')); } },
            { label: 'Max Foreleg Reach', get: (a) => `${(a.paradigmState && a.paradigmState.reachabilityThreshMm) || 3.8} mm` },
            { label: 'Tactile Probe Time', get: (a) => `${((a.paradigmState && a.paradigmState.probingDurationMs) || 0).toFixed(0)} ms` }
        ],
        drawChart: (ctx, w, h, a, hInst) => {
            ctx.fillStyle = '#4ade80';
            ctx.font = '9px monospace';
            ctx.fillText('Psychometric Curve: Crossing Probability vs Gap (mm)', 8, 14);
            ctx.strokeStyle = '#4ade80'; ctx.lineWidth = 2.0;
            ctx.beginPath();
            for (let i = 0; i < 40; i++) {
                const x = 30 + (i / 40) * (w - 60);
                const gap = 1.5 + (i / 40) * 3.5;
                const p = 1.0 / (1.0 + Math.exp((gap - 3.4) * 3.0));
                const y = h - 12 - p * (h - 28);
                if (i === 0) ctx.moveTo(x, y); else ctx.lineTo(x, y);
            }
            ctx.stroke();
        }
    },
    'circadian-dam': {
        title: 'DAM Locomotor Sleep/Wake Monitor',
        badge: 'KONOPKA & ALLADA',
        ref: 'Konopka & Benzer (1971) / Allada (2010) DAM Sleep & Circadian Biology',
        sliders: [
            { key: 'incubatorTemp', label: 'Incubator Temperature', min: 18, max: 32, step: 1, val: 25, unit: ' °C', desc: 'Incubator ambient temperature (°C) modulating TRPA1 / circadian clock cycling speed.', apply: (a, v) => { if (a.paradigmState) a.paradigmState.temp = v; } },
            { key: 'circadianSpeed', label: 'Time Dilation (1h = sec)', min: 1, max: 60, step: 5, val: 12, unit: ' x', desc: 'Time acceleration factor (1 hour = N seconds) for rapid multi-day sleep/wake analysis.', apply: (a, v) => { if (a.paradigmState) a.paradigmState.dilation = v; } }
        ],
        actions: [
            { label: 'Toggle Light/Dark Phase', class: 'primary', handler: (a, h) => { if (a.paradigmState) a.paradigmState.isDark = !a.paradigmState.isDark; } },
            { label: 'Trigger Arousal Tap', class: 'danger', handler: (a, h) => { a.fly.speed = 12.0; } },
            { label: 'Clear DAM Activity Logs', class: '', handler: (a, h) => { if (a.paradigmState) { a.paradigmState.beamCrossings = 0; a.paradigmState.totalSleepMin = 0; a.paradigmState.sleepBouts = 0; } } }
        ],
        metrics: [
            // The DAM assay runs at 1 sim second = 2 fly-minutes (see step()).
            { label: 'Zeitgeber Time (ZT)', get: (a) => { const mins = a.paradigmElapsedSec * 2.0; const hh = Math.floor((mins / 60) % 24); const mm = Math.floor(mins % 60); return `ZT ${String(hh).padStart(2, '0')}:${String(mm).padStart(2, '0')}`; } },
            { label: 'Sleep Fraction', get: (a) => { const p = a.paradigmState || {}; const mins = Math.max(1, a.paradigmElapsedSec * 2.0); return `${(100 * (p.totalSleepMin || 0) / mins).toFixed(1)}%`; } },
            { label: 'Beam Crossings', get: (a) => `${(a.paradigmState && a.paradigmState.beamCrossings) || 0}` }
        ],
        drawChart: (ctx, w, h, a, hInst) => {
            ctx.fillStyle = '#fbbf24';
            ctx.font = '9px monospace';
            ctx.fillText('24-Hour Binned Double-Plotted Actogram', 8, 14);
            const barW = (w - 30) / 24;
            for (let i = 0; i < 24; i++) {
                const act = Math.sin((i / 24) * Math.PI * 2 - Math.PI / 2) * 0.5 + 0.5;
                const bh = act * (h - 28);
                ctx.fillStyle = i < 12 ? 'rgba(251, 191, 36, 0.7)' : 'rgba(99, 102, 241, 0.7)';
                ctx.fillRect(15 + i * barW, h - 10 - bh, barW - 2, bh);
            }
        }
    },
    'courtship': {
        title: 'Courtship Conditioning & Song',
        badge: 'SIEGEL & HALL (1979)',
        ref: 'Siegel & Hall (1979) PNAS / Keleman (Nature 2007) Courtship Plasticity',
        sliders: [
            { key: 'cvaLevel', label: 'cVA Pheromone Level', min: 0.0, max: 1.0, step: 0.05, val: 0.65, unit: ' cVA', desc: 'Anti-aphrodisiac cis-vaccenyl acetate (cVA) pheromone concentration driving courtship suppression.', apply: (a, v) => { if (a.paradigmState) a.paradigmState.cva = v; } },
            { key: 'femaleSpeed', label: 'Target Female Walking Speed', min: 0, max: 15, step: 1, val: 5, unit: ' mm/s', desc: 'Target decoy/female locomotion speed in circular courtship observation chamber.', apply: (a, v) => { if (a.paradigmState) a.paradigmState.femaleSpeed = v; } }
        ],
        actions: [
            { label: 'Trigger Wing Vibration (Song)', class: 'primary', handler: (a, h) => { a.mb.stepPlasticity(1.0, 0.0, 3.0); } },
            { label: 'Toggle Female Receptivity', class: '', handler: (a, h) => { if (a.paradigmState) a.paradigmState.receptive = !a.paradigmState.receptive; } },
            { label: 'Reset Courtship Suppression', class: '', handler: (a, h) => { a.mb.reset(false); } }
        ],
        metrics: [
            { label: 'Courtship Index (CI)', get: (a) => `${(a.paradigmState && Number.isFinite(a.paradigmState.courtshipIndex) ? a.paradigmState.courtshipIndex : 0).toFixed(2)}` },
            { label: 'Wing Extension', get: (a) => `${((a.paradigmState && a.paradigmState.wingAngleDeg) || 0).toFixed(0)}°` },
            { label: 'Target Proximity', get: (a) => { const p = a.paradigmState; return p && p.femalePos ? `${Math.hypot(a.fly.x - p.femalePos[0], a.fly.y - p.femalePos[1]).toFixed(1)} mm` : '--'; } }
        ],
        drawChart: (ctx, w, h, a, hInst) => {
            ctx.fillStyle = '#f43f5e';
            ctx.font = '9px monospace';
            ctx.fillText('Courtship Index CI Decay Curve (Suppression)', 8, 14);
            ctx.strokeStyle = '#f43f5e'; ctx.lineWidth = 2.0;
            ctx.beginPath();
            for (let i = 0; i < 40; i++) {
                const x = 25 + (i / 40) * (w - 50);
                const ci = 0.85 * Math.exp(-i / 14.0) + 0.15;
                const y = h - 12 - ci * (h - 26);
                if (i === 0) ctx.moveTo(x, y); else ctx.lineTo(x, y);
            }
            ctx.stroke();
        }
    },
    'labyrinth': {
        title: 'Multi-Junction Obstacle Labyrinth',
        badge: 'SPATIAL DECISION NETWORK',
        ref: 'Continuous Sliding Collision Physics & Multi-Junction Maze',
        sliders: [
            { key: 'wallFriction', label: 'Wall Coulomb Friction', min: 0.0, max: 0.8, step: 0.05, val: 0.5, unit: '', desc: 'Coulomb crawling friction coefficient along corridor walls during sliding contacts.', apply: (a, v) => { (a.currentWalls || []).forEach(w => { w.friction = v; }); } },
            { key: 'exitOdor', label: 'Goal Exit Odor Emission', min: 0.5, max: 3.0, step: 0.2, val: 1.8, unit: ' x', desc: 'Sucrose food volatile emission strength emanating from terminal goal chamber.', apply: (a, v) => { if (a.paradigmState) a.paradigmState.exitOdor = v; } }
        ],
        actions: [
            { label: 'Bait Exit Chamber', class: 'primary', handler: (a, h) => { a.spawnFoodNearFly(); } },
            { label: 'Return to Maze Start', class: 'danger', handler: (a, h) => { a.resetTrial(false, true); } },
            { label: 'Clear Maze Trail', class: '', handler: (a, h) => { a.fly.trail = []; } }
        ],
        metrics: [
            { label: 'Wall Contacts', get: (a) => `${(a.paradigmState && a.paradigmState.wallCollisions) || 0}` },
            { label: 'Path Traversed', get: (a) => `${((a.paradigmState && a.paradigmState.pathLength) || 0).toFixed(1)} mm` },
            { label: 'Goal Reached', get: (a) => `${a.paradigmState && a.paradigmState.goalReached ? 'YES' : 'NAVIGATING'}` }
        ],
        drawChart: (ctx, w, h, a, hInst) => {
            ctx.fillStyle = '#38bdf8';
            ctx.font = '9px monospace';
            ctx.fillText('Path Efficiency vs Optimal Euclidean Path', 8, 14);
            const tort = (a.paradigmState && a.paradigmState.tortuosity) || 1.0;
            const eff = Math.max(0.0, Math.min(1.0, 1.0 / tort));
            ctx.fillStyle = '#0284c7';
            ctx.fillRect(25, h / 2 - 10, (w - 50) * eff, 20);
            ctx.strokeStyle = '#38bdf8';
            ctx.strokeRect(25, h / 2 - 10, w - 50, 20);
            ctx.fillStyle = '#ffffff';
            ctx.fillText(`${(eff * 100).toFixed(1)}% Optimal Routing (tortuosity ${tort.toFixed(2)})`, 32, h / 2 + 4);
        }
    },
    'multisensory-sandbox': {
        title: 'Whole-Body Embodied Multisensory Benchmark',
        badge: 'INTEGRATED BENCHMARK',
        ref: 'Simultaneous Visual, Thermal, Olfactory, Wind & Articulated Kinematics',
        sliders: [
            { key: 'cpgCadence', label: 'Kuramoto CPG Base Cadence', min: 3.0, max: 14.0, step: 0.5, val: 8.0, unit: ' Hz', desc: 'Kuramoto tripod gait base stepping frequency coordinating 6 articulated limb phases.', apply: (a, v) => { a.cpg.baseFreq = v; } },
            { key: 'wallRepulsion', label: 'Boundary Repulsion', min: 0.2, max: 3.0, step: 0.1, val: 1.0, unit: 'x', desc: 'Gain of the anticipatory wall-avoidance steering (antennal proximity whiskers) that turns the fly away from walls and pillars.', apply: (a, v) => { a.wallRepulsion = v; } }
        ],
        actions: [
            { label: 'Optogenetic DNa02 Turn', class: 'primary', handler: (a, h) => { a.fly.heading += 0.4; if (h.daemonBridge) h.daemonBridge.sendCommand('inject_stimulus', { type: 'optogenetic_dna02', value: 0.4 }); } },
            { label: 'Trigger Thermal Flash', class: 'danger', handler: (a, h) => { a.mb.stepPlasticity(0, 1, 5); if (h.daemonBridge) h.daemonBridge.sendCommand('inject_stimulus', { type: 'thermal_flash', value: 40.0 }); } },
            { label: 'Sugar Odor Puff', class: 'primary', handler: (a, h) => { a.mb.stepPlasticity(1, 0, 4); if (h.daemonBridge) h.daemonBridge.sendCommand('inject_stimulus', { type: 'odor_puff', value: 1.0 }); } }
        ],
        metrics: [
            { label: 'Composite Score', get: (a) => `${(a.paradigmState && Number.isFinite(a.paradigmState.compositeScore) ? a.paradigmState.compositeScore : 0).toFixed(1)} / 100` },
            { label: 'Tripod Coordination', get: (a) => `${((a.paradigmState && a.paradigmState.coordinationScore) || 0) * 100 | 0}%` },
            { label: 'Sensory Alignment', get: (a) => `${(((a.paradigmState && a.paradigmState.sensoryIntegrationScore) || 0) * 100).toFixed(1)}%` }
        ],
        drawChart: (ctx, w, h, a, hInst) => {
            ctx.fillStyle = '#38bdf8';
            ctx.font = '9px monospace';
            ctx.fillText('4-Quadrant Benchmark Radar (Coord / Sens / Effic / Smooth)', 8, 14);
            const cx = w / 2, cy = h / 2 + 8, r = 36;
            ctx.strokeStyle = 'rgba(255,255,255,0.15)';
            ctx.beginPath();
            ctx.moveTo(cx - r, cy); ctx.lineTo(cx + r, cy);
            ctx.moveTo(cx, cy - r); ctx.lineTo(cx, cy + r);
            ctx.stroke();
            ctx.fillStyle = 'rgba(56, 189, 248, 0.35)';
            ctx.strokeStyle = '#38bdf8';
            ctx.lineWidth = 1.5;
            ctx.beginPath();
            ctx.moveTo(cx, cy - r * 0.95);
            ctx.lineTo(cx + r * 0.90, cy);
            ctx.lineTo(cx, cy + r * 0.88);
            ctx.lineTo(cx - r * 0.92, cy);
            ctx.closePath();
            ctx.fill();
            ctx.stroke();
        }
    }
};


// =============================================================================
// 8. MODERN SCIENTIFIC HUD & DIRECTOR'S DASHBOARD
// =============================================================================

const LESION_INFO = {
    'WT': {
        name: 'Wild-Type Control (Canton-S / w1118)',
        driver: 'Intact Baseline Genotype',
        mechanism: 'Compact modular model: 120 Kenyon cells, a heading compass, locomotion and sensory-response modules. The downloaded whole connectome runs separately.',
        expectedDeficit: 'No components disabled. Performance must be measured per assay; no target score is assumed.',
        color: '#4ade80'
    },
    'DELTA_MB': {
        name: 'ΔMB Kenyon Cell Silencing',
        driver: 'MB247-GAL4 > UAS-TNT / rutabaga / dunce',
        mechanism: 'Genetic ablation of ~4,000 Mushroom Body Kenyon Cells and MBONs. Completely abolishes anti-Hebbian dopamine (PPL1/PAM) synaptic plasticity.',
        expectedDeficit: 'Total loss of associative olfactory & thermal learning: Fails odor avoidance conditioning (PI ~ 0.00 in T-maze), fails heat-maze place learning, fails courtship memory.',
        color: '#fda4af'
    },
    'DELTA_CX': {
        name: 'ΔCX Central Complex Compass Knockout',
        driver: 'R60D05-GAL4 > UAS-TNT / ccd (central complex deranged)',
        mechanism: 'Silences 16-wedge E-PG ring attractor compass neurons and PFL3 steering decoders, destroying the fly\'s internal allocentric heading coordinate frame.',
        expectedDeficit: 'Loss of spatial orientation & landmark navigation: Cannot fixate visual stripes (fails Buridan), cannot triangulate cool refuge in heat-maze, wanders aimlessly.',
        color: '#f43f5e'
    },
    'DELTA_GF': {
        name: 'ΔGF Giant Fiber Looming Ablation',
        driver: 'R68A06-GAL4 > UAS-shi[ts] / Passover / shakB',
        mechanism: 'Silences descending Giant Fiber pair (DNp01/GF) that integrates looming optical expansion from lobula columnar neurons LPLC2 and Col4.',
        expectedDeficit: 'Blind to approaching predatory shadows: Fails to trigger rapid 5ms tergotrochanteral motor jump takeoff, resulting in 100% predatory strike capture.',
        color: '#fb923c'
    },
    'DELTA_JO': {
        name: 'ΔJO Johnston’s Organ Mechanosensory Knockout',
        driver: 'tilB (touch-insensitive-larva-B) / nompA / JO-GAL4',
        mechanism: 'Silences Johnston’s organ chordotonal neurons in the second antennal segment (pedicel), abolishing wind drag and acoustic vibration transduction.',
        expectedDeficit: 'Loss of wind anemotaxis and courtship hearing: Fails upwind surge-and-cast flight in wind tunnel, fails courtship song recognition, disorients in airflow.',
        color: '#a78bfa'
    },
    'DELTA_OFF': {
        name: 'ΔOFF T5 Motion Detector Silencing',
        driver: 'T5-split-GAL4 > UAS-Kir2.1 / dark-edge motion blind',
        mechanism: 'Silences T5 columnar neurons in the optic lobe medulla/lobula, which compute elementary motion detection for moving dark edges (OFF pathway).',
        expectedDeficit: 'Loss of dark-edge optomotor gaze stabilization: Fly fails to compensate for rotating dark stripes, causing severe heading drift during visual flow.',
        color: '#38bdf8'
    }
};

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

        // Initialize with default paradigm guide & specialized assay tools
        this.updateActiveCard('open-arena');
        this.renderExperimentGuide('open-arena');
        this.renderAssayTools('open-arena');

        // Connect to continuous learning cluster daemon (Ryzen / local)
        this.daemonBridge = new DaemonBridgeClient(this.arena, this);
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

        if (this.daemonBridge && this.daemonBridge.connected) {
            this.daemonBridge.sendCommand('set_speed', { speed: this.simSpeed });
        }
    }

    setupCatalogEvents() {
        const cards = document.querySelectorAll('.experiment-card');
        cards.forEach(card => {
            card.addEventListener('click', () => {
                const pid = card.dataset.paradigm;
                if (!pid) return;
                this.selectParadigm(pid);
            });
        });
    }

    selectParadigm(pid) {
        if (this.daemonBridge?.connected || this.arena.remoteDriven || this.arena.awaitingDaemon) {
            return this.daemonBridge?.requestParadigmSwitch(pid);
        }
        this.arena.initParadigm(pid);
        this.updateActiveCard(pid);
        this.renderExperimentGuide(pid);
        this.renderAssayTools(pid);
        if (pid === 'multisensory-sandbox') document.getElementById('tabLimbDeck')?.click();
        const badge = document.getElementById('navbarParadigmBadge');
        if (badge) badge.textContent = pid.toUpperCase().replace(/-/g, ' ');
    }

    setupDeckTabs() {
        const tabGuide = document.getElementById('tabGuide');
        const tabAssayTools = document.getElementById('tabAssayTools');
        const tabLimbDeck = document.getElementById('tabLimbDeck');
        const tabTraining = document.getElementById('tabTraining');
        const trainingPanel = document.getElementById('trainingPanel');
        const guideContent = document.getElementById('guideTabContent');
        const assayToolsPanel = document.getElementById('assayToolsPanel');
        const limbPanel = document.getElementById('limbDeckPanel');

        const selectTab = (activeTab) => {
            if (tabGuide) tabGuide.classList.toggle('active', tabGuide === activeTab);
            if (tabAssayTools) tabAssayTools.classList.toggle('active', tabAssayTools === activeTab);
            if (tabLimbDeck) tabLimbDeck.classList.toggle('active', tabLimbDeck === activeTab);
            if (tabTraining) tabTraining.classList.toggle('active', tabTraining === activeTab);
            if (trainingPanel) trainingPanel.style.display = activeTab === tabTraining ? 'flex' : 'none';

            if (guideContent) guideContent.style.display = (activeTab === tabGuide) ? 'block' : 'none';
            if (assayToolsPanel) {
                assayToolsPanel.style.display = (activeTab === tabAssayTools) ? 'flex' : 'none';
                if (activeTab === tabAssayTools && this.activeAssayUpdater) {
                    this.activeAssayUpdater();
                }
            }
            if (limbPanel) limbPanel.style.display = (activeTab === tabLimbDeck) ? 'flex' : 'none';
        };

        if (tabGuide) tabGuide.addEventListener('click', () => selectTab(tabGuide));
        if (tabAssayTools) tabAssayTools.addEventListener('click', () => selectTab(tabAssayTools));
        if (tabLimbDeck) tabLimbDeck.addEventListener('click', () => selectTab(tabLimbDeck));
        if (tabTraining) tabTraining.addEventListener('click', () => selectTab(tabTraining));
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
                        ${p.desc ? `<div class="slider-desc" style="font-size:8.5px; color:#94a3b8; margin-top:3px; line-height:1.25;">${p.desc}</div>` : ''}
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

        const updateLesionCard = (type) => {
            const info = LESION_INFO[type] || LESION_INFO['WT'];
            const nameEl = document.getElementById('lesionCardName');
            const driverEl = document.getElementById('lesionCardDriver');
            const descEl = document.getElementById('lesionCardDesc');
            const deficitEl = document.getElementById('lesionCardDeficit');

            if (nameEl) {
                nameEl.textContent = info.name;
                nameEl.style.color = info.color;
            }
            if (driverEl) driverEl.textContent = info.driver;
            if (descEl) descEl.textContent = info.mechanism;
            if (deficitEl) deficitEl.textContent = info.expectedDeficit;
        };

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
                    updateLesionCard(item.type);
                });
            }
        });

        // Initialize with WT info
        updateLesionCard('WT');

        // Science Guide Modal controls
        const btnOpen = document.getElementById('btnOpenScienceGuide');
        const btnClose = document.getElementById('btnCloseScienceGuide');
        const modal = document.getElementById('scienceGuideModal');

        if (btnOpen && modal) {
            btnOpen.addEventListener('click', () => {
                modal.style.display = 'flex';
            });
        }
        if (btnClose && modal) {
            btnClose.addEventListener('click', () => {
                modal.style.display = 'none';
            });
        }
        if (modal) {
            modal.addEventListener('click', (e) => {
                if (e.target === modal) modal.style.display = 'none';
            });
            document.addEventListener('keydown', (e) => {
                if (e.key === 'Escape' && modal.style.display === 'flex') {
                    modal.style.display = 'none';
                }
            });
        }
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

    renderAssayTools(pid) {
        const panel = document.getElementById('assayToolsPanel');
        if (!panel) return;

        this.liveAssayMounted = !!(this.arena.remoteDriven && this.arena.remotePacket?.live_assay);
        if (this.liveAssayMounted) {
            this.liveAssayParadigm = this.arena.remotePacket.paradigm;
            window.mountLiveAssay(this, panel);
            return;
        }
        const spec = ASSAY_CONFIGS[pid] || ASSAY_CONFIGS['open-arena'];
        panel.innerHTML = `
            <!-- Assay Header Card -->
            <div class="hud-card" style="margin-bottom:2px;">
                <div class="hud-card-header">
                    <span style="color:#38bdf8; font-weight:800;">${spec.title}</span>
                    <span class="badge badge-amber">${spec.badge}</span>
                </div>
                <div style="font-size:9px; color:#94a3b8; font-style:italic; line-height:1.3; margin-top:2px;">
                    ${spec.ref}
                </div>
            </div>

            <!-- Assay Specific Parameters -->
            <div class="hud-card">
                <div class="hud-card-header">
                    <span>Assay Parameters</span>
                    <span style="color:#cbd5e1; font-size:8.5px;">LIVE TUNING</span>
                </div>
                <div id="assaySlidersContainer">
                    ${(spec.sliders || []).map(s => `
                        <div class="param-slider-row">
                            <div class="slider-header">
                                <span>${s.label}</span>
                                <span><b id="assay_val_${s.key}">${s.val}</b>${s.unit}</span>
                            </div>
                            <input type="range" id="assay_slider_${s.key}" min="${s.min}" max="${s.max}" step="${s.step}" value="${s.val}">
                            ${s.desc ? `<div class="slider-desc" style="font-size:8.5px; color:#94a3b8; margin-top:3px; line-height:1.25;">${s.desc}</div>` : ''}
                        </div>
                    `).join('')}
                </div>
            </div>

            <!-- Assay Levers & Interventions -->
            <div class="hud-card">
                <div class="hud-card-header">
                    <span>Assay Levers & Interventions</span>
                    <span style="color:#fbbf24; font-size:8.5px;">COMMAND DISPATCH</span>
                </div>
                <div class="stim-btn-row" id="assayActionsContainer" style="display:flex; flex-wrap:wrap; gap:5px;">
                    ${(spec.actions || []).map((act, idx) => `
                        <button class="stim-btn ${act.class || ''}" id="assay_btn_${idx}" style="flex:1 1 45%; padding:5px 7px; font-size:9px;">${act.label}</button>
                    `).join('')}
                </div>
            </div>

            <!-- Assay Live Scientific Metrics -->
            <div class="hud-card">
                <div class="hud-card-header">
                    <span>Live Assay Readouts</span>
                    <span style="color:#4ade80; font-size:8.5px;">TELEMETRY</span>
                </div>
                <div style="display:grid; grid-template-columns:repeat(3, 1fr); gap:6px;" id="assayMetricsGrid">
                    ${(spec.metrics || []).map((m, idx) => `
                        <div style="background:rgba(255,255,255,0.03); border:1px solid rgba(255,255,255,0.08); border-radius:4px; padding:4px 6px;">
                            <div style="color:#94a3b8; font-size:8.5px;">${m.label}</div>
                            <div id="assay_metric_val_${idx}" style="font-weight:700; color:#f8fafc; font-size:11px; font-family:monospace;">${m.get(this.arena)}</div>
                        </div>
                    `).join('')}
                </div>
            </div>

            <!-- Domain-Specific Learning & Performance Chart -->
            <div class="hud-card">
                <div class="hud-card-header">
                    <span>Domain Performance Analyzer</span>
                    <span style="color:#c084fc; font-size:8.5px;">LIVE GRAPH</span>
                </div>
                <canvas id="assayChartCanvas" width="380" height="110" style="width:100%; height:110px; background:rgba(0,0,0,0.35); border:1px solid rgba(255,255,255,0.06); border-radius:4px;"></canvas>
            </div>
        `;

        // Wire slider events
        (spec.sliders || []).forEach(s => {
            const sliderEl = panel.querySelector(`#assay_slider_${s.key}`);
            if (sliderEl) {
                sliderEl.addEventListener('input', (e) => {
                    const val = parseFloat(e.target.value);
                    const valEl = panel.querySelector(`#assay_val_${s.key}`);
                    if (valEl) valEl.textContent = val;
                    if (s.apply) s.apply(this.arena, val);
                    if (this.daemonBridge && this.daemonBridge.connected) {
                        this.daemonBridge.sendCommand('set_param', { name: s.key, value: val });
                    }
                });
            }
        });

        // Wire action button events
        (spec.actions || []).forEach((act, idx) => {
            const btnEl = panel.querySelector(`#assay_btn_${idx}`);
            if (btnEl) {
                btnEl.addEventListener('click', () => {
                    if (act.handler) act.handler(this.arena, this);
                });
            }
        });

        // Register updater for requestAnimationFrame
        this.activeAssayUpdater = () => {
            if (panel.style.display === 'none') return;

            // Update live metrics readouts
            (spec.metrics || []).forEach((m, idx) => {
                const el = panel.querySelector(`#assay_metric_val_${idx}`);
                if (el) el.textContent = m.get(this.arena);
            });

            // Redraw chart canvas
            const canvas = panel.querySelector('#assayChartCanvas');
            if (canvas && spec.drawChart) {
                const ctx = canvas.getContext('2d');
                ctx.clearRect(0, 0, canvas.width, canvas.height);
                spec.drawChart(ctx, canvas.width, canvas.height, this.arena, this);
            }
        };
    }

    updateAssayTools() {
        if ((!this.liveAssayMounted || this.liveAssayParadigm !== this.arena.remotePacket?.paradigm) && this.arena.remoteDriven && this.arena.remotePacket?.live_assay) this.renderAssayTools(this.arena.activeParadigmId);
        if (this.activeAssayUpdater) {
            this.activeAssayUpdater();
        }
    }

    downloadCsv() {
        if (!this.arena.telemetryBuffer || this.arena.telemetryBuffer.length === 0) {
            alert('Telemetry buffer is currently empty. Run simulation steps first.');
            return;
        }
        const keys = [...new Set(this.arena.telemetryBuffer.flatMap(r => Object.keys(r)))];
        const quote = v => '"' + String(v ?? '').replaceAll('"', '""') + '"';
        const headers = keys.map(quote).join(',');
        const rows = this.arena.telemetryBuffer.map(r => keys.map(k => quote(r[k])).join(',')).join('\n');
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
                project: 'Project NeuroFly — compact modular model',
                dataSource: this.arena.remoteDriven ? 'daemon' : 'local_preview',
                trajectoryRule: 'Never connect positions across segment boundaries. REST, pauses, teaching and errors are explicit.',
                activeParadigm: this.arena.activeParadigmId,
                paradigmTitle: this.arena.activeParadigmTitle,
                reference: this.arena.activeParadigmRef,
                lesion: this.arena.lesion,
                currentTrial: this.arena.currentTrial,
                elapsedTimeSec: this.arena.paradigmElapsedSec,
                date: new Date().toISOString(),
                totalSteps: this.arena.stepCount
            },
            // Controller identity and the full run manifest of the streamed run (daemon
            // only; the local preview is illustrative and has no manifest).
            identity: this.arena.remoteDriven ? (this.arena.remotePacket?.identity || null) : null,
            manifest: this.arena.remoteDriven && this.daemonBridge?.manifest?.run_id === this.arena.remotePacket?.identity?.run_id
                ? this.daemonBridge.manifest : null,
            motor: this.arena.remoteDriven ? (this.arena.remotePacket?.motor || null) : {previewWallAssist: !!this.arena.previewWallAssist},
            canonicalMetrics: this.arena.remoteDriven ? this.arena.remotePacket?.metrics : this.arena.getParadigmMetrics(),
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
        if (this.arena.remoteDriven) { this.renderLiveOutcome(); return; }
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

    renderLiveOutcome() {
        const t=this.arena.remotePacket, metric=this.arena.getCanonicalMetricInfo();
        if(this.liveOutcomeSegment!==t.segment_id){
            this.liveOutcomeSegment=t.segment_id;this.liveOutcomeHistory=[];this.liveOutcomeStep=null;
        }
        if(this.liveOutcomeStep!==t.step){
            this.liveOutcomeHistory.push({time:t.sim_time_s,value:metric.rawValue});
            this.liveOutcomeHistory=this.liveOutcomeHistory.slice(-300);this.liveOutcomeStep=t.step;
        }
        const ctx=this.curveCtx,w=this.curveWidth||this.curveCanvas.clientWidth,h=this.curveHeight||this.curveCanvas.clientHeight;
        ctx.clearRect(0,0,w,h);
        const rows=this.liveOutcomeHistory, values=rows.map(r=>r.value).filter(Number.isFinite);
        ctx.fillStyle='#94a3b8';ctx.font='8px monospace';ctx.textAlign='center';
        ctx.fillText('Repeated measures · sim s',w/2,10);
        if(!values.length){ctx.fillText('Outcome not observed',w/2,h/2);return;}
        let lo=Math.min(...values),hi=Math.max(...values);
        const pad=Math.max((hi-lo)*.1,.05);lo-=pad;hi+=pad;
        const left=38,right=w-8,top=23,bottom=h-18;
        ctx.textAlign='right';ctx.fillText(hi.toFixed(2),left-4,top+3);ctx.fillText(lo.toFixed(2),left-4,bottom);
        ctx.strokeStyle='#334155';ctx.beginPath();ctx.moveTo(left,top);ctx.lineTo(left,bottom);ctx.lineTo(right,bottom);ctx.stroke();
        const start=rows[0].time,end=rows[rows.length-1].time,span=Math.max(.02,end-start);
        ctx.strokeStyle='#38bdf8';ctx.lineWidth=1.5;ctx.beginPath();let connected=false;
        for(const row of rows){
            if(!Number.isFinite(row.value)){connected=false;continue;}
            const x=left+(right-left)*(row.time-start)/span,y=bottom-(bottom-top)*(row.value-lo)/(hi-lo);
            connected?ctx.lineTo(x,y):ctx.moveTo(x,y);connected=true;
        }
        ctx.stroke();ctx.textAlign='left';ctx.fillText(start.toFixed(1),left,h-4);
        ctx.textAlign='right';ctx.fillText(end.toFixed(1),right,h-4);
    }

    update() {
        if (this.arena.remoteDriven && this.arena.remotePacket?.live_assay) {
            const t=this.arena.remotePacket;
            const watch=document.getElementById('guideWhatToWatch');
            if(watch)watch.textContent=t.live_assay.limitation+' Use Assay Tools & Levers for connected controls and measured motor responses.';
            const open=t.paradigm==='open-arena';
            ['toolFood','toolAlarm','toolWind','toolPredator'].forEach(id=>{const e=document.getElementById(id);if(e){e.disabled=!open||id==='toolPredator'||this.daemonBridge.readOnly;e.title=e.disabled?'Not available in this live assay':'Place a real stimulus in the daemon arena';}});
            document.querySelectorAll('#paramControlsBox input,#limbDeckPanel input,#limbDeckPanel button,[id^="btnLesion"]').forEach(e=>{e.disabled=true;e.title='Standalone preview control. Use the connected controls in Assay Tools & Levers.';});
        }
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
        if (needleEl) {
            // The gauge saturates at its endpoints; the numeric readout stays raw.
            needleEl.style.left = `${((Math.max(-1, Math.min(1, valence)) + 1) / 2) * 100}%`;
            needleEl.title = `Raw valence: ${valence.toFixed(4)} (gauge spans -1 to +1)`;
        }
        if (curPiEl) curPiEl.textContent = (valence >= 0 ? '+' : '') + valence.toFixed(2);
        this.renderKcMatrix();
        this.renderLearningCurve();
        if (this.arena.remoteDriven && curPiEl) curPiEl.textContent = metric.value;

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
        if (pfl3ErrorEl) pfl3ErrorEl.textContent = this.arena.remoteDriven ? 'Not streamed' : (this.arena.cx.pfl3ErrorR - this.arena.cx.pfl3ErrorL).toFixed(2);
        if (antennaDeflectEl) antennaDeflectEl.textContent = this.arena.remoteDriven ? 'Not streamed' : `${(Math.hypot(this.arena.windVector[0], this.arena.windVector[1]) * 0.12).toFixed(1)} μN`;
        this.renderCompass();

        // Oscilloscope Channels
        const scopeKey=this.arena.remoteDriven ? `${this.arena.remoteSegment}:${this.arena.stepCount}` : null;
        if(!scopeKey || scopeKey!==this.lastScopeKey) this.scopeHistory.push({
            dna02: this.arena.dn.dna02Diff,
            dnp09: this.arena.dn.dnp09,
            bpn: this.arena.dn.bpn,
            mdn: this.arena.dn.mdn,
            dnp01: this.arena.dn.escapeActive ? 50.0 : 0.0
        });
        this.lastScopeKey=scopeKey;
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

        // Update active assay tools panel & metrics if visible
        this.updateAssayTools();
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
        const w = this.scopeWidth || this.scopeCanvas.clientWidth;
        const h = this.scopeHeight || this.scopeCanvas.clientHeight;
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
            if(this.arena.remoteDriven && ch.key==='bpn')continue; // Not supplied by the daemon.
            this.scopeCtx.strokeStyle = ch.color;
            this.scopeCtx.lineWidth = 1.4;
            this.scopeCtx.beginPath();
            for (let i = 0; i < this.scopeHistory.length; i++) {
                const x = (i / Math.max(1,this.scopeHistory.length - 1)) * w;
                const val = this.scopeHistory[i][ch.key];
                const range = this.arena.remoteDriven ? ({dna02:4, dnp09:3.5, bpn:60, mdn:1, dnp01:50}[ch.key]) : 60;
                const y = h / 2 - (val / range) * (h / 2) * ch.scale;
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

// Anything that escapes the handlers below is still reported with assay/run/step.
window.addEventListener('error', (event) => {
    NeuroflyErrors.report('uncaught', event.error || event.message);
});
window.addEventListener('unhandledrejection', (event) => {
    NeuroflyErrors.report('uncaught-promise', event.reason);
});

window.addEventListener('load', () => {
    document.getElementById('btnErrorDismiss')?.addEventListener('click', () => NeuroflyErrors.hide());
    document.getElementById('btnErrorResume')?.addEventListener('click', () => NeuroflyErrors.resume());
    try {
        startNeuroflyApp();
        const previewAssist = document.getElementById('previewWallAssist');
        if (previewAssist && window.arena) {
            previewAssist.checked = !!window.arena.previewWallAssist;
            previewAssist.addEventListener('change', () => { window.arena.previewWallAssist = previewAssist.checked; });
        }
    } catch (e) {
        // Startup failed: say so with context instead of leaving a blank or half-built page.
        NeuroflyErrors.report('startup', e);
    }
});

function startNeuroflyApp() {
    const arena = new ScientificBioArena('arenaCanvas');
    const hud = new ScientificHUD(arena);
    window.arena = arena;
    window.hud = hud;

    let isPaused = false;
    const btnPause = document.getElementById('btnPauseToggle');
    if (btnPause) {
        btnPause.addEventListener('click', () => {
            if (hud.daemonBridge?.connected) {
                hud.daemonBridge.sendCommand('set_paused', {paused:!arena.remotePacket?.paused});
                return;
            }
            isPaused = !isPaused;
            btnPause.textContent = isPaused ? 'Resume' : 'Pause';
            btnPause.classList.toggle('primary', isPaused);
        });
    }

    // Expose convenient top-level app handle
    window.app = {
        arena,
        hud,
        selectParadigm: (pid) => hud.selectParadigm(pid),
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
    let renderFaultPending = NEUROFLY_INJECT === 'render';

    // Each frame has two phases: 'update' (local preview integration + HUD) and
    // 'render' (canvas). A phase that throws is reported once with assay/run/step and
    // then suspended -- not retried every frame -- so the last drawn frame stays on
    // screen. Daemon frames keep arriving and are recorded meanwhile; "Resume view"
    // in the error banner re-enables the phases.
    function runPhase(phase, fn) {
        if (NeuroflyErrors.isSuspended(phase)) return;
        try {
            fn();
        } catch (e) {
            NeuroflyErrors.suspend(phase);
            NeuroflyErrors.report(phase, e);
        }
    }

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
            runPhase('update', () => {
                const speed = (hud && hud.simSpeed) ? hud.simSpeed : 1.0;
                accumulator += rawDt * speed;

                // Fixed physical timestep for the local preview at every display speed;
                // speed changes how many steps run per frame, never the step size.
                const stepDt = 0.02;
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
            });
        }
        runPhase('render', () => {
            if (renderFaultPending && arena.remotePacket) {
                renderFaultPending = false;   // one-shot test-build fault (?inject=render)
                throw new Error('Injected renderer fault (test build)');
            }
            arena.render();
        });
        requestAnimationFrame(loop);
    }
    requestAnimationFrame(loop);
}
