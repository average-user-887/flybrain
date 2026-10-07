/**
 * NeuroFly dashboard and standalone browser preview
 * ==================================================
 * When no daemon drives the view, this file runs an ENGINEERED, illustrative preview,
 * not the connectome and not fitted to fly data:
 * 1. Mushroom body: 120 Kenyon-cell units with a hand-built reward/punishment rate rule.
 * 2. Heading compass: a 16-wedge bump that follows the preview heading, plus a goal-steering term.
 * 3. Descending-drive labels (DNa02, DNp09, MDN, GF ...) computed from the preview's own steering.
 * 4. Engineered Kuramoto tripod oscillator for leg pose; not a biological nerve cord.
 * 5. 2D wall contacts with Coulomb sliding (friction mu, restitution).
 * 6. 14 assay previews (open-arena, t-maze, y-maze, heat-maze, buridan, visual-operant,
 *    wind-tunnel, looming-escape, optomotor, gap-crossing, circadian-dam, courtship,
 *    labyrinth, multisensory-sandbox). EXPERIMENT_GUIDES states what each preview actually
 *    does; scripted choices, hand-set thresholds and stored-coordinate steering are named
 *    there, and unsupported biology (conditioning, place or working memory, motor
 *    planning, efference copy, circadian rhythm, courtship memory) is post-v0.4 backlog
 *    work (docs/POST_V04_FEATURES.md), not a capability of this file or the connectome.
 * 7. Telemetry display and CSV/JSON exporters.
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

    setRealEpgProfile(epgWedges) {
        if (!Array.isArray(epgWedges) || epgWedges.length === 0) return;
        const maxVal = Math.max(0.001, ...epgWedges);
        for (let i = 0; i < this.numWedges; i++) {
            this.bumpProfile[i] = (epgWedges[i] || 0.0) / maxVal;
        }
        let sinSum = 0, cosSum = 0;
        for (let i = 0; i < this.numWedges; i++) {
            sinSum += this.bumpProfile[i] * Math.sin(this.wedgeAngles[i]);
            cosSum += this.bumpProfile[i] * Math.cos(this.wedgeAngles[i]);
        }
        this.headingBump = Math.atan2(sinSum, cosSum);
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

// A display projection, not a scientific measurement producer. Remote values
// come only from the accepted packet for the currently selected sandbox assay.
function sandboxScoreView(arena) {
    const unavailable = 'Unavailable';
    const names = {
        composite: ['composite_benchmark_score', 'compositeScore'],
        coordination: ['locomotor_coordination_index', 'coordinationScore'],
        sensory: ['multisensory_integration_score', 'sensoryIntegrationScore'],
        efficiency: ['biomechanical_efficiency', 'efficiencyScore'],
        smoothness: ['kinematic_smoothness', 'smoothnessScore'],
    };
    const sandbox = arena?.activeParadigmId === 'multisensory-sandbox';
    const packet = arena?.remotePacket;
    const remote = sandbox && arena.remoteDriven && !arena.awaitingDaemon
        && packet?.paradigm === 'multisensory-sandbox';
    const preview = sandbox && !arena.remoteDriven && !arena.awaitingDaemon && !packet;
    const source = remote ? (packet.timing?.replay ? 'Recording' : 'Daemon')
        : preview ? 'Local preview' : unavailable;
    const raw = {};
    const text = {};
    for (const [key, [remoteKey, localKey]] of Object.entries(names)) {
        const value = remote ? packet.metrics?.[remoteKey]
            : preview ? arena.paradigmState?.[localKey] : null;
        raw[key] = Number.isFinite(value) ? value : null;
        text[key] = raw[key] === null ? unavailable : key === 'composite'
            ? `${raw[key].toFixed(1)} / 100` : `${(raw[key] * 100).toFixed(1)}%`;
    }
    return {raw, text, source,
        label: 'Sandbox heuristic body proxy',
        note: sandbox ? `${source} · heuristic body proxy; not learned biological validation.`
            : 'Unavailable for this assay · sandbox heuristic body proxy.',
        catalog: raw.composite === null ? unavailable
            : `${text.composite} · ${preview ? 'preview proxy' : 'body proxy'}`};
}

function renderSandboxScorecard(arena, doc) {
    const score = sandboxScoreView(arena);
    if ((arena.remoteDriven || arena.awaitingDaemon) && arena.getObservationDisplay
            && arena.getObservationDisplay().live.state !== 'available') {
        for (const key of Object.keys(score.text)) score.text[key] = 'Unavailable';
        score.note = 'Heuristic body proxy unavailable · ' + arena.getObservationDisplay().transportLabel;
    }
    const fields = {deckCompositeScore: 'composite', deckCoordScore: 'coordination',
        deckSensoryScore: 'sensory', deckEfficacyScore: 'efficiency', deckSmoothScore: 'smoothness'};
    for (const [id, field] of Object.entries(fields)) {
        const el = doc.getElementById(id);
        if (el) { el.textContent = score.text[field]; el.title = score.note; }
    }
    const source = doc.getElementById('deckScorecardSource');
    if (source) source.textContent = score.note;
    return score;
}

// Readable reasons a daemon gives for an unknown trial clock (provenance.py).
const TRIAL_CLOCK_REASONS = {
    checkpoint_predates_trial_clock: 'This brain was saved before trial timing was recorded.',
};

/** Text for the navbar and trial clock readouts; each names the clock it shows.
 *  Live step/sim time count from zero in every daemon process (clocks.session); a
 *  replay shows the recording's session clock as recorded; a graph run's own neural
 *  time is shown separately.  Trial values read Unknown only when the packet's
 *  trial clock says so; packets without clocks keep their values unchanged. */
function clockReadouts(view, packet, remote, replay) {
    const trial = remote ? packet?.clocks?.trial : null;
    const elapsedUnknown = !!trial && trial.elapsed_known === false;
    const trialUnknown = !!trial && trial.trial_known === false;
    const why = trial?.reason ? (TRIAL_CLOCK_REASONS[trial.reason] || `Reason: ${trial.reason}.`) : '';
    const g = remote ? packet?.clocks?.graph : null;
    const graph = g && Number.isFinite(g.elapsed_s) && Number.isFinite(g.step) ? g : null;
    const elapsed = Number(view.paradigmElapsedSec).toFixed(2) + 's';
    const number = view.currentTrial;
    // The measurement window is its own clock: after a restore it starts again
    // while the trial clock continues (clocks.observation.lineage).
    const obs = remote ? packet?.clocks?.observation : null;
    const lineage = obs?.lineage || null;
    const windowElapsed = Number.isFinite(obs?.segment_elapsed_s) ? obs.segment_elapsed_s.toFixed(2) + 's' : '--';
    const restoredFrom = Number.isFinite(lineage?.restored_trial_elapsed_s)
        ? lineage.restored_trial_elapsed_s.toFixed(2) + 's' : 'an unknown time';
    return {
        timeLabel: replay ? 'Recorded session time' : remote ? 'Session time' : 'Preview time',
        stepLabel: replay ? 'Recorded session step' : remote ? 'Session step' : 'Preview step',
        sessionTitle: replay ? 'Simulated time of the daemon session that made this recording, as recorded.'
            : remote ? 'Simulated time since this daemon process started. It starts again from zero when '
                + 'the daemon restarts; the saved brain, world and trial clock are not reset.'
            : 'Time in this browser preview; no daemon is driving it.',
        simTime: Number(view.simTime).toFixed(2) + 's',
        step: String(view.stepCount),
        graphVisible: !!graph,
        graphTime: graph ? graph.elapsed_s.toFixed(2) + 's' : '--',
        graphTitle: graph ? `Simulated time of this saved connectome brain (step ${graph.step}), carried `
            + 'across daemon restarts. It is not the trial clock.' : '',
        trial: trialUnknown ? 'Unknown' : '#' + number,
        guideTrial: trialUnknown ? 'TRIAL UNKNOWN' : 'TRIAL #' + number,
        trialTitle: trialUnknown ? `The trial number is unknown. ${why} This is trial ${number} counted since then.` : '',
        windowVisible: !!lineage,
        window: lineage ? `Window ${windowElapsed} · restarted after `
            + (lineage.reason === 'daemon_restart' ? 'daemon restart' : 'assay switch') : '',
        windowTitle: lineage ? 'The measurement window started again; it does not continue the earlier one. '
            + (lineage.parent_observation === 'terminal_recorded'
                ? 'The earlier measurement has a recorded end' + (lineage.parent_observation_evidence?.end_reason
                    ? ` (${lineage.parent_observation_evidence.end_reason})` : '') + '; that record is kept. '
                : 'Whether the earlier measurement finished is unknown: no recorded end was found. ')
            + `Its metric values were not carried over. The trial clock continues from ${restoredFrom}, `
            + 'so trial elapsed time can exceed the window.' : '',
        elapsed: elapsedUnknown ? 'Unknown' : elapsed,
        elapsedTitle: elapsedUnknown ? `The trial's elapsed time is unknown until a new trial starts. ${why} `
            + `At least ${elapsed} have passed since the brain was restored.` : '',
    };
}

class ScientificBioArena {
    constructor(canvasId) {
        this.canvas = document.getElementById(canvasId);
        this.ctx = this.canvas.getContext('2d');

        this.resize();
        window.addEventListener('resize', () => this.resize());
        if (typeof ResizeObserver !== 'undefined') {
            this.resizeObserver = new ResizeObserver(() => this.resize());
            this.resizeObserver.observe(this.canvas);
        }

        this.activeParadigmId = 'open-arena';
        // Illustrative local preview only: the wall-avoidance reflex is an engineered
        // assist, not physics. On by default (legacy preview), toggled in the identity bar.
        this.previewWallAssist = true;
        this.activeParadigmTitle = 'Open Arena Multi-Modal Foraging';
        this.activeParadigmRef = 'Background: Budick & Dickinson (2006); Maimon et al. (2010) (citation not verified in this repository)';
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
        if (rect.width === 0 || rect.height === 0) return;
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

            const capabilities = arenaToolCapabilities(this, window.hud?.daemonBridge);
            if (!capabilities.tools[this.toolMode]?.enabled) {
                this.toolMode = 'select';window.hud?.reconcileToolCapabilities?.();return;
            }
            if (capabilities.remote) {
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
                this.cancelPreviewGust?.();
                this.windVector = [-Math.cos(angle) * 15.0, -Math.sin(angle) * 15.0];
            }
        });
    }

    isStandalonePreview() {
        const bridge = window.hud?.daemonBridge;
        return !this.remoteDriven && !this.awaitingDaemon && !bridge?.connected
            && !bridge?.replayMode && !bridge?.switchPending;
    }

    cancelPreviewGust(restore = false) {
        const gust = this.previewGust;
        if (!gust) return;
        clearTimeout(gust.timer);
        this.previewGust = null;
        if (restore && this.isStandalonePreview() && this.paradigmState === gust.owner
                && this.activeParadigmId === gust.assay && this.windVector === gust.vector
                && Object.is(this.windVector[0], gust.applied[0]) && Object.is(this.windVector[1], gust.applied[1])) {
            this.windVector = gust.prior;
        }
    }

    startPreviewGust(kind = 'axial') {
        if (!this.isStandalonePreview() || !['axial', 'crosswind'].includes(kind)) return;
        this.cancelPreviewGust(true);
        const vector = kind === 'crosswind'
            ? [this.windVector[0], (Math.random() - 0.5) * 30.0] : [-35, 0];
        const gust = {owner: this.paradigmState, assay: this.activeParadigmId,
            prior: [...this.windVector], vector, applied: [...vector]};
        this.windVector = gust.vector;
        this.previewGust = gust;
        gust.timer = setTimeout(() => {
            if (this.previewGust === gust) this.cancelPreviewGust(true);
        }, kind === 'crosswind' ? 2500 : 2000);
    }

    setLesion(type) {
        this.lesion = type;
        this.cx.isLesioned = (type === 'DELTA_CX');
        this.mb.plasticityEnabled = (type !== 'DELTA_MB');
        this.gfLesioned = (type === 'DELTA_GF');
        this.joLesioned = (type === 'DELTA_JO');
    }

    initParadigm(paradigmId) {
        this.cancelPreviewGust?.(true);
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
                this.activeParadigmTitle = 'Open Arena Multi-Modal Foraging';
                this.activeParadigmRef = 'Background: Budick & Dickinson (2006); Maimon et al. (2010) (citation not verified in this repository)';
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
                this.activeParadigmTitle = 'T-Maze Odour Choice';
                this.activeParadigmRef = 'Background: Tully & Quinn (1985) J Comp Physiol A 157:263–277 (abstract read); Dudai (1976) (citation not verified in this repository)';
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
                this.activeParadigmTitle = 'Y-Maze Exploration';
                this.activeParadigmRef = 'Background: Buchanan et al. (2015) is a handedness study, not an alternation study; Churgin (2017) (citation not verified in this repository)';
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
                this.activeParadigmTitle = 'Thermal Heat-Maze';
                this.activeParadigmRef = 'Background: Ofstad, Zuker & Reiser (2011) Nature 474:204–207 (cited only for the existence of visual place learning)';
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
                this.activeParadigmTitle = 'Buridan\'s Paradigm';
                this.activeParadigmRef = 'Background: Götz (1980); Colomb et al. (2012) (citation not verified in this repository)';
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
                this.activeParadigmTitle = 'Visual Operant Flight Simulator';
                this.activeParadigmRef = 'Background: Wolf & Heisenberg (1991); Liu et al. (2006) (citation not verified in this repository)';
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
                this.activeParadigmTitle = 'Wind Tunnel Plume';
                this.activeParadigmRef = 'Background: Álvarez-Salvado et al. (2018) eLife (recorded, not re-read); Demir et al. (2020) (citation not verified in this repository)';
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
                this.activeParadigmTitle = 'Looming Escape';
                this.activeParadigmRef = 'Background: von Reyn et al. (2014) Nat Neurosci 17:962–970; Card & Dickinson (2008) (citation not verified in this repository)';
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
                this.activeParadigmTitle = 'Optomotor Drum';
                this.activeParadigmRef = 'Background: Götz (1964); Kim et al. (2017) on efference copy (citation not verified in this repository)';
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
                this.activeParadigmTitle = 'Gap Crossing';
                this.activeParadigmRef = 'Background: Pick & Strauss (2005); Triphan et al. (2010) (citation not verified in this repository)';
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
                this.activeParadigmTitle = 'Circadian DAM Monitor';
                this.activeParadigmRef = 'Background: Konopka & Benzer (1971); Allada & Chung (2010) (citation not verified in this repository)';
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
                this.activeParadigmTitle = 'Courtship Chamber';
                this.activeParadigmRef = 'Background: Siegel & Hall (1979); Keleman et al. (2007) (citation not verified in this repository)';
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
                this.activeParadigmTitle = 'Corridor Obstacle Labyrinth';
                this.activeParadigmRef = 'Engineered maze with Coulomb sliding contacts (no biological reference)';
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
                this.activeParadigmTitle = 'Multisensory Sandbox';
                this.activeParadigmRef = 'Project NeuroFly compact modular sensorimotor model';
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
        if(window.hud?.daemonBridge?.replayMode)return;
        this.cancelPreviewGust?.(true);
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
        if (this.remoteDriven || this.awaitingDaemon) return this.getCanonicalMetricInfo().rawValue;
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
        if (pid === 'multisensory-sandbox') return sandboxScoreView(this).raw.composite;
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
                this.paradigmStatus = isSaccade ? 'PREVIEW SACCADE (HAND-SET SLIP CUT)' : 'OPTO-STABILIZATION';
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

                // When the daemon streams this paradigm its kinematics and benchmark metrics
                // are mirrored into paradigmState by DaemonBridgeClient; do not overwrite them.
                if (this.remoteDriven) {
                    this.paradigmStatus = `${sandboxScoreView(this).text.composite} · daemon body proxy`;
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

                this.paradigmStatus = `LOCAL PREVIEW BODY PROXY: ${p.compositeScore.toFixed(1)} / 100`;
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
        if (this.remoteDriven || this.awaitingDaemon) {
            return window.NeuroFlyObservationRenderer.primary(this.getObservationDisplay());
        }
        const preview = this.getPreviewMetricInfo();
        return {...preview, label: preview.label + ' · local preview', sub: 'Preview output · ' + (preview.sub || '')};
    }

    getObservationDisplay() {
        const bridge = window.hud?.daemonBridge;
        const options = {connected: !!bridge?.connected || !!bridge?.replayMode,
            pending: !!bridge?.switchPending, assay: this.activeParadigmId,
            replay: !!bridge?.replayMode || !!this.remotePacket?.timing?.replay,
            validityUpdate: bridge?.observationValidityUpdate || null};
        const key = JSON.stringify(options);
        if (this.observationPacket !== this.remotePacket || this.observationOptions !== key) {
            this.observationPacket = this.remotePacket;
            this.observationOptions = key;
            this.observationDisplay = window.NeuroFlyObservationRenderer.view(this.remotePacket, options);
        }
        return this.observationDisplay;
    }

    getPreviewMetricInfo() {
        if (this.activeParadigmId === 'multisensory-sandbox') {
            const score = sandboxScoreView(this);
            return {label: score.label + ' · ' + score.source, rawValue: score.raw.composite,
                unit: ' /100', value: score.text.composite, sub: score.note};
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
                    label: 'Occupancy index',
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
                    sub: `Looming Size: ${p.thetaDeg.toFixed(1)}° | Preview Vm proxy: ${p.vm.toFixed(0)} mV`
                };
            case 'optomotor':
                return {
                    label: 'Optomotor Gain',
                    value: (p.gain || 0.88).toFixed(2),
                    sub: `Preview HS proxy: ${p.hsFiringRate.toFixed(0)} Hz | Hand-set saccade slip cut: 85%`
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
        const remote = this.remoteDriven || this.awaitingDaemon || !!this.remotePacket;
        const current = !this.awaitingDaemon && this.remotePacket?.paradigm === 'optomotor';
        const angle = remote ? (current ? this.remotePacket.scene?.drum_angle_deg : null) : p.drumAngleDeg;
        if (!Number.isFinite(angle)) {
            ctx.fillStyle = '#94a3b8'; ctx.font = '10px monospace';
            ctx.fillText('Grating phase unavailable', sc.x - 70, sc.y + 95);
            return;
        }

        const numStripes = 24;
        if (p.contrast===0) {
            ctx.fillStyle='#64748b';ctx.beginPath();ctx.arc(sc.x,sc.y,85,0,Math.PI*2);ctx.fill();
        }
        for (let i = 0; p.contrast!==0 && i < numStripes; i++) {
            const a1 = ((angle + i * (360 / numStripes)) * Math.PI) / 180;
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
        ref: "Background: Budick & Dickinson (2006); Maimon et al. (2010) (citation not verified in this repository)",
        whatToWatch: [
            "Preview: food odour (Odor A) and alarm odour (Odor B) are sampled at two antenna points and averaged, so the browser preview has no left–right odour comparison and does not steer toward the food. (The daemon's modular controller does compare both sides.)",
            "Preview: alarm odour is subtracted from food odour in a single turning term.",
            "Preview threats are a distance-and-approach-speed trigger in this browser model, not optical-expansion detection; threat placement is unavailable live."
        ],
        params: [
            { key: 'predatorSpeed', label: 'Predator Speed', min: 10, max: 60, step: 5, val: 25, unit: 'mm/s', desc: 'Speed of preview threats. The preview escape fires on distance and approach speed, not on optical expansion.', apply: (a, v) => { a.predators.forEach(p => { const sp = Math.hypot(p.vx, p.vy) || 1; p.vx = (p.vx / sp) * v; p.vy = (p.vy / sp) * v; }); } },
            { key: 'windVelocity', label: 'Wind Velocity', min: 0, max: 40, step: 5, val: 15, unit: 'mm/s', desc: 'Preview wind vector. The preview turns it into an upwind steering term; no Johnston’s organ model is simulated.', apply: (a, v) => { a.cancelPreviewGust?.(); a.windVector = [-v, 0]; } }
        ]
    },
    't-maze': {
        title: "T-Maze Odour Choice",
        ref: "Background: Tully & Quinn (1985) J Comp Physiol A 157:263–277 (abstract read); Dudai (1976) (citation not verified in this repository)",
        whatToWatch: [
            "Preview: the fly walks up the stem to the junction.",
            "Preview: Arm A carries Odor A and a reward signal; Arm B carries Odor B and a shock signal.",
            "Preview arm choice is scripted: once the preview mushroom-body valence moves past ±0.05 the fly takes Arm A, otherwise it picks at random. This is not learned conditioning.",
            "Connectome odour–shock learning is planned, not in v0.4 (NEXT-03)."
        ],
        params: []
    },
    'y-maze': {
        title: "Y-Maze Exploration",
        ref: "Background: Buchanan et al. (2015) is a handedness study, not an alternation study; Churgin (2017) (citation not verified in this repository)",
        whatToWatch: [
            "Preview: the fly visits the 3 arms at 120° intervals and returns to the hub.",
            "Preview alternation is a hand-set rule (72% chance of switching turn direction), so the alternation rate is built in, not computed by any circuit.",
            "No Delta7 or DNa02 handedness mechanism is simulated in the preview. Working memory is planned, not in v0.4 (NEXT-07)."
        ],
        params: []
    },
    'heat-maze': {
        title: "Thermal Heat-Maze",
        ref: "Background: Ofstad, Zuker & Reiser (2011) Nature 474:204–207 (cited only for the existence of visual place learning)",
        whatToWatch: [
            "Preview: the floor is hot (36.5 °C by default) outside a 24 °C cool refuge.",
            "Preview: before the first visit the fly searches at random; after it, the preview steers straight to the refuge's stored coordinates. The perimeter stripes are drawn but not used for navigation.",
            "This is not visual place learning. Landmark place memory is planned, not in v0.4 (NEXT-07)."
        ],
        params: [
            { key: 'refugeRadius', label: 'Refuge Radius', min: 6, max: 15, step: 1, val: 9, unit: 'mm', desc: 'Preview cool-tile radius. The preview uses stored coordinates, not landmark triangulation.', apply: (a, v) => { a.paradigmState.refugeRadius = v; } }
        ]
    },
    'buridan': {
        title: "Buridan's Paradigm",
        ref: "Background: Götz (1980); Colomb et al. (2012) (citation not verified in this repository)",
        whatToWatch: [
            "Preview: two opposing black stripes at 0° and 180°; the blue ring is drawn, and containment is a geometric boundary.",
            "Preview: the fly is steered toward one stripe, and the target switches when it passes a fixed position, so walking back and forth is scripted.",
            "Fixation and centrophobism (time outside the 25 mm centre) are measured from the preview path; no target score applies."
        ],
        params: [
            { key: 'platformRadius', label: 'Platform drawing radius (preview)', min: 35, max: 60, step: 5, val: 50, unit: 'mm', desc: 'Changes the 2D preview disk drawing only; containment and locomotion remain unchanged.', apply: (a, v) => { a.paradigmState.platformRadius = v; } }
        ]
    },
    'visual-operant': {
        title: "Visual Operant Flight Simulator",
        ref: "Background: Wolf & Heisenberg (1991); Liu et al. (2006) (citation not verified in this repository)",
        whatToWatch: [
            "Preview: the fly's yaw rotates a 360° drum through a coupling gain.",
            "Preview: two opposite drum quadrants give a heat-punishment signal; the other two are safe.",
            "Watch yaw and safe-quadrant occupancy. Pattern-specific operant memory is planned, not in v0.4 (NEXT-07)."
        ],
        params: [
            { key: 'couplingGain', label: 'Yaw Coupling Gain', min: 50, max: 200, step: 10, val: 120, unit: '°/s', desc: 'Closed-loop coupling gain between flight yaw torque and drum rotation. Higher values make steering more responsive.', apply: (a, v) => { a.paradigmState.couplingGain = v; } }
        ]
    },
    'wind-tunnel': {
        title: "Wind Tunnel Plume",
        ref: "Background: Álvarez-Salvado et al. (2018) eLife (recorded, not re-read); Demir et al. (2020) (citation not verified in this repository)",
        whatToWatch: [
            "Preview: airflow of 25 mm/s and a stationary Gaussian odour plume from the nozzle; it is not intermittent or turbulent.",
            "Preview states are labels: odour above a fixed threshold is called SURGE, below it CAST. The fly steers upwind in both.",
            "Turbulent plumes are planned, not in v0.4 (NEXT-09)."
        ],
        params: []
    },
    'looming-escape': {
        title: "Looming Escape",
        ref: "Background: von Reyn et al. (2014) Nat Neurosci 17:962–970; Card & Dickinson (2008) (citation not verified in this repository)",
        whatToWatch: [
            "Preview: a disk expands as 2·atan(r/v ÷ time to collision).",
            "Preview: an escape is triggered when the disk passes a hand-set 65° threshold; the displayed membrane value is a formula of disk size, not a simulated LPLC2/Col4 or Giant Fiber neuron.",
            "von Reyn et al. (2017) report GF spikes on average at a fixed delay after a disk passes 39°; the preview threshold is not fitted to this. GF threshold control is planned, not in v0.4 (NEXT-02)."
        ],
        params: [
            { key: 'rOverV', label: 'Looming r/v Ratio', min: 10, max: 50, step: 5, val: 25, unit: 'ms', desc: 'Preview disk size-to-speed ratio (r/v). Smaller values expand faster.', apply: (a, v) => { a.paradigmState.rOverVS = v / 1000; } }
        ]
    },
    'optomotor': {
        title: "Optomotor Drum",
        ref: "Background: Götz (1964); Kim et al. (2017) on efference copy (citation not verified in this repository)",
        whatToWatch: [
            "Preview: the drum rotates and the preview fly turns at a fixed 0.88 × drum speed; no T4/T5 or HS/VS cells are simulated in the preview.",
            "Preview saccades are scripted every 2.2 s, and slip during them is cut by a hand-set 85%. This is not an efference-copy model; efference copy is a post-v0.4 proposal (NEXT-06).",
            "Connectome: with photoreceptor-only input no T4/T5 direction selectivity appears, and about 55% of lamina cells have no photoreceptor input in the scan. The provisional optomotor result relies on an encoder that imposes direction selectivity."
        ],
        params: []
    },
    'gap-crossing': {
        title: "Gap Crossing",
        ref: "Background: Pick & Strauss (2005); Triphan et al. (2010) (citation not verified in this repository)",
        whatToWatch: [
            "Preview: the fly walks along a runway toward a gap.",
            "Preview decision is a hand-set threshold: gaps up to 3.8 mm are crossed, wider gaps trigger a turn back. No central-complex circuit, leg reach or learning is simulated.",
            "Leg-reach planning is planned, not in v0.4 (NEXT-10)."
        ],
        params: [
            { key: 'gapWidth', label: 'Chasm Width', min: 2.0, max: 5.5, step: 0.2, val: 3.5, unit: 'mm', desc: 'Preview gap width. The preview crosses gaps up to its hand-set 3.8 mm threshold and turns back otherwise.', apply: (a, v) => { a.paradigmState.gapWidthMm = v; } }
        ]
    },
    'circadian-dam': {
        title: "Circadian DAM Monitor",
        ref: "Background: Konopka & Benzer (1971); Allada & Chung (2010) (citation not verified in this repository)",
        whatToWatch: [
            "Preview: the fly walks in a tube; crossings of the mid-tube beam are counted.",
            "Preview: immobility of at least 5 preview minutes (2 preview minutes per simulated second) is counted as a sleep bout.",
            "No clock is simulated, so there are no morning or evening anticipation peaks. An endogenous oscillator is planned, not in v0.4 (NEXT-08)."
        ],
        params: []
    },
    'courtship': {
        title: "Courtship Chamber",
        ref: "Background: Siegel & Hall (1979); Keleman et al. (2007) (citation not verified in this repository)",
        whatToWatch: [
            "Preview: the male is steered straight toward a stationary female.",
            "Preview: close to her, a wing-extension angle is drawn; a mated female adds rejection kicks and a punishment signal.",
            "No P1 neurons or cVA are simulated in the preview, and approach does not change with experience. Courtship memory is planned, not in v0.4 (NEXT-08)."
        ],
        params: []
    },
    'labyrinth': {
        title: "Corridor Obstacle Labyrinth",
        ref: "Engineered maze with Coulomb sliding contacts (no biological reference)",
        whatToWatch: [
            "Preview: a corridor maze with junctions and dead ends.",
            "Preview walls use Coulomb sliding (friction mu, default 0.5; restitution 0.1) so the fly neither sticks nor tunnels.",
            "Preview steering heads toward the goal's coordinates, with engineered wall-proximity avoidance; no Johnston's organ, map or planner is simulated (NEXT-07)."
        ],
        params: [
            { key: 'friction', label: 'Wall Friction', min: 0.1, max: 0.9, step: 0.1, val: 0.5, unit: 'mu', desc: 'Coulomb crawling contact friction against corridor walls. Higher friction dampens sliding velocity.', apply: (a, v) => { a.currentWalls.forEach(w => w.friction = v); } }
        ]
    },
    'multisensory-sandbox': {
        title: "Multisensory Sandbox",
        ref: "Project NeuroFly compact modular sensorimotor model",
        whatToWatch: [
            "Preview: food, repellent, cVA, thermal and wind cues are computed. Steering heads toward the food's coordinates; the other cues do not steer the preview fly.",
            "Inspect 6 legs whose coxa, femur and tibia angles come from an engineered Kuramoto tripod oscillator, not a biological nerve cord.",
            "Use explicitly labeled local preview cadence, GF escape, and wind gust controls in the Limb & Preview Controls deck.",
            "Inspect a heuristic body proxy across Coordination, Sensory Alignment, Smoothness, and Efficiency; this aggregate does not validate learned biology."
        ],
        params: [
            { key: 'windMagnitude', label: 'Wind Velocity', min: 0, max: 40, step: 5, val: 15, unit: 'mm/s', desc: 'Preview wind vector speed. In the sandbox it does not steer the preview fly, and no Johnston’s organ model is simulated.', apply: (a, v) => { a.cancelPreviewGust?.(); a.windVector = [-v, 0]; } },
            { key: 'hotspotTemp', label: 'Hotspot Temp', min: 28, max: 45, step: 1, val: 38.5, unit: '°C', desc: 'Peak temperature of the preview hot spot. It feeds a punishment signal only; it does not steer the preview fly.', apply: (a, v) => { if (a.paradigmState) a.paradigmState.hotspotTemp = v; } },
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
/** Achieved speed with enough digits to stay truthful when far below 1x (0.011x, not 0.0x). */
function formatSimSpeed(x) {
    if (!Number.isFinite(x)) return '--';
    if (x >= 10) return `${x.toFixed(0)}x`;
    if (x >= 1) return `${x.toFixed(1)}x`;
    if (x <= 0) return '0x';
    return `${Number(x.toPrecision(2))}x`;
}
window.neuroflyFormatSimSpeed = formatSimSpeed;

function validateRequestedSpeed(raw) {
    if (raw === '' || raw === null || raw === undefined) {
        return {ok:false, message:'Enter a speed from 0.1x through 100x.'};
    }
    const value = Number(raw);
    if (!Number.isFinite(value)) return {ok:false, message:'Speed must be a finite number.'};
    if (value < 0.1 || value > 100) return {ok:false, message:'Speed must be between 0.1x and 100x.'};
    return {ok:true, value};
}

function requestedSpeedText(value) {
    return `${Number(value)}x`;
}

function deliveryBuildState(pageBuild, delivery) {
    const local = !pageBuild || pageBuild === '__NEUROFLY_WEB_BUILD__';
    const daemonBuild = typeof delivery?.web_build === 'string' ? delivery.web_build : null;
    return {page: local ? 'local source' : pageBuild, daemon: daemonBuild,
            stale: !local && !!daemonBuild && pageBuild !== daemonBuild};
}

window.neuroflyValidateRequestedSpeed = validateRequestedSpeed;
window.neuroflyDeliveryBuildState = deliveryBuildState;

const ARENA_TOOL_IDS = ['toolSelect', 'toolFood', 'toolAlarm', 'toolPredator', 'toolWind'];
function arenaToolCapabilities(arena, bridge) {
    const replay = !!bridge?.replayMode || !!arena?.remotePacket?.timing?.replay;
    const remote = replay || !!bridge?.connected || !!arena?.remoteDriven || !!arena?.awaitingDaemon;
    const open = !replay && !!bridge?.connected && !bridge.readOnly && !bridge.switchPending
        && !!arena?.remoteDriven && !arena?.awaitingDaemon
        && arena.activeParadigmId === 'open-arena' && arena.remotePacket?.paradigm === 'open-arena';
    const reason = replay ? 'Replay is read-only.' : !bridge?.connected ? 'Connect to the live daemon first.'
        : bridge.readOnly ? 'This daemon is read-only.' : bridge.switchPending ? 'Wait for the switch acknowledgement.'
        : 'Spatial food and alarm placement is available only in the connected Open Arena.';
    const tool = (enabled, label, title) => ({enabled, label, title});
    return {remote, replay, previewAssist: !remote,
        help: remote ? 'Food/alarm: connected Open Arena only. Threat placement unavailable live. Wind: use Airflow in Assay Tools & Levers.'
            : 'Standalone preview tools. Wind clicks set 15 mm/s toward the arena origin; no drag control.',
        tools: {
            select: tool(true, 'Neutral selection', 'Neutral pointer; clicking places no stimulus.'),
            food: tool(!remote || open, 'Drop Food (Odor A)', remote ? (open ? 'Place food in the connected Open Arena.' : reason) : 'Place food in the standalone preview.'),
            alarm: tool(!remote || open, 'Alarm Pheromone (Odor B)', remote ? (open ? 'Place alarm odor in the connected Open Arena.' : reason) : 'Place alarm odor in the standalone preview.'),
            predator: tool(!remote, remote ? 'Threat unavailable (live/replay)' : 'Deploy preview threat', 'Connected threat placement is not implemented; standalone preview only.'),
            wind: tool(!remote, remote ? 'Wind: use Airflow control' : 'Click to set preview wind (15 mm/s)', remote ? 'Spatial wind placement is unsupported. Use connected Airflow in Assay Tools & Levers.' : 'Click to set preview wind at 15 mm/s toward the arena origin.'),
        }};
}
window.neuroflyArenaToolCapabilities = arenaToolCapabilities;

/** Live panel declarations belong to the accepted controller, not only its assay.
 * Parameter values update in place so ordinary changes preserve input focus. */
function liveAssayMountKey(packet) {
    if (!packet?.live_assay) return null;
    const capability = packet.live_assay;
    return JSON.stringify([packet.paradigm, packet.identity, packet.brain_id,
        {...capability, parameters: capability.parameters.map(({value, ...parameter}) => parameter)}]);
}

const SELECTABLE_BACKENDS = ['modular', 'connectome-fixed', 'connectome-plastic', 'connectome-with-trained-readout'];
function verifiedBackendIdentity(identity) {
    return !!identity && SELECTABLE_BACKENDS.includes(identity.backend)
        && ['run_id', 'instance_id', 'daemon_run_id'].every(key => typeof identity[key] === 'string' && identity[key].length > 0)
        && Number.isInteger(identity.activation) && identity.activation >= 0;
}
function backendSelectorState(arena, bridge, waiting = false) {
    const pkt = arena?.remotePacket;
    const current = verifiedBackendIdentity(pkt?.identity) && pkt.identity.daemon_run_id === pkt.run_id
        ? pkt.identity : null;
    const ack = bridge?.lastBackendAck?.identity;
    const identity = !bridge?.replayMode && !pkt?.timing?.replay && current && verifiedBackendIdentity(ack) && ack.daemon_run_id === current.daemon_run_id
        && ack.activation >= current.activation ? ack : current;
    let reason = '';
    if (bridge?.replayMode || pkt?.timing?.replay) reason = 'Replay is read-only; exit replay to change the controller.';
    else if (!bridge?.connected || !bridge?.activeUrl) reason = 'Connect to the live daemon to change the controller.';
    else if (bridge.readOnly) reason = 'This daemon is read-only.';
    else if (arena?.awaitingDaemon || !arena?.remoteDriven || !identity) reason = 'Waiting for a verified live controller identity.';
    else if (waiting || bridge.switchPending) reason = 'Waiting for the final switch acknowledgement.';
    return {backend: identity?.backend || '', identity, reason, allowed: !reason};
}
window.neuroflyBackendSelectorState = backendSelectorState;

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

// Graph controller backends (provenance.GRAPH_BACKENDS). On these the optomotor
// assay runs the WP5 loop (166,700-neuron LIF at 2-20 ms steps), measured at about
// 0.1 simulated seconds per wall second: 1x is unattainable on this hardware and the
// UI says so rather than letting the display pretend otherwise (docs/WP5_OPTOMOTOR.md
// section 7). The daemon runs slower than requested; it never drops simulation steps.
const GRAPH_BACKENDS = ['connectome-fixed', 'connectome-plastic', 'connectome-with-trained-readout'];
const OPTOMOTOR_SIM_S_PER_WALL_S = 0.1;

/** Derive display/control capabilities only from the active packet and measured probes. */
function graphPanelCapabilities(pkt = {}) {
    const identity = pkt.identity || {};
    const backend = String(identity.backend || pkt.backend || 'modular');
    const graph = GRAPH_BACKENDS.includes(backend);
    const wp6 = pkt.plasticity?.wp6 || pkt.connectome?.wp6;
    const wp6Measured = graph && backend === 'connectome-plastic' && wp6
        && Number.isInteger(wp6.n_edges) && wp6.n_edges > 0
        && Number.isFinite(wp6.mean_delta) && Number.isFinite(wp6.max_delta);
    const epgWedges = pkt.neural?.epg_wedges;
    // The dedicated optomotor loop currently publishes a zero-filled placeholder;
    // the general graph path computes wedges from resolved EPG neuron counts.
    const epgMeasured = graph && pkt.connectome?.epg_available === true
        && Array.isArray(epgWedges) && epgWedges.length === 16
        && epgWedges.every(Number.isFinite) && Number.isFinite(pkt.connectome?.epg_bump_phase);
    const label = identity.label || backend;
    const modularReason = graph
        ? `Unavailable: the 120-KC modular mushroom body is not the controller of this ${label} run.`
        : '';
    return {
        backend, label, graph, modularMemory: !graph,
        wp6Measured: !!wp6Measured, epgMeasured,
        learningControl: !graph || !!wp6Measured,
        teach: !graph, reverse: !graph, probe: !graph,
        saveCheckpoint: true,
        modularReason,
        gaitLabel: graph ? 'Body gait proxy · model-derived from streamed pose' : 'Kuramoto tripod gait · modular model',
    };
}
window.neuroflyGraphPanelCapabilities = graphPanelCapabilities;

/** Keep legacy modular assay claims from being presented as graph-controller facts. */
function assayLimitationForController(pkt = {}) {
    const limitation = String(pkt.live_assay?.limitation || 'No assay limitation was declared.');
    return graphPanelCapabilities(pkt).graph
        ? `Legacy modular assay note (not a graph measurement or capability): ${limitation}`
        : limitation;
}
window.neuroflyAssayLimitationForController = assayLimitationForController;

/** Select DN display evidence without promoting pose-derived compatibility values. */
function graphDnReadout(pkt = {}, capabilities = graphPanelCapabilities(pkt)) {
    if (capabilities.graph) {
        const rates = pkt.connectome?.dn_rates;
        return {
            rates: rates || {},
            unavailable: pkt.connectome?.dn_unavailable || {},
            missingReason: rates ? null : 'No graph-controller DN measurement was streamed for this step.'
        };
    }
    return {rates: pkt.dn_rates || pkt.descending?.dn_rates || null, unavailable: {}, missingReason: null};
}
window.neuroflyGraphDnReadout = graphDnReadout;

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
    // WP5 provenance: the engineered-assistance gate and the resolved optomotor IO map.
    const opto = motor.optomotor || {};
    const assistance = opto.engineered_assistance_enabled;
    set('identAssistance', assistance === undefined || assistance === null ? '--' : (assistance ? 'ON' : 'OFF'),
        assistance ? `Engineered assistance ON: ${(opto.engineered_assistance_applied || []).join('; ') || 'inputs that bypass the sensory pathways'}`
                   : 'Engineered assistance OFF: only the verified sensory encoder drives the graph.');
    const ioMap = opto.optomotor_io_map_sha256;
    set('identOptoMap', ioMap ? String(ioMap).slice(0, 12) : '--',
        ioMap ? `optomotor_io_map_sha256 ${ioMap}` : 'No optomotor IO map resolved for this run.');
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
    if (opto.optomotor_unsupported) lines.push(`OPTOMOTOR MAPPING UNSUPPORTED: ${opto.optomotor_unsupported}`);
    if (pkt?.paradigm === 'optomotor' && GRAPH_BACKENDS.includes(id.backend)) {
        const achieved = pkt?.timing?.achieved_speed;
        const measured = Number.isFinite(achieved) ? `${achieved} sim s per wall s measured` : 'measuring';
        lines.push(`OPTOMOTOR ON THE GRAPH: 1x real time is unattainable on this computer `
            + `(about ${OPTOMOTOR_SIM_S_PER_WALL_S} simulated s per wall s; ${measured}). `
            + `The daemon runs slower than requested instead of dropping simulation steps.`);
        if (opto.forward_drive) lines.push(`FORWARD DRIVE: ${opto.forward_drive}`);
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
        // Commands the daemon answered "queued" (a long step was running): resolved when
        // their acknowledgement arrives in a stream frame (``command_acks``).
        this.pendingCommands = new Map();
        this.commandAckCache = new Map(); // SSE may beat the HTTP queued reply.
        this.commandAckTimeoutMs = 120000;
        this.lastHeartbeat = null;          // {step_in_progress_s, last_step_wall_s, at}
        this.daemonHalt = null;             // {error, detail} while a step error halts the daemon
        // Page watchdog (audit F, F4): when the step last INCREASED, independent of
        // whether frames keep arriving.  A dead simulation thread behind a live HTTP
        // server shows old frames forever; only the step age tells it apart.
        this.lastStepAdvanceTime = 0;
        this.lastStepSeen = null;
        this.stepKey = null;
        this.daemonLiveness = null;         // {state, last_advance_age_s, ...} from frames/heartbeats
        this.daemonPersistence = null;      // {state, reason, last_ok_save_age_s, ...}
        this.daemonRecordingError = null;
        this.daemonValidity = null;         // result_validity: {state, mode, incidents, other_runs_incomplete}
        this.observationValidityUpdate = null; // owner-qualified downgrade; never a simulated frame
        this.daemonMode = null;             // "scientific" | "exploratory"
        this.daemonPaused = false;
        this.daemonError = null;            // status error text (dead/stalled) from heartbeats
        this.rejectedIdentityPackets = 0;
        this.lastIdentityRejection = null;
        this.manifest = null;               // full run manifest (GET /api/manifest), for exports
        this.manifestRunId = null;
        this.injectPending = NEUROFLY_INJECT;

        this.addressBadge = document.getElementById('daemonAddress');
        this.offeredUrl = null;   // a daemon that answered but was not chosen for this page
        const reloadBuild = document.getElementById('btnReloadBuild');
        if (reloadBuild) reloadBuild.addEventListener('click', () => window.location.reload());
        if (this.statusPill) {
            this.statusPill.addEventListener('click', () => {
                // Clicking a found-but-unchosen daemon is the explicit choice: record it in
                // the address bar (?daemon=) so a reload keeps the same daemon.
                if (!this.connected && this.offeredUrl) this.chooseDaemon(this.offeredUrl);
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
     * Which daemons this page may use.  ``auto`` daemons are connected without asking:
     *   1. `?daemon=http://host:port` (explicit; nothing else is ever tried),
     *   2. the page's own origin (the dashboard served by the daemon itself),
     *   3. the documented observatory pair: a page on port 8780 uses port 8781 on the
     *      same host (scripts/observatory.py).
     * ``offer`` daemons (the default port 8769 on this host or localhost) are probed but
     * NEVER connected silently: a static page on some other port would otherwise attach
     * to whatever daemon happens to hold 8769.  The pill offers them; a click connects.
     */
    get daemonCandidates() {
        const auto = [], offer = [];
        const add = (list, u) => { if (u && !auto.includes(u) && !offer.includes(u)) list.push(u); };
        const port = this.daemonPort;
        if (typeof window !== 'undefined' && window.location) {
            const loc = window.location;
            try {
                const param = new URLSearchParams(loc.search || '').get('daemon');
                if (param === 'off') return {auto, offer};
                // An explicit endpoint must never fall through to a different live lab.
                if (param) return {auto: [param.replace(/\/+$/, '')], offer};
            } catch (e) {}
            const httpLike = loc.protocol === 'http:' || loc.protocol === 'https:';
            if (httpLike && loc.origin && loc.origin !== 'null') add(auto, loc.origin);
            if (httpLike && loc.hostname && loc.port === '8780') add(auto, `${loc.protocol}//${loc.hostname}:8781`);
            if (httpLike && loc.hostname) add(offer, `${loc.protocol}//${loc.hostname}:${port}`);
        }
        add(offer, `http://localhost:${port}`);
        add(offer, `http://127.0.0.1:${port}`);
        return {auto, offer};
    }

    /** Back-compatible: the daemons this page connects to without asking. */
    get candidateUrls() {
        return this.daemonCandidates.auto;
    }

    chooseDaemon(url) {
        try {
            const next = new URL(window.location.href);
            next.searchParams.set('daemon', url);
            window.history.replaceState(null, '', next.toString());
        } catch (e) {}
        this.offeredUrl = null;
    }

    async probeDaemon(url) {
        try {
            const res = await fetch(`${url}/api/status`, {method: 'GET', signal: AbortSignal.timeout(4000)});
            if (!res.ok) return null;
            const status = await res.json();
            // A daemon that answers is connected to, whatever its health: a halted,
            // stalled, dead-loop or not-saving daemon must be SHOWN as such (the pill and
            // banner say so), never hidden behind "disconnected" or the local preview.
            const known = ['online', 'degraded', 'error'];
            return status && known.includes(status.status) && Number.isFinite(status.total_steps) ? status : null;
        } catch (e) {
            return null;
        }
    }

    showDaemonAddress(status) {
        const badge = this.addressBadge;
        if (!badge) return;
        if (this.connected && this.activeUrl) {
            let shown = this.activeUrl;
            try { shown = new URL(this.activeUrl).host; } catch (e) {}
            const compute = status?.compute?.detail ? ` Brain compute: ${status.compute.detail}.` : '';
            badge.textContent = `daemon ${shown}`;
            badge.style.color = '#4ade80';
            badge.title = `Connected to the NeuroFly daemon at ${this.activeUrl}`
                + (status?.version ? ` (version ${status.version}).` : '.')
                + (status ? ` Initial connection snapshot: backend ${status.backend || '?'}.` : '')
                + compute + ' Choose another with ?daemon=http://host:port.';
        } else if (this.offeredUrl) {
            badge.textContent = `daemon found: ${this.offeredUrl} (not connected)`;
            badge.style.color = '#fbbf24';
            badge.title = `A NeuroFly daemon answers at ${this.offeredUrl}, but this page was not served by it `
                + `and no ?daemon= was given, so it is not used automatically. Click the status pill to connect `
                + `to it, or open this page with ?daemon=<url> to choose a daemon.`;
        } else {
            badge.textContent = 'no daemon';
            badge.style.color = '#94a3b8';
            badge.title = 'Not connected to a daemon. Start one with `neurofly run` and open the URL it prints, '
                + 'or add ?daemon=http://host:port to this page.';
        }
    }

    renderDeliveryIdentity(status) {
        const meta = document.querySelector('meta[name="neurofly-web-build"]');
        const state = deliveryBuildState(meta?.content, status?.delivery);
        const revision = status?.delivery?.revision || 'unavailable';
        const dirty = status?.delivery?.source_dirty === true ? '+dirty' : '';
        const badge = document.getElementById('deliveryBadge');
        if (badge) {
            badge.textContent = `page ${state.page} · daemon revision ${revision}${dirty}`;
            badge.title = `Page build ${state.page}; daemon web build ${state.daemon || 'not reported'}; daemon revision ${revision}${dirty}`;
            badge.style.color = state.stale ? '#fbbf24' : '#94a3b8';
        }
        const compute = status?.compute || {};
        const device = document.getElementById('identDevice');
        if (device) {
            const deviceName = compute.device || 'unknown';
            device.textContent = `Initial connection snapshot: ${compute.gpu ? `${deviceName} · ${compute.gpu}` : deviceName}`;
            device.title = `Initial connection snapshot: ${compute.detail || compute.error || `Daemon compute device: ${deviceName}`}`;
            device.style.color = compute.error ? '#f87171' : '#e2e8f0';
        }
        const banner = document.getElementById('deliveryBanner');
        const text = document.getElementById('deliveryBannerText');
        if (banner && text) {
            text.textContent = state.stale
                ? `STALE DASHBOARD ASSETS: this tab has page build ${state.page}, while daemon ${revision} serves ${state.daemon}. Reload this page; the daemon run, brain, assay, pause state and requested speed stay unchanged.`
                : '';
            banner.style.display = state.stale ? 'flex' : 'none';
        }
        return state;
    }

    renderTiming(source) {
        if (this.replayMode) { window.neuroflyReplay?.updateButtons(); return; }
        if (Number.isFinite(source?.sim_speed)) this.hud.reconcileRequestedSpeed(source.sim_speed);
        const achievedEl = document.getElementById('statAchieved');
        if (achievedEl?.parentElement) achievedEl.parentElement.title = 'Simulation speed the daemon actually achieved (measured), versus the requested speed';
        const timing = source?.timing;
        if (!achievedEl || !timing || !Number.isFinite(timing.achieved_speed)) {
            if (achievedEl) achievedEl.textContent = '--';
            return;
        }
        achievedEl.textContent = !this.replayMode && this.daemonHalt ? '0x · halted'
            : source.paused ? 'paused' : formatSimSpeed(timing.achieved_speed);
        achievedEl.style.color = timing.overloaded ? '#fbbf24' : '';
        const dropped = this.streamStats ? ` Display decimation: ${this.streamStats.decimated_snapshots} snapshots skipped in the last second (latest-value-wins).` : '';
        achievedEl.title = `Requested ${timing.requested_speed}x, measured ${timing.achieved_speed}x `
            + `(= ${timing.achieved_speed} simulated seconds per wall-clock second); `
            + `fixed dt ${timing.integration_dt_s} s; ${timing.steps_in_frame ?? '?'} steps in this frame`
            + (timing.last_step_wall_s > 0.5 ? `; one step takes about ${timing.last_step_wall_s.toFixed(1)} s of wall time` : '') + '.'
            + (timing.overloaded ? ' This computer cannot run the requested speed; the daemon runs slower instead of skipping steps.' : '') + dropped;
    }

    scheduleReconnect() {
        if (this.reconnectTimer) clearTimeout(this.reconnectTimer);
        if (this.replayMode) return;   // a recording is playing; exitReplay() reconnects
        this.reconnectTimer = setTimeout(() => this.initConnection(false), this.reconnectDelayMs);
        this.reconnectDelayMs = Math.min(30000, Math.round(this.reconnectDelayMs * 1.5));
    }

    async initConnection(force = false) {
        if (this.reconnectTimer) clearTimeout(this.reconnectTimer);
        if (this.replayMode) return;
        if (this.probing && !force) return;
        this.probing = true;
        if (force) this.reconnectDelayMs = 4000;
        let foundUrl = null;
        const {auto, offer} = this.daemonCandidates;

        for (const url of auto) {
            const status = await this.probeDaemon(url);
            if (status) {
                foundUrl = url;
                this.activeUrl = url;
                this.offeredUrl = null;
                this.onDaemonConnected(status);
                break;
            }
        }
        if (!foundUrl) {
            this.offeredUrl = null;
            for (const url of offer) {
                if (await this.probeDaemon(url)) { this.offeredUrl = url; break; }
            }
        }
        this.probing = false;

        if (!foundUrl) {
            this.onDaemonDisconnected();
            this.scheduleReconnect();
        }
    }

    onDaemonConnected(status) {
        this.arena.cancelPreviewGust?.(true);
        this.connected = true;
        this.readOnly = false;
        this.reconnectDelayMs = 4000;
        this.lastPacketTime = performance.now();
        this.lastOrderedPacket = null;
        this.lastBackendAck = null;
        this.lastTrailStep = -1;
        this.showingStale = false;
        this.freshnessState = 'live';
        // Start the step clock from the status probe; a frozen daemon is then caught
        // even if it never sends a frame with a newer step.
        this.stepKey = null;
        this.lastStepSeen = null;
        this.noteStep(status.total_steps, 'probe');
        this.daemonLiveness = status.liveness ? {...status.liveness, at: performance.now()} : null;
        this.daemonPersistence = status.persistence || null;
        this.daemonRecordingError = status.recording_error || null;
        this.daemonValidity = status.result_validity || null;
        this.observationValidityUpdate = status.observation_validity_update || null;
        this.daemonMode = status.mode || null;
        this.daemonPaused = !!status.paused;
        this.daemonError = status.error || null;
        this.renderDeliveryIdentity(status);
        this.renderTiming(status);
        // Connecting during a fault: start from the daemon's own account of it.
        this.healthStep = null;
        this.daemonHalt = status.halted ? {error: status.error || 'unknown error', detail: status.error_detail || null}
            : null;
        this.simProcessDown = null;
        this.applySimProcessLiveness(status);
        if (this.statusPill) {
            this.statusPill.textContent = '● LIVE DAEMON';
            this.statusPill.style.background = 'rgba(34, 197, 94, 0.25)';
            this.statusPill.style.border = '1px solid #22c55e';
            this.statusPill.style.color = '#4ade80';
            this.statusPill.title = this.connectedPillTitle = `Connected to the learning daemon at ${this.activeUrl}`;
        }
        this.showDaemonAddress(status);
        const versionBadge = document.getElementById('appVersionBadge');
        if (versionBadge && status.version) versionBadge.textContent = `v${status.version}`;
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
        // The probe already tells us about a pause, halt or dead loop. Apply it
        // before the first paint instead of briefly advertising a fault as LIVE.
        this.updateFreshness();
        this.startStreaming();
    }

    onDaemonDisconnected(reason = '') {
        if (this.replayMode) return;
        this.connected = false;
        this.readOnly = false;
        this.disconnectReason = reason;
        // Keep the last scientific frame still during an outage; never silently
        // substitute a locally generated trajectory into a daemon recording.
        // remoteDriven stays as it was, so labels, units and measured metrics of the
        // last frame remain on screen instead of switching to local preview values.
        this.arena.awaitingDaemon = !!this.activeUrl;
        if (!this.activeUrl) this.arena.remoteDriven = false;
        if (!this.arena.awaitingDaemon && !this.arena.remoteDriven) {
            const label = document.getElementById('arenaRunState');
            if (label) label.textContent = 'Standalone preview · local engine';
        }
        if (this.statusPill) {
            this.statusPill.textContent = this.arena.awaitingDaemon ? '○ DISCONNECTED · FROZEN VIEW'
                : this.offeredUrl ? '○ LOCAL ENGINE · DAEMON FOUND, CLICK TO CONNECT' : '○ LOCAL ENGINE';
            this.statusPill.style.background = 'rgba(148, 163, 184, 0.15)';
            this.statusPill.style.border = '1px solid #64748b';
            this.statusPill.style.color = '#94a3b8';
            this.statusPill.title = this.arena.awaitingDaemon
                ? `Daemon connection lost${reason ? ' (' + reason + ')' : ''}. The last measured frame (step ${this.lastOrderedPacket?.step ?? '?'}) is kept unchanged; no local data replaces it. Click to retry now.`
                : this.offeredUrl
                    ? `In-browser simulation engine active. A daemon answers at ${this.offeredUrl}, but this page was not served by it, so it is not used automatically. Click to connect to it.`
                    : `In-browser simulation engine active (offline/standalone mode${reason ? ': ' + reason : ''}). Click to retry the daemon connection, or open the page with ?daemon=http://host:${this.daemonPort}.`;
        }
        this.showDaemonAddress(null);
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
        if (this.replayMode) {
            if (ageEl) { ageEl.textContent = 'replay'; ageEl.style.color = '#c084fc'; }
            const stepAgeEl = document.getElementById('statStepAge');
            if (stepAgeEl) { stepAgeEl.textContent = 'replay'; stepAgeEl.style.color = '#c084fc'; }
            return;
        }
        const age = this.lastValidDataTime ? (performance.now() - this.lastValidDataTime) / 1000 : null;
        if (ageEl) {
            ageEl.textContent = age === null ? '--' : `${age < 10 ? age.toFixed(1) : Math.round(age)}s${this.connected ? '' : ' (frozen)'}`;
            ageEl.style.color = age === null ? '' : (!this.connected ? '#f87171' : age * 1000 > this.staleAfterMs ? '#fbbf24' : '#4ade80');
        }
        const stepAge = this.stepAgeSeconds();
        const stepAgeEl = document.getElementById('statStepAge');
        this.updatePersistenceBanner();
        if (!this.connected || !this.statusPill || (age === null && stepAge === null)) {
            if (stepAgeEl && !this.connected) { stepAgeEl.textContent = '--'; stepAgeEl.style.color = ''; }
            return;
        }
        const stale = age !== null && age * 1000 > this.staleAfterMs;
        // Old data while the daemon reports a step still running is a slow computer,
        // not a lost connection: say so instead of "stale".
        const slowStep = this.slowStepSeconds();
        const halt = this.daemonHalt;
        const stopped = this.notAdvancing(stepAge, slowStep);
        const paused = this.daemonPaused || this.daemonLiveness?.state === 'paused';
        const state = halt ? 'error' : stopped ? 'stopped' : (stale && slowStep !== null) ? 'slow'
            : paused ? 'paused' : stale ? 'stale' : 'live';
        if (stepAgeEl) {
            stepAgeEl.textContent = stepAge === null ? '--' : paused && !stopped && !halt ? 'paused'
                : `${stepAge < 10 ? stepAge.toFixed(1) : Math.round(stepAge)}s`;
            stepAgeEl.style.color = stopped || halt ? '#f87171' : paused ? '#94a3b8'
                : stepAge !== null && stepAge > 3 ? '#fbbf24' : '#4ade80';
        }
        const ro = this.readOnly ? ' (READ-ONLY)' : '';
        if (state === 'stopped') {
            // Never LIVE while the step does not advance (audit F, the "static page").
            const n = Math.round(stepAge ?? this.daemonLiveness?.last_advance_age_s ?? 0);
            const live = this.daemonLiveness;
            this.statusPill.textContent = `● SIMULATION NOT ADVANCING${ro} · ${n}s`;
            const why = this.daemonError || halt?.error || (live?.state === 'dead'
                ? 'the simulation thread in the daemon has stopped' : `no new step for ${n} s`);
            const fix = live?.state === 'dead'
                ? 'Select an assay to rebuild and restart the simulation, or restart the daemon.'
                : 'Not paused and not halted, yet the step counter has not moved. If it does not recover, '
                  + 'select an assay or restart the daemon (it resumes from the last checkpoint).';
            this.statusPill.title = `The daemon answers but the simulation is not advancing (step `
                + `${this.lastStepSeen ?? '?'}, unchanged for ${n} s): ${why}. ${fix}`;
            // The last frame's achieved speed is history, not the present.
            const achievedEl = document.getElementById('statAchieved');
            if (achievedEl) { achievedEl.textContent = '0x'; achievedEl.style.color = '#f87171'; }
        }
        if (state === 'paused') {
            this.statusPill.textContent = `● DAEMON CONNECTED${ro} · PAUSED`;
            this.statusPill.title = 'The daemon is connected and paused: the simulation does not advance until you resume it.';
        }
        if (state === 'error') {
            this.renderCurrentHalt();
            // Connected and fresh, but the simulation does not advance: never show LIVE.
            const cls = halt.detail?.failure_class;
            this.statusPill.textContent = this.simProcessDown
                ? `● SIMULATION PROCESS ${this.simProcessDown.state === 'sim_process_down' ? 'DOWN' : 'NOT RESPONDING'}${ro}`
                : cls === 'compute' ? `● SIMULATION HALTED${ro} · GPU/COMPUTE ERROR`
                : (cls === 'persistence' && halt.detail?.channel) ? `● SIMULATION STOPPED${ro} · NOT SAVED`
                : `● SIMULATION HALTED${ro} · ERROR`;
            this.statusPill.title = this.simProcessDown
                ? `The web page server answers, but the simulation process is not running (${halt.error}). `
                    + `Nothing advances and commands are disabled. The last frame shown belongs to the stopped run.`
                : `The daemon is connected but the simulation is halted and not advancing: `
                + `${halt.error}` + (halt.detail?.paradigm ? ` (assay ${halt.detail.paradigm}, step ${halt.detail.step}). ` : '. ')
                + (halt.detail?.recover || 'Select an assay to rebuild the controller and resume.');
        }
        if (state === 'slow') {
            this.statusPill.textContent = `● LIVE DAEMON${ro} · STEP RUNNING ${Math.round(slowStep)}s`;
            this.statusPill.title = `The daemon is connected and computing: the current simulation step has run for `
                + `${slowStep.toFixed(1)} s (the last one took ${this.lastHeartbeat.last_step_wall_s?.toFixed?.(1) ?? '?'} s). `
                + `This computer runs the simulation slower than real time; no steps are skipped.`;
            if (ageEl) ageEl.style.color = '#38bdf8';
        }
        const liveText = `● LIVE DAEMON${ro}${this.daemonMode === 'exploratory' ? ' · EXPLORATORY' : ''}`;
        if (state === this.freshnessState && !['slow', 'error', 'stopped', 'stale'].includes(state)
                && !(state === 'live' && this.statusPill.textContent !== liveText)) return;
        this.freshnessState = state;
        this.showingStale = state === 'stale';
        if (state === 'live') {
            this.statusPill.textContent = liveText;
            if (this.connectedPillTitle && !this.readOnly) this.statusPill.title = this.connectedPillTitle;
        }
        if (state === 'stale') {
            // Connected, but no new frame: not "LIVE" (the step may not be advancing).
            this.statusPill.textContent = `● DAEMON CONNECTED${ro} · NO NEW DATA ${Math.round(age)}s`;
            this.statusPill.title = `The daemon is connected but has sent no new frame for ${Math.round(age)} s. `
                + `The last step seen is ${this.lastStepSeen ?? '?'}.`;
        }
        const color = {live: ['#4ade80', '#22c55e'], stale: ['#fbbf24', '#f59e0b'], slow: ['#38bdf8', '#0ea5e9'],
                       error: ['#f87171', '#ef4444'], stopped: ['#f87171', '#ef4444'],
                       paused: ['#cbd5e1', '#64748b']}[state];
        this.statusPill.style.background = (state === 'error' || state === 'stopped') ? 'rgba(239, 68, 68, 0.25)'
            : state === 'paused' ? 'rgba(148, 163, 184, 0.18)' : 'rgba(34, 197, 94, 0.25)';
        this.statusPill.style.color = color[0];
        this.statusPill.style.border = `1px solid ${color[1]}`;
    }

    /** Seconds since the daemon's step last increased (frames, heartbeats or status), else null. */
    stepAgeSeconds() {
        return this.lastStepAdvanceTime ? (performance.now() - this.lastStepAdvanceTime) / 1000 : null;
    }

    /** Record a step reported by the daemon; the clock restarts only when it increases
     *  (or the run/assay changes), never merely because a frame arrived. */
    noteStep(step, key) {
        if (!Number.isFinite(step)) return;
        // A new run, assay or activation starts a new step context (the counter may
        // even go down after a daemon restart); the status probe has no context.
        const newContext = !!(key && this.stepKey && key !== 'probe' && this.stepKey !== 'probe'
                              && key !== this.stepKey);
        if (newContext || this.lastStepSeen === null || step > this.lastStepSeen) {
            this.lastStepAdvanceTime = performance.now();
            this.lastStepSeen = step;
        }
        if (key !== 'probe' || this.stepKey === null) this.stepKey = key;
    }

    /** F4: the simulation is not advancing.  The daemon says so (stalled/dead), or,
     *  as a fallback for daemons without a watchdog, the step has not increased for
     *  max(10 s, 20 steps at the requested speed) while not paused, halted or slow. */
    notAdvancing(stepAge, slowStep) {
        const live = this.daemonLiveness;
        if (live && (live.state === 'stalled' || live.state === 'dead')) return true;
        if (stepAge === null || this.daemonHalt || this.daemonPaused || slowStep !== null) return false;
        if (live && ['paused', 'halted', 'slow'].includes(live.state)) return false;
        const speed = Number(this.lastOrderedPacket?.sim_speed) || 1;
        const threshold = Math.max(10, 20 * 0.02 / Math.max(speed, 1e-3), 3 * (this.lastHeartbeat?.last_step_wall_s || 0));
        return stepAge > threshold;
    }

    /** Save-policy banner (Codex direction, decision 3):
     *  - red "STOPPED · RESULT INCOMPLETE" when a required save stopped the run (scientific mode);
     *  - amber "EXPLORATORY · NOT SAVING" while an exploratory run continues unsaved;
     *  - amber "NOT SAVING" for a diagnostic log only;
     *  - amber "RESULT INCOMPLETE" after recovery: the failure stays on the run's record. */
    updatePersistenceBanner() {
        const el = document.getElementById('persistenceBanner');
        if (!el) return;
        const p = this.connected ? this.daemonPersistence : null;
        const rec = this.connected ? this.daemonRecordingError : null;
        const v = this.connected ? this.daemonValidity : null;
        const halt = this.daemonHalt;
        const incidents = v?.incidents || [];
        const incomplete = v?.state === 'incomplete';
        const earlier = (v?.other_runs_incomplete || []).length;
        const ago = (s) => s === null || s === undefined ? 'not yet in this run'
            : s < 90 ? `${Math.round(s)} s ago` : s < 5400 ? `${Math.round(s / 60)} min ago` : `${(s / 3600).toFixed(1)} h ago`;
        const reason = p && p.ok === false ? (p.state === 'disk_low' ? 'disk almost full' : (p.reason || 'write failed')) : null;
        let text = null, red = false;
        if (halt && halt.detail?.failure_class === 'persistence' && halt.detail?.channel) {
            red = true;
            text = halt.detail.channel === 'recording'
                ? `STOPPED · RESULT INCOMPLETE: requested recording failed at step ${halt.detail.step}. `
                    + (halt.detail.recover || 'Repair storage and restart with a new recording name using the same saved-brain output directory. The failed prefix is kept.')
                : `STOPPED · RESULT INCOMPLETE: a required save failed (${reason || halt.detail.channel}) at step `
                    + `${halt.detail.step}. The last good checkpoint and the in-memory state are kept. Free disk space, `
                    + `then select the assay: it saves first and resumes only if that works.`;
        } else if (halt && halt.detail?.failure_class === 'compute') {
            red = true;
            text = `HALTED · RESULT INCOMPLETE: GPU/compute error at step ${halt.detail.step} (${halt.error}). `
                + `Nothing is saved over the last good checkpoint. Select an assay to rebuild, or restart the daemon.`;
        } else if (reason && this.daemonMode === 'exploratory' && Object.values(p.failing || {}).some(c => c.required)) {
            text = `EXPLORATORY · NOT SAVING: ${reason} (last saved ${ago(p.last_ok_save_age_s)}). The run continues `
                + `unsaved; the gap is recorded and this run is marked INCOMPLETE.`;
        } else if (rec) {
            text = `RECORDING INVALID: ${rec.message || 'the recording stopped'}`;
        } else if (reason) {
            text = Object.values(p.failing || {}).some(c => c.required)
                ? `REQUIRED SAVE UNRESOLVED: ${reason}. Check the current recording and recovery status.`
                : `NOT SAVING (diagnostic log only): ${reason}. The scientific record is unaffected.`;
        } else if (incomplete) {
            const last = incidents[incidents.length - 1] || {};
            text = `RESULT INCOMPLETE: this run had ${incidents.length} failure(s); the last at step ${last.step} `
                + `(${last.reason}${last.recovered_at && Number.isSafeInteger(last.recovered_step) && last.recovered_step >= 0 ? ', recovered at step ' + last.recovered_step : ''}). `
                + `The failure stays on the run's record.`;
        } else if (earlier) {
            text = `${earlier} earlier run(s) in this daemon are marked INCOMPLETE (see /api/status result_validity).`;
        }
        if (!text) {
            if (el.style.display !== 'none') { el.style.display = 'none'; el.textContent = ''; }
            return;
        }
        el.textContent = text;
        el.title = [p?.summary, rec?.message, ...incidents.map(i => `${i.reason} @ step ${i.step}: ${i.error || ''}`)]
            .filter(Boolean).join(' | ');
        el.style.background = red ? 'rgba(239,68,68,0.16)' : 'rgba(245,158,11,0.18)';
        el.style.color = red ? '#fca5a5' : '#fbbf24';
        el.style.borderBottom = red ? '1px solid #7f1d1d' : '1px solid #b45309';
        el.style.display = 'block';
    }

    /** Seconds the daemon's current step has run, from a recent heartbeat, else null. */
    slowStepSeconds() {
        const beat = this.lastHeartbeat;
        if (!beat || !(beat.step_in_progress_s > 0)) return null;
        const since = (performance.now() - beat.at) / 1000;
        return since < 2.5 ? beat.step_in_progress_s + since : null;
    }

    /** Resolve commands answered "queued" once their acknowledgement is in a frame. */
    resolveCommandAcks(acks) {
        if (!Array.isArray(acks)) return;
        for (const ack of acks) {
            if (!ack?.command_id || !['ok', 'error'].includes(ack.status)) continue;
            for (const key of [ack.command_id, ack.client_command_id].filter(Boolean)) this.commandAckCache.set(key, ack);
            while (this.commandAckCache.size > 50) this.commandAckCache.delete(this.commandAckCache.keys().next().value);
            const pending = this.pendingCommands.get(ack.command_id)
                || (ack.client_command_id ? this.pendingCommands.get(ack.client_command_id) : null);
            if (pending) pending.resolve(ack);
            this.hud?.backendCommandAck?.(ack);
        }
    }

    /** Wait for a final acknowledgement; queued is never an applied result. */
    awaitCommandAck(commandId, clientCommandId = null) {
        const keys = [commandId, clientCommandId].filter(Boolean);
        const cached = keys.map(key => this.commandAckCache.get(key)).find(Boolean);
        if (cached) {
            for (const key of keys) this.commandAckCache.delete(key);
            return Promise.resolve(cached);
        }
        return new Promise((resolve) => {
            const done = (value) => {
                clearTimeout(timer);
                for (const key of keys) { this.pendingCommands.delete(key); this.commandAckCache.delete(key); }
                resolve(value);
            };
            const timer = setTimeout(() => done({status: 'error', command_id: commandId, client_command_id: clientCommandId,
                timed_out: true,
                message: 'Timed out waiting for the final acknowledgement; the command outcome is unknown.'}),
                this.commandAckTimeoutMs);
            for (const key of keys) this.pendingCommands.set(key, {resolve: done});
        });
    }

    /** A request id the daemon echoes on its reply and final acknowledgement. */
    newClientCommandId() {
        return `nf-${Date.now().toString(36)}-${Math.random().toString(36).slice(2, 10)}`;
    }

    /** Read-only: the daemon's record of one request (acknowledged/pending/queued/applying/unknown). */
    async lookupCommandAck(clientCommandId) {
        if (!this.connected || !this.activeUrl || !clientCommandId) return null;
        try {
            const res = await fetch(`${this.activeUrl}/api/command_ack?client_command_id=${encodeURIComponent(clientCommandId)}`,
                {method: 'GET', signal: AbortSignal.timeout(2000)});
            if (!res.ok) return null;
            return await res.json();
        } catch (e) { return null; }
    }

    /** A later reply for an older activation never replaces a newer acknowledgement. */
    static newerAck(previous, next) {
        const a = previous?.identity, b = next?.identity;
        if (!a || !b || a.daemon_run_id !== b.daemon_run_id) return true;
        return !(Number.isInteger(a.activation) && Number.isInteger(b.activation)) || b.activation >= a.activation;
    }

    acceptsHeartbeatOwner(beat) {
        const packet = this.lastOrderedPacket || this.arena.remotePacket;
        if (this.replayMode || this.switchPending || packet?.identity?.assay !== this.arena.activeParadigmId
                || (Object.prototype.hasOwnProperty.call(beat?.identity || {}, 'brain_id')
                    && beat.identity.brain_id !== beat.brain_id)
                || !window.NeuroFlyObservationRenderer.matchesObservationOwner(packet,
                    {...beat?.identity, brain_id: beat?.brain_id}, beat?.segment_id)
                || (beat.result_validity && beat.result_validity.run_id !== beat.identity.run_id)) return false;
        for (const ack of [this.lastSwitchAck, this.lastBackendAck, this.lastAck]) {
            if (identityRejection({identity: beat.identity, run_id: beat.identity.daemon_run_id}, ack)) return false;
        }
        return true;
    }

    /** Split mode (web process up, simulation process down or not answering): the web
     *  reports liveness sim_process_down / sim_process_unresponsive.  Show that, never the
     *  last run's PAUSED state; clear that run's identity and disable commands.  Returns
     *  true while the simulation process is down. */
    applySimProcessLiveness(source) {
        const state = source?.liveness?.state;
        if (state !== 'sim_process_down' && state !== 'sim_process_unresponsive') {
            if (this.simProcessDown) {
                // Back: the restarted process's own frames and status set identity and controls.
                this.simProcessDown = null;
                if (this.daemonHalt?.detail?.kind === 'simulation_process') this.daemonHalt = null;
            }
            return false;
        }
        const error = source.error || (state === 'sim_process_down' ? 'simulation process is down'
            : 'simulation process is not responding');
        this.simProcessDown = {state, error};
        this.daemonHalt = {error, detail: source.error_detail || {kind: 'simulation_process', message: error}};
        this.daemonPaused = false;
        this.daemonError = error;
        this.daemonLiveness = {...source.liveness, at: performance.now()};
        // Acknowledgements and the frame order belong to the process that is gone.
        this.lastSwitchAck = this.lastBackendAck = this.lastAck = null;
        const run = document.getElementById('identRun');
        if (run) { run.textContent = 'run --'; run.title = 'No simulation process: the previous run is no longer current.'; }
        const label = document.getElementById('arenaRunState');
        if (label) label.textContent = `SIMULATION PROCESS ${state === 'sim_process_down' ? 'DOWN' : 'NOT RESPONDING'} · commands disabled`;
        const pause = document.getElementById('btnPauseToggle');
        if (pause) { pause.textContent = 'Sim down'; pause.disabled = true; }
        return true;
    }

    // Health is current control-plane evidence, independent of the last pose frame.
    applyDaemonHealth(beat) {
        if (!this.acceptsHeartbeatOwner(beat)) return false;
        this.daemonHalt = beat.halted || beat.error
            ? {error: beat.error || 'unknown error', detail: beat.error_detail || null} : null;
        this.healthStep = beat.step;
        this.daemonPaused = !!beat.paused;
        this.renderPlaybackState({...this.lastOrderedPacket, paused: this.daemonPaused,
            error: this.daemonHalt?.error || null});
        this.renderCurrentHalt();
        return true;
    }

    currentHaltForPacket(pkt) {
        return !this.replayMode && this.daemonHalt && Number.isFinite(this.healthStep)
            && Number.isFinite(pkt.step) && pkt.step <= this.healthStep
            && this.acceptsHeartbeatOwner(this.lastHeartbeat);
    }

    renderCurrentHalt() {
        if (this.replayMode || !this.daemonHalt) return;
        const fault = document.getElementById('identFault');
        if (fault) { fault.textContent = this.daemonHalt.error; fault.style.color = '#f87171'; }
        const achieved = document.getElementById('statAchieved');
        if (achieved) { achieved.textContent = '0x · halted'; achieved.style.color = '#f87171'; }
    }

    applyObservationValidityUpdate(update) {
        const packet = this.lastOrderedPacket || this.arena.remotePacket;
        if (this.replayMode || this.switchPending || packet?.identity?.assay !== this.arena.activeParadigmId
                || !window.NeuroFlyObservationRenderer.matchesValidityUpdate(packet, update)) return false;
        for (const ack of [this.lastSwitchAck, this.lastBackendAck, this.lastAck]) {
            if (identityRejection({identity: update.identity, run_id: update.identity.daemon_run_id}, ack)) return false;
        }
        this.observationValidityUpdate = JSON.parse(JSON.stringify(update));
        return true;
    }

    reconcileObservationValidityUpdate(packet) {
        if (this.observationValidityUpdate
                && !window.NeuroFlyObservationRenderer.matchesValidityUpdate(packet, this.observationValidityUpdate)) {
            this.observationValidityUpdate = null;
        }
    }

    startStreaming() {
        if (!this.activeUrl) return;
        if (this.eventSource) this.eventSource.close();

        try {
            this.eventSource = new EventSource(`${this.activeUrl}/api/stream`);
            this.eventSource.onmessage = (event) => this.receive(event.data);
            // Liveness without new data (paused or slow daemon): keeps the stream open.
            this.eventSource.addEventListener('heartbeat', (event) => {
                this.lastPacketTime = performance.now();
                let beat;
                try { beat = {...JSON.parse(event.data), at: this.lastPacketTime}; }
                catch (e) { this.lastHeartbeat = null; return; }
                // Split mode: a down simulation process belongs to no run, so it is applied
                // before the run-owner fence and overrides any paused presentation.
                if (this.applySimProcessLiveness(beat)) { this.lastHeartbeat = beat; this.updateFreshness(); return; }
                if (beat.identity && !this.acceptsHeartbeatOwner(beat)) return;
                this.lastHeartbeat = beat;
                if (beat && beat.liveness) {
                    // Qualified heartbeats are fenced before any owner health is
                    // applied. Legacy liveness-only messages cannot change metrics.
                    // F3/F4: heartbeats say whether the simulation advances even when no
                    // frame is published (a dead loop publishes nothing).
                    this.noteStep(beat.step, this.stepKey);
                    this.daemonLiveness = {...beat.liveness, at: this.lastPacketTime};
                    this.daemonPersistence = beat.persistence || null;
                    this.applyDaemonHealth(beat);
                    this.daemonError = beat.error || null;
                    if (beat.result_validity) this.daemonValidity = beat.result_validity;
                    if (beat.mode) this.daemonMode = beat.mode;
                    this.applyObservationValidityUpdate(beat.observation_validity_update);
                    this.updateFreshness();
                }
            });
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
        this.resolveCommandAcks(pkt.command_acks);

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
        // A frame is published only by a running simulation process: it ends a sim-down
        // state even when no heartbeat follows (an advancing run sends frames, not heartbeats).
        if (this.simProcessDown) this.applySimProcessLiveness({liveness: pkt.liveness || {state: 'advancing'}});
        const step = Number.isFinite(pkt.step) ? pkt.step : null;
        this.lastPacketTime = performance.now();
        this.lastOrderedPacket = pkt;
        this.reconcileObservationValidityUpdate(pkt);
        // A step error halts the daemon (nothing advances) until a switch rebuilds it.
        // Frames still arrive, so without this the pill would read LIVE over a frozen run.
        const healthHalt = this.currentHaltForPacket(pkt);
        const halt = healthHalt ? this.daemonHalt : (pkt.halted || pkt.error) ? {error: pkt.error || 'unknown error', detail: pkt.error_detail || null} : null;
        const haltChanged = (halt?.error || null) !== (this.daemonHalt?.error || null);
        this.daemonHalt = halt;
        // Page watchdog (F4): the step clock moves only when the step increases.
        const advanced = step !== null && (this.lastStepSeen === null || step > this.lastStepSeen);
        this.noteStep(step, `${pkt.run_id}|${pkt.paradigm}|${pkt.identity?.activation ?? ''}`);
        if (!healthHalt) this.daemonPaused = !!pkt.paused;
        // A new stream first replays the last published frame.  Its liveness is history:
        // it must not overwrite a "dead"/"stalled" report unless the step has moved since.
        const reportedStopped = ['dead', 'stalled', 'halted'].includes(this.daemonLiveness?.state);
        if (pkt.liveness && (advanced || !reportedStopped)) this.daemonLiveness = {...pkt.liveness, at: performance.now()};
        if (!healthHalt) {
            this.daemonPersistence = pkt.persistence || null;
            this.daemonRecordingError = pkt.recording_error || null;
            if (pkt.result_validity) this.daemonValidity = pkt.result_validity;
            if (pkt.mode) this.daemonMode = pkt.mode;
            this.daemonError = pkt.error || null;
        }
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
        if (haltChanged) this.updateFreshness();

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
            this.renderPlaybackState(pkt);
            this.renderTiming(pkt);
            this.renderCurrentHalt();
            const panelCaps = graphPanelCapabilities(pkt);
            // The daemon retains a modular helper brain during graph runs, but it is
            // not the selected controller. Never copy those helper values into the
            // visible graph readouts.
            if (panelCaps.modularMemory) {
                this.arena.mb.kcFiring = pkt.neural?.kc_hz || this.arena.mb.kcFiring;
                if (Number.isFinite(pkt.neural?.net_valence)) this.arena.mb.netValence = pkt.neural.net_valence;
                this.arena.mb.pamRate = Number.isFinite(pkt.neural?.pam_trace) ? pkt.neural.pam_trace : 0;
                this.arena.mb.ppl1Rate = Number.isFinite(pkt.neural?.ppl1_trace) ? pkt.neural.ppl1_trace : 0;
            }
            if (pkt.neural?.epg_wedges && Array.isArray(pkt.neural.epg_wedges) && pkt.neural.epg_wedges.length > 0) {
                this.arena.cx.setRealEpgProfile(pkt.neural.epg_wedges);
            } else {
                this.arena.cx.updateBump(pkt.neural?.compass_heading ?? pkt.fly.heading);
            }
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
                this.arena.telemetryBuffer.push({source:pkt.timing?.replay ? 'recording' : 'daemon', run_id:pkt.run_id || '', brain_id:pkt.brain_id, segment,
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
            const state = this.arena.paradigmState, m = graphPanelCapabilities(pkt).graph ? {} : (pkt.metrics || {}), stimulus = pkt.stimuli || {}, assay = pkt.assay_state || {};
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
                if (activeParadigm==='optomotor') state.drumAngleDeg=Number.isFinite(pkt.scene?.drum_angle_deg)?pkt.scene.drum_angle_deg:null;
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
            const num = (v) => Number.isFinite(v) ? v : null;
            if (p) {
                p.compositeScore = num(m.composite_benchmark_score);
                p.coordinationScore = num(m.locomotor_coordination_index);
                p.sensoryIntegrationScore = num(m.multisensory_integration_score);
                p.efficiencyScore = num(m.biomechanical_efficiency);
                p.smoothnessScore = num(m.kinematic_smoothness);
                p.wallCollisions = Number.isFinite(m.wall_collisions) ? m.wall_collisions : p.wallCollisions;
                p.totalDistance = Number.isFinite(m.total_distance_mm) ? m.total_distance_mm : p.totalDistance;
                p.totalEnergy = Number.isFinite(m.total_energy_atp) ? m.total_energy_atp : p.totalEnergy;
            }
        }
        // Per-region activity and spike raster panel (web/replay.js), live and replay alike.
        window.neuroflyActivityPanel?.update(pkt);
    }

    /** Playback controls belong to the active player, not historical frame pause. */
    renderPlaybackState(pkt) {
        if (this.replayMode) { window.neuroflyReplay?.updateButtons(); return; }
        const currentHalt = this.currentHaltForPacket(pkt);
        const error = currentHalt ? this.daemonHalt.error : pkt.error;
        const paused = currentHalt ? this.daemonPaused : pkt.paused;
        this.arena.paradigmStatus = error ? `SIMULATION ERROR: ${error}` : paused ? 'PAUSED' : pkt.brain?.teaching ? 'CUE TEACHING · ARENA PAUSED' : `${pkt.fly.state} · ${pkt.continuous ? 'CONTINUOUS OBSERVATION' : 'TRIAL ' + pkt.trial}`;
        const phaseLabel = document.getElementById('arenaRunState');
        if (phaseLabel) phaseLabel.textContent = this.arena.paradigmStatus;
        const pause = document.getElementById('btnPauseToggle');
        if (pause) {
            const halted = !this.replayMode && !!error;
            pause.textContent = halted ? 'Halted' : paused ? 'Resume' : 'Pause';
            pause.disabled = halted;
        }
    }

    /** Stop live delivery; recording frames drive the same panels through handleDaemonPacket. */
    enterReplay(label) {
        this.arena.cancelPreviewGust?.(true);
        this.replayMode = true;
        window.dispatchEvent(new Event('neurofly-replay-mode-change'));
        if (this.reconnectTimer) clearTimeout(this.reconnectTimer);
        if (this.eventSource) {
            this.eventSource.onerror = null;
            this.eventSource.onmessage = null;
            this.eventSource.close();
            this.eventSource = null;
        }
        this.connected = false;
        this.resetReplayView();
        if (this.statusPill) {
            this.statusPill.textContent = `▶ REPLAY · ${label}`;
            this.statusPill.style.background = 'rgba(168, 85, 247, 0.2)';
            this.statusPill.style.border = '1px solid #a855f7';
            this.statusPill.style.color = '#d8b4fe';
            this.statusPill.title = 'Playing a recorded run at its recorded simulation time. Nothing is computed live; exit the replay to reconnect to the daemon.';
        }
    }

    /** Forget ordering state so a seek (including backwards) applies the next frame. */
    resetReplayView() {
        this.lastOrderedPacket = null;
        this.observationValidityUpdate = null;
        this.healthStep = null;
        this.lastSwitchAck = null;
        this.lastTrailStep = -1;
        this.arena.remoteSegment = null;
        if (this.arena.fly) this.arena.fly.trail = [];
    }

    exitReplay() {
        if (!this.replayMode) return;
        this.replayMode = false;
        window.dispatchEvent(new Event('neurofly-replay-mode-change'));
        this.resetReplayView();
        this.arena.remoteDriven = false;
        this.updateFreshness();
        this.initConnection(true);
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

    commandQueuedNotice(action) {
        const label = document.getElementById('arenaRunState');
        if (label) label.textContent = `${action.replaceAll('_', ' ')} queued: applied when the current simulation step finishes`;
    }

    async sendCommand(action, params = {}, onQueued = null, options = {}) {
        if (!this.connected || !this.activeUrl || this.readOnly || this.simProcessDown) return null;
        const switching = ['switch_paradigm', 'switch_backend'].includes(action);
        const clientCommandId = options.clientCommandId || this.newClientCommandId();
        if (switching) this.switchPending = true;
        try {
            const res = await fetch(`${this.activeUrl}/api/command`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ action, params, client_command_id: clientCommandId }),
                signal: AbortSignal.timeout(2000)
            });
            if (res.status === 403) {
                this.markReadOnly();
                return {status:'error', applied:false, message:'This dashboard is read-only; the command was not applied.'};
            }
            let data;
            try { data = await res.json(); }
            catch (e) {
                if (['TimeoutError', 'AbortError'].includes(e?.name)) throw e;
                data = {status:'error', message:`Daemon returned HTTP ${res.status} without a JSON acknowledgement.`};
            }
            if (res.ok || data) {
                // A long step was running: the command is queued and applied at the next
                // step boundary; its acknowledgement arrives in the stream.
                if (data.status === 'queued' && data.command_id) {
                    if (onQueued) onQueued(data);
                    else this.commandQueuedNotice(action);
                    data = await this.awaitCommandAck(data.command_id, clientCommandId);
                }
                if(data.status==='error') console.warn('[DaemonBridge] Command rejected:',data.message);
                // The daemon acknowledges the step at which the command took effect.
                if (data.ack) this.lastAck = {...data.ack, action};
                // A switch is acknowledged only after the target's brain and world are
                // ready; from now on older-identity packets are rejected.  A late reply
                // for an older activation never replaces a newer acknowledgement.
                if (switching && data.status === 'ok' && data.ack?.applied === true && data.ack?.identity
                        && DaemonBridgeClient.newerAck(this.lastSwitchAck, data.ack)) this.lastSwitchAck = data.ack;
                if (action === 'switch_backend' && data.ack?.identity
                        && DaemonBridgeClient.newerAck(this.lastBackendAck, data.ack)) this.lastBackendAck = data.ack;
                return data;
            }
        } catch (e) {
            console.warn('[DaemonBridge] sendCommand error:', e);
            const timedOut = ['TimeoutError', 'AbortError'].includes(e?.name);
            // No reply means no daemon command id: the request id lets the outcome be
            // matched later from the stream or GET /api/command_ack.
            return {status:'error', timed_out:timedOut, request_unanswered:true, client_command_id:clientCommandId,
                message:timedOut
                    ? 'The command request timed out; its outcome is unknown.'
                    : 'The command request failed; no final acknowledgement was received.'};
        } finally { if (switching) this.switchPending = false; }
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
        ref: 'Background: Budick & Dickinson (2006); Maimon et al. (2010) (citation not verified in this repository)',
        sliders: [
            { key: 'wallRepulsion', label: 'Boundary Repulsion', min: 0.2, max: 3.0, step: 0.1, val: 1.0, unit: 'x', desc: 'Gain of existing anticipatory wall-avoidance steering in the local preview.', apply: (a, v) => { a.wallRepulsion = v; } },
            { key: 'windStrength', label: 'Wind Vector Speed', min: 0.0, max: 40.0, step: 2.0, val: 15.0, unit: ' mm/s', desc: 'Preview wind vector, turned into an upwind steering term; no Johnston’s organ model is simulated.', apply: (a, v) => { a.cancelPreviewGust?.(); a.windVector[0] = -v; } }
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
        title: 'T-Maze Odour Choice',
        badge: 'ODOUR CHOICE',
        ref: 'Background: Tully & Quinn (1985) J Comp Physiol A 157:263–277 (abstract read); Dudai (1976) (citation not verified in this repository)',
        sliders: [],
        actions: [
            { label: 'Reset preview MB memory', class: '', handler: (a, h) => { a.mb.reset(false); } }
        ],
        metrics: [
            { label: 'Performance Index (PI)', get: (a) => `${(a.paradigmState && a.paradigmState.performanceIndex !== undefined ? a.paradigmState.performanceIndex : 0.0).toFixed(2)}` },
            { label: 'Preview reinforcement arms', get: () => 'A reward / B punishment' },
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
        title: 'Y-Maze Exploration',
        badge: 'EXPLORATION',
        ref: 'Background: Buchanan et al. (2015) is a handedness study, not an alternation study; Churgin (2017) (citation not verified in this repository)',
        sliders: [],
        actions: [
            { label: 'Clear choice history (keeps last SAR)', class: '', handler: (a, h) => { if (a.paradigmState) { a.paradigmState.turnDirections = []; a.paradigmState.armSequence = []; } } }
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
        title: 'Thermal Heat-Maze',
        badge: 'THERMAL',
        ref: 'Background: Ofstad, Zuker & Reiser (2011) Nature 474:204–207 (cited only for the existence of visual place learning)',
        sliders: [
            { key: 'floorTemp', label: 'Arena Floor Temperature', min: 32, max: 44, step: 1, val: 36.5, unit: ' °C', desc: 'Preview hot-floor temperature (°C). It sets the preview punishment signal; it does not guide the fly to the refuge.', apply: (a, v) => { if (a.paradigmState) a.paradigmState.hotTemp = v; } },
            { key: 'refugeRadius', label: 'Cool Refuge Radius', min: 6, max: 20, step: 1, val: 9, unit: ' mm', desc: 'Preview cool-tile radius (mm). The preview uses stored coordinates, not landmark triangulation.', apply: (a, v) => { if (a.paradigmState) a.paradigmState.refugeRadius = v; } }
        ],
        actions: [
            { label: 'Relocate Cool Refuge', class: 'primary', handler: (a, h) => {
                if (a.paradigmState) {
                    const angles = [Math.PI / 4, 3 * Math.PI / 4, 5 * Math.PI / 4, 7 * Math.PI / 4];
                    const choice = angles[Math.floor(Math.random() * angles.length)];
                    a.paradigmState.refugePos = [60.0 + Math.cos(choice) * 28.0, 60.0 + Math.sin(choice) * 28.0];
                }
            } }
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
        title: "Buridan's Paradigm",
        badge: 'VISION',
        ref: 'Background: Götz (1980); Colomb et al. (2012) (citation not verified in this repository)',
        sliders: [],
        actions: [
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
            ctx.fillText('Current preview heading relative to stripes', 8, 14);
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
        illustrativeChart: true,
        title: 'Visual Operant Drum (heat quadrants)',
        badge: 'CLOSED LOOP',
        ref: 'Background: Wolf & Heisenberg (1991); Liu et al. (2006) (citation not verified in this repository)',
        sliders: [],
        actions: [
            { label: 'Clear preview sector counters', class: '', handler: (a, h) => { if (a.paradigmState) { a.paradigmState.timeSafeMs = 0; a.paradigmState.timePunishedMs = 0; } } }
        ],
        metrics: [
            { label: 'Occupancy index (preview)', get: (a) => `${(a.paradigmState && Number.isFinite(a.paradigmState.learningIndex) ? a.paradigmState.learningIndex : 0).toFixed(2)}` },
            { label: 'Current Sector', get: (a) => `${a.paradigmState && a.paradigmState.laserActive ? 'PUNISHED (LASER ON)' : 'SAFE SECTOR'}` },
            { label: 'Laser Cumulative', get: (a) => `${((a.paradigmState && a.paradigmState.timePunishedMs) ? a.paradigmState.timePunishedMs / 1000 : 0.0).toFixed(1)}s` }
        ],
        drawChart: (ctx, w, h, a, hInst) => {
            ctx.fillStyle = '#f43f5e';
            ctx.font = '9px monospace';
            ctx.fillText('Illustrative sector diagram — not measured', 8, 14);
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
        illustrativeChart: true,
        controlNote: 'Wind vector speed and gust affect the drawing/readout; plume transport and steering remain fixed in this preview.',
        title: 'Wind Tunnel Plume',
        badge: 'ODOUR PLUME',
        ref: 'Background: Álvarez-Salvado et al. (2018) eLife (recorded, not re-read); Demir et al. (2020) (citation not verified in this repository)',
        sliders: [
            { key: 'windVelocity', label: 'Preview wind vector speed', min: 5, max: 40, step: 1, val: 18, unit: ' mm/s', desc: 'Sets the displayed wind vector and wind-speed readout; plume transport and steering use the existing fixed preview model.', apply: (a, v) => { a.cancelPreviewGust?.(); a.windVector[0] = -v; } },
            { key: 'plumeWidth', label: 'Gaussian Plume Width', min: 6, max: 30, step: 1, val: 14, unit: ' mm', desc: 'Width of the stationary preview plume. It changes where odour crosses the threshold that labels SURGE or CAST.', apply: (a, v) => { if (a.paradigmState) a.paradigmState.filamentSigma = v / 4.0; } }
        ],
        actions: [
            { label: 'Shift Plume Source', class: 'primary', handler: (a, h) => { if (a.paradigmState) a.paradigmState.nozzlePos[1] = 30.0 + (Math.random() - 0.5) * 30.0; } },
            { label: 'Turbulent Crosswind Gust', class: 'danger', handler: (a, h) => a.startPreviewGust('crosswind') },
            { label: 'Clear preview surge/cast counters', class: '', handler: (a, h) => { if (a.paradigmState) { a.paradigmState.surgeSteps = 0; a.paradigmState.castSteps = 0; } } }
        ],
        metrics: [
            { label: 'Surge/Cast Ratio', get: (a) => { const p = a.paradigmState || {}; return p.castSteps > 0 ? `${(p.surgeSteps / p.castSteps).toFixed(2)}x` : '0.00x'; } },
            { label: 'Upwind Progress', get: (a) => `${(a.paradigmState && Number.isFinite(a.paradigmState.upwindProgress) ? a.paradigmState.upwindProgress : 0).toFixed(1)} mm` },
            { label: 'Preview wind vector speed', get: (a) => `${Math.hypot(a.windVector[0], a.windVector[1]).toFixed(1)} mm/s` }
        ],
        drawChart: (ctx, w, h, a, hInst) => {
            ctx.fillStyle = '#38bdf8';
            ctx.font = '9px monospace';
            ctx.fillText('Illustrative surge/cast diagram — not measured', 8, 14);
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
        illustrativeChart: true,
        title: 'Looming Escape',
        badge: 'LOOMING',
        ref: 'Background: von Reyn et al. (2014) Nat Neurosci 17:962–970; Card & Dickinson (2008) (citation not verified in this repository)',
        sliders: [],
        actions: [
            { label: 'Clear preview escape flag', class: '', handler: (a, h) => { a.dn.escapeActive = false; } }
        ],
        metrics: [
            { label: 'Preview escape state', get: (a) => a.dn.escapeActive ? 'ACTIVE' : 'INACTIVE' },
            { label: 'Current preview heading', get: (a) => `${((a.fly.heading * 180) / Math.PI).toFixed(0)}°` },
            { label: 'Current preview speed', get: (a) => `${a.fly.speed.toFixed(1)} mm/s` }
        ],
        drawChart: (ctx, w, h, a, hInst) => {
            ctx.fillStyle = '#f43f5e';
            ctx.font = '9px monospace';
            ctx.fillText('Illustrative looming curve — not measured', 8, 14);
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
        illustrativeChart: true,
        title: 'Optomotor Drum',
        badge: 'MOTION',
        ref: 'Background: Götz (1964); Kim et al. (2017) on efference copy (citation not verified in this repository)',
        sliders: [
            { key: 'patternSpeed', label: 'Preview grating velocity', min: -120, max: 120, step: 10, val: 30, unit: ' °/s', desc: 'Sets preview drum rotation and the existing slip proxy; not a measured visual response.', apply: (a, v) => { if (a.paradigmState) a.paradigmState.drumVelocityDegS = v; } }
        ],
        actions: [
            { label: 'Invert Grating Direction', class: 'primary', handler: (a, h) => { if (a.paradigmState) a.paradigmState.drumVelocityDegS = -(a.paradigmState.drumVelocityDegS || 30); } },
            { label: 'Toggle preview grating drawing', class: '', handler: (a, h) => { if (a.paradigmState) a.paradigmState.contrast = (a.paradigmState.contrast === 0 ? 0.9 : 0); } }
        ],
        metrics: [
            { label: 'Preview descending steering drive', get: (a) => `${(a.dn.dna02Diff).toFixed(2)}` },
            { label: 'Retinal Slip Rate', get: (a) => `${(a.paradigmState && Number.isFinite(a.paradigmState.effectiveSlip) ? a.paradigmState.effectiveSlip : 0).toFixed(1)} °/s` },
            { label: 'Assumed preview gain', get: (a) => `${(a.paradigmState && Number.isFinite(a.paradigmState.gain) ? a.paradigmState.gain : 0.88).toFixed(2)}` }
        ],
        drawChart: (ctx, w, h, a, hInst) => {
            ctx.fillStyle = '#38bdf8';
            ctx.font = '9px monospace';
            ctx.fillText('Illustrative response curve — not measured', 8, 14);
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
        illustrativeChart: true,
        title: 'Gap Crossing',
        badge: 'GAP',
        ref: 'Background: Pick & Strauss (2005); Triphan et al. (2010) (citation not verified in this repository)',
        sliders: [
            { key: 'gapWidth', label: 'Chasm Void Width', min: 1.5, max: 5.0, step: 0.25, val: 3.5, unit: ' mm', desc: 'Preview gap width. The preview crosses gaps up to its hand-set 3.8 mm threshold and turns back otherwise; no leg reach is simulated.', apply: (a, v) => { if (a.paradigmState) a.paradigmState.gapWidthMm = v; } }
        ],
        actions: [
            { label: 'Widen Gap (+0.5mm)', class: '', handler: (a, h) => { if (a.paradigmState) a.paradigmState.gapWidthMm = Math.min(5.0, (a.paradigmState.gapWidthMm || 3.5) + 0.5); } },
            { label: 'Narrow Gap (-0.5mm)', class: '', handler: (a, h) => { if (a.paradigmState) a.paradigmState.gapWidthMm = Math.max(1.5, (a.paradigmState.gapWidthMm || 3.5) - 0.5); } }
        ],
        metrics: [
            { label: 'Crossing Outcome', get: (a) => { const p = a.paradigmState || {}; return p.crossingSuccess ? 'CROSSED' : (p.decisionOutcome || (p.isProbing ? 'PROBING' : 'APPROACH')); } },
            { label: 'Preview crossing threshold', get: (a) => `${(a.paradigmState && a.paradigmState.reachabilityThreshMm) || 3.8} mm` },
            { label: 'Tactile Probe Time', get: (a) => `${((a.paradigmState && a.paradigmState.probingDurationMs) || 0).toFixed(0)} ms` }
        ],
        drawChart: (ctx, w, h, a, hInst) => {
            ctx.fillStyle = '#4ade80';
            ctx.font = '9px monospace';
            ctx.fillText('Illustrative gap curve — not measured', 8, 14);
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
        illustrativeChart: true,
        title: 'Circadian DAM Monitor',
        badge: 'ACTIVITY',
        ref: 'Background: Konopka & Benzer (1971); Allada & Chung (2010) (citation not verified in this repository)',
        sliders: [],
        actions: [
            { label: 'Set preview speed to 12 mm/s', class: 'danger', handler: (a, h) => { a.fly.speed = 12.0; } },
            { label: 'Clear preview activity counters', class: '', handler: (a, h) => { if (a.paradigmState) { a.paradigmState.beamCrossings = 0; a.paradigmState.totalSleepMin = 0; a.paradigmState.sleepBouts = 0; } } }
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
            ctx.fillText('Illustrative activity curve — not measured', 8, 14);
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
        illustrativeChart: true,
        title: 'Courtship Chamber',
        badge: 'COURTSHIP',
        ref: 'Background: Siegel & Hall (1979); Keleman et al. (2007) (citation not verified in this repository)',
        sliders: [],
        actions: [
            { label: 'Reset preview MB memory', class: '', handler: (a, h) => { a.mb.reset(false); } }
        ],
        metrics: [
            { label: 'Courtship Index (CI)', get: (a) => `${(a.paradigmState && Number.isFinite(a.paradigmState.courtshipIndex) ? a.paradigmState.courtshipIndex : 0).toFixed(2)}` },
            { label: 'Wing Extension', get: (a) => `${((a.paradigmState && a.paradigmState.wingAngleDeg) || 0).toFixed(0)}°` },
            { label: 'Target Proximity', get: (a) => { const p = a.paradigmState; return p && p.femalePos ? `${Math.hypot(a.fly.x - p.femalePos[0], a.fly.y - p.femalePos[1]).toFixed(1)} mm` : '--'; } }
        ],
        drawChart: (ctx, w, h, a, hInst) => {
            ctx.fillStyle = '#f43f5e';
            ctx.font = '9px monospace';
            ctx.fillText('Illustrative decay curve — not measured', 8, 14);
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
        badge: 'ENGINEERED MAZE',
        ref: 'Engineered maze with Coulomb sliding contacts (no biological reference)',
        sliders: [
            { key: 'wallFriction', label: 'Wall Coulomb Friction', min: 0.0, max: 0.8, step: 0.05, val: 0.5, unit: '', desc: 'Coulomb crawling friction coefficient along corridor walls during sliding contacts.', apply: (a, v) => { (a.currentWalls || []).forEach(w => { w.friction = v; }); } }
        ],
        actions: [
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
        title: 'Multisensory Sandbox · Heuristic Body Proxy',
        badge: 'BODY PROXY',
        ref: 'Engineered preview: odour, thermal and wind cues with Kuramoto leg pose (not a biological nerve cord)',
        sliders: [
            { key: 'cpgCadence', label: 'Kuramoto CPG Base Cadence', min: 3.0, max: 14.0, step: 0.5, val: 8.0, unit: ' Hz', desc: 'Base frequency of the engineered Kuramoto tripod oscillator that sets the 6 preview leg phases.', apply: (a, v) => { a.cpg.baseFreq = v; } },
            { key: 'wallRepulsion', label: 'Boundary Repulsion', min: 0.2, max: 3.0, step: 0.1, val: 1.0, unit: 'x', desc: 'Gain of the anticipatory wall-avoidance steering (antennal proximity whiskers) that turns the fly away from walls and pillars.', apply: (a, v) => { a.wallRepulsion = v; } }
        ],
        actions: [
            { label: 'Rotate preview fly', class: 'primary', handler: (a) => { if (a.isStandalonePreview()) a.fly.heading += 0.4; } }
        ],
        metrics: [
            { label: 'Heuristic Composite', get: (a) => sandboxScoreView(a).catalog },
            { label: 'Tripod Coordination Proxy', get: (a) => sandboxScoreView(a).text.coordination },
            { label: 'Sensory Alignment Proxy', get: (a) => sandboxScoreView(a).text.sensory }
        ],
        drawChart: (ctx, w, h, a, hInst) => {
            ctx.fillStyle = '#38bdf8';
            ctx.font = '9px monospace';
            ctx.fillText('Heuristic Body Proxy (Coord / Sens / Effic / Smooth)', 8, 14);
            const score = sandboxScoreView(a);
            if (['coordination', 'sensory', 'efficiency', 'smoothness'].some(key => score.raw[key] === null)) {
                ctx.fillText('Unavailable: no complete current sandbox proxy', 8, h / 2);
                return;
            }
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
            ctx.moveTo(cx, cy - r * score.raw.coordination);
            ctx.lineTo(cx + r * score.raw.sensory, cy);
            ctx.lineTo(cx, cy + r * score.raw.efficiency);
            ctx.lineTo(cx - r * score.raw.smoothness, cy);
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
    // Preview-only toggles named after published experiments (Science Guide, Part A).
    // They switch off parts of the hand-built browser preview, never the connectome,
    // and predict no deficit size. Connectome lesions are planned: POST_V04 NEXT-02.
    'WT': {
        name: 'WT Control (preview)',
        driver: 'Nothing switched off',
        mechanism: 'Engineered browser preview: 120 Kenyon-cell units, a 16-wedge heading compass, locomotion and reflex modules. It is illustrative and is not the connectome, which runs separately and unmodified.',
        expectedDeficit: 'Nothing is switched off. No performance is predicted or targeted; measure each assay.',
        color: '#4ade80'
    },
    'DELTA_MB': {
        name: 'ΔMB (preview toggle)',
        driver: 'Named after mushroom-body silencing experiments',
        mechanism: 'Stops the preview\'s Kenyon-cell learning update. Kenyon-cell activity is still computed.',
        expectedDeficit: 'Preview only; no deficit size is predicted. Connectome lesions are planned, not in v0.4 (NEXT-02).',
        color: '#fda4af'
    },
    'DELTA_CX': {
        name: 'ΔCX (preview toggle)',
        driver: 'Named after central-complex silencing experiments',
        mechanism: 'Replaces the preview compass with random jitter, so its steering term becomes random.',
        expectedDeficit: 'Preview only; no deficit size is predicted. Connectome lesions are planned, not in v0.4 (NEXT-02).',
        color: '#f43f5e'
    },
    'DELTA_GF': {
        name: 'ΔGF (preview toggle)',
        driver: 'Named after giant-fibre silencing experiments',
        mechanism: 'A looming trigger no longer starts the preview\'s escape.',
        expectedDeficit: 'Preview only; no deficit size is predicted. Connectome lesions are planned, not in v0.4 (NEXT-02).',
        color: '#fb923c'
    },
    'DELTA_JO': {
        name: 'ΔJO (preview toggle)',
        driver: 'Named after Johnston\'s organ experiments',
        mechanism: 'In the Open Arena preview only, sets the wind-direction input to the preview compass to zero.',
        expectedDeficit: 'Preview only; no deficit size is predicted. Connectome lesions are planned, not in v0.4 (NEXT-02).',
        color: '#a78bfa'
    }
};

class ScientificHUD {
    constructor(arena) {
        this.arena = arena;
        this.simSpeed = 1.0;
        this.speedOptions = [0.5, 1.0, 2.0, 3.0, 5.0, 10.0, 15.0, 20.0, 50.0, 100.0];
        this.speedIdx = 1;

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

    speedFeedback(message, error = false) {
        const el = document.getElementById('speedFeedback');
        if (!el) return;
        el.textContent = message || '';
        el.style.color = error ? '#f87171' : '#4ade80';
    }

    reconcileRequestedSpeed(speed) {
        const checked = validateRequestedSpeed(speed);
        if (!checked.ok) return false;
        const num = checked.value;
        this.simSpeed = num;
        const matchedIdx = this.speedOptions.indexOf(num);
        if (matchedIdx !== -1) this.speedIdx = matchedIdx;
        else {
            const next = this.speedOptions.findIndex(value => value > num);
            this.speedIdx = next === -1 ? this.speedOptions.length - 1 : Math.max(0, next - 1);
        }
        const shown = requestedSpeedText(num);

        const btnSpeed = document.getElementById('btnSpeedToggle');
        if (btnSpeed) btnSpeed.textContent = `Speed: ${shown}`;

        const statSpeed = document.getElementById('statSpeed');
        if (statSpeed) statSpeed.textContent = shown;

        const selSpeed = document.getElementById('selectSpeed');
        if (selSpeed) {
            for (const old of [...selSpeed.querySelectorAll('option[data-current-custom]')]) old.remove();
            let option = [...selSpeed.options].find(item => Number(item.value) === num);
            if (!option) {
                option = document.createElement('option');
                option.value = String(num);
                option.textContent = `${shown} (custom)`;
                option.dataset.currentCustom = 'true';
                selSpeed.appendChild(option);
            }
            selSpeed.value = option.value;
        }
        const custom = document.getElementById('customSpeed');
        if (custom && document.activeElement !== custom) custom.value = String(num);
        return true;
    }

    async setSpeed(speed) {
        const checked = validateRequestedSpeed(speed);
        if (!checked.ok) {
            this.speedFeedback(checked.message, true);
            this.reconcileRequestedSpeed(this.simSpeed);
            return false;
        }
        const requested = checked.value;

        if (this.daemonBridge?.replayMode) {
            window.neuroflyReplay?.setSpeed(requested);
            this.reconcileRequestedSpeed(requested);
            this.speedFeedback(`Replay speed set to ${requestedSpeedText(requested)}.`);
        } else if (this.daemonBridge && this.daemonBridge.connected) {
            const result = await this.daemonBridge.sendCommand('set_speed', {speed:requested});
            // A live request may finish after replay takes ownership of the header.
            if (this.daemonBridge?.replayMode) { window.neuroflyReplay?.updateButtons(); return false; }
            if (!result || result.status !== 'ok' || !Number.isFinite(result.sim_speed)) {
                this.speedFeedback(result?.message || 'The daemon did not acknowledge the speed; it was not changed.', true);
                this.reconcileRequestedSpeed(this.simSpeed);
                return false;
            }
            this.reconcileRequestedSpeed(result.sim_speed);
            this.speedFeedback(`Requested speed set to ${requestedSpeedText(result.sim_speed)}.`);
        } else {
            this.reconcileRequestedSpeed(requested);
            this.speedFeedback(`Local preview speed set to ${requestedSpeedText(requested)}.`);
        }
        return true;
    }

    reconcileBackendSelector() {
        const state = backendSelectorState(this.arena, this.daemonBridge, this.backendCommandWaiting);
        const select = document.getElementById('selectBackend');
        if (select) {
            select.disabled = !state.allowed;
            select.value = state.backend;
            select.title = state.reason || 'Request a controller backend; only the daemon can apply it.';
        }
        this.describeUnresolvedBackendCommand?.();
        // A refusal stays current until the SAME daemon acknowledges a later applied
        // rebuild (higher activation) and reports no halt; only then is it shown as
        // history.  A halt that has not recovered is never relabelled.
        const refusal = this.backendRefusal;
        if (refusal && !this.backendCommandWaiting && this.backendCommandError) {
            const bridge = this.daemonBridge, pkt = this.arena?.remotePacket;
            const ack = bridge?.lastSwitchAck;
            const recovered = ack && ack !== refusal.priorAck && ack.applied === true
                && verifiedBackendIdentity(ack.identity) && ack.identity.daemon_run_id === refusal.daemonRunId
                && Number.isInteger(refusal.activation) && ack.identity.activation > refusal.activation
                && !bridge?.daemonHalt && !pkt?.halted && !pkt?.error;
            if (recovered) {
                const step = Number.isInteger(ack.applied_step) && ack.applied_step >= 0 ? ` at step ${ack.applied_step}` : '';
                const action = String(ack.action || 'rebuild').replaceAll('_', ' ');
                this.backendCommandText = `Previous refusal of controller ${refusal.target} at ${refusal.at}, since recovered `
                    + `(${action} applied${step}). Current controller: ${state?.backend || ack.identity.backend}.`;
                this.backendCommandError = false;
                this.backendRefusal = null;
            }
        }
        const status = document.getElementById('backendCommandStatus');
        if (status) {
            const text = [this.backendCommandText, state.reason].filter(Boolean).join(' · ')
                || `Current controller: ${state.backend}.`;
            if (status.textContent !== text) status.textContent = text;
            status.style.color = this.backendCommandError ? '#fca5a5' : '#94a3b8';
        }
        return state;
    }

    // While a backend request is unresolved (queued, or its HTTP request timed out),
    // say what the daemon reports for THIS request: its rebuild phase and elapsed
    // time, or that a different daemon process is now answering.  Nothing here ever
    // claims success; only the request's own final acknowledgement does.
    describeUnresolvedBackendCommand() {
        const cmd = this.backendCommand;
        if (!cmd || !(this.backendCommandWaiting || cmd.unknown)) return;
        const pkt = this.arena?.remotePacket;
        const runId = pkt?.identity?.daemon_run_id || pkt?.run_id;
        if (runId && cmd.daemonRunId && runId !== cmd.daemonRunId) {
            this.backendCommandWaiting = false;
            cmd.unknown = false; cmd.restarted = true;
            this.backendCommandError = true;
            this.backendCommandText = `Controller ${cmd.target}: outcome unknown. The daemon process changed before `
                + `this request was acknowledged; the new process reports controller ${pkt?.identity?.backend || 'unreported'}, `
                + 'which is not an acknowledgement of this request.';
            return;
        }
        const pending = pkt?.observation_lifecycle?.pending_control;
        const mine = pending && ((cmd.commandId && pending.command_id === cmd.commandId)
            || (cmd.clientCommandId && pending.client_command_id === cmd.clientCommandId));
        if (mine) this.noteBackendProgress(cmd, pending);
    }

    noteBackendProgress(cmd, pending) {
        if (cmd !== this.backendCommand) return;
        const phase = String(pending?.persistence_phase || 'applying').replaceAll('_', ' ');
        const elapsed = Number.isFinite(pending?.elapsed_s) ? `, ${pending.elapsed_s.toFixed(1)} s` : '';
        this.backendCommandText = `Controller ${cmd.target}: the daemon is rebuilding (${phase}${elapsed}); not yet applied`
            + (cmd.unknown ? ' (request reply timed out; waiting for this request\'s acknowledgement).' : '.');
    }

    // The HTTP reply was lost or late: look the request up by its own id until the
    // daemon reports its final acknowledgement, the daemon process changes, a newer
    // request supersedes it, or the wait ends with the outcome still unknown.
    reconcileBackendOutcome(cmd) {
        const bridge = this.daemonBridge;
        const deadline = Date.now() + (bridge?.commandAckTimeoutMs || 120000);
        const poll = async () => {
            if (cmd !== this.backendCommand || !cmd.unknown) return;
            const record = await bridge.lookupCommandAck(cmd.clientCommandId);
            if (cmd !== this.backendCommand || !cmd.unknown) return;
            if (record?.daemon_run_id && cmd.daemonRunId && record.daemon_run_id !== cmd.daemonRunId) {
                cmd.unknown = false; cmd.restarted = true;
                this.backendCommandError = true;
                this.backendCommandText = `Controller ${cmd.target}: outcome unknown. The daemon process changed before `
                    + 'this request was acknowledged.';
                this.reconcileBackendSelector();
                return;
            }
            if (record?.state === 'acknowledged' && record.ack) {
                cmd.unknown = false;
                this.finishBackendCommand(record.ack, cmd.target);
                return;
            }
            if (record?.state === 'pending' && record.pending_control) this.noteBackendProgress(cmd, record.pending_control);
            else if (['queued', 'applying'].includes(record?.state))
                this.backendCommandText = `Controller ${cmd.target}: the daemon has the request (${record.state}); not yet applied `
                    + '(request reply timed out; waiting for this request\'s acknowledgement).';
            if (Date.now() >= deadline) {
                cmd.unknown = false;
                this.backendCommandText = `Controller ${cmd.target}: the command request timed out and no acknowledgement for it `
                    + 'was found; its outcome is unknown.';
                this.reconcileBackendSelector();
                return;
            }
            this.reconcileBackendSelector();
            cmd.reconcileTimer = setTimeout(poll, 1500);
        };
        cmd.reconcileTimer = setTimeout(poll, 0);
    }

    finishBackendCommand(result, target) {
        const sameDaemon = verifiedBackendIdentity(result?.ack?.identity)
            && result.ack.identity.daemon_run_id === this.backendCommand?.daemonRunId
            && this.daemonBridge.activeUrl === this.backendCommand?.url;
        const applied = sameDaemon && result?.status === 'ok' && result.ack?.applied === true
            && result.ack.identity.backend === target;
        const newer = prev => DaemonBridgeClient.newerAck(prev, result.ack);
        if (sameDaemon && newer(this.daemonBridge.lastBackendAck)) this.daemonBridge.lastBackendAck = result.ack;
        if (applied && newer(this.daemonBridge.lastSwitchAck)) this.daemonBridge.lastSwitchAck = result.ack;
        const cmd = this.backendCommand;
        if (cmd) cmd.unknown = !!result?.request_unanswered && !!result?.timed_out && !!result?.client_command_id;
        this.backendCommandWaiting = result?.status === 'queued' || (!!result?.timed_out && !!result.command_id);
        this.backendCommandError = !applied;
        this.backendRefusal = null;
        if (applied) this.backendCommandText = `Applied controller ${target} at step ${Number.isInteger(result.ack.applied_step) && result.ack.applied_step >= 0 ? result.ack.applied_step : 'unreported'}.`;
        else if (result?.timed_out) {
            this.backendCommandText = `Controller ${target}: ${result.message}`
                + (cmd?.unknown ? ' Waiting for the daemon\'s acknowledgement of this request.' : '');
            if (cmd?.unknown) this.reconcileBackendOutcome(cmd);
        }
        else if ((result?.ack?.applied === false || result?.applied === false) && result.status !== 'queued') {
            this.backendCommandText = `Controller ${target} refused: ${result.message || 'the daemon did not apply the request'}.`;
            const activation = sameDaemon ? result.ack.identity.activation
                : backendSelectorState(this.arena, this.daemonBridge).identity?.activation;
            this.backendRefusal = {target, daemonRunId: this.backendCommand?.daemonRunId, activation,
                                   priorAck: this.daemonBridge?.lastSwitchAck || null,
                                   at: new Date().toLocaleTimeString()};
        }
        else this.backendCommandText = `Controller ${target} not confirmed: ${result?.message || 'no final applied acknowledgement received'}.`;
        this.reconcileBackendSelector();
        return applied;
    }

    backendCommandAck(ack) {
        // Only the CURRENT request's own acknowledgement (daemon id or request id)
        // resolves it; a superseded request's late ack never overwrites a newer one.
        const cmd = this.backendCommand;
        if (!cmd || !ack) return;
        const mine = (cmd.commandId && ack.command_id === cmd.commandId)
            || (cmd.clientCommandId && ack.client_command_id === cmd.clientCommandId);
        if (!mine) return;
        if (cmd.unknown) cmd.unknown = false;
        this.finishBackendCommand(ack, cmd.target);
    }

    async setBackend(target) {
        const state = this.reconcileBackendSelector();
        if (!state.allowed || !SELECTABLE_BACKENDS.includes(target)) {
            this.backendCommandText = `Controller change refused: ${state.reason || 'unknown backend'}`;
            this.backendCommandError = true;
            this.reconcileBackendSelector();
            return false;
        }
        const previous = this.backendCommand;
        if (previous) { previous.unknown = false; clearTimeout(previous.reconcileTimer); }
        const clientCommandId = this.daemonBridge.newClientCommandId?.() || null;
        const cmd = this.backendCommand = {target, commandId: null, clientCommandId,
            daemonRunId: state.identity.daemon_run_id, url: this.daemonBridge.activeUrl};
        this.backendCommandWaiting = true;this.backendCommandError = false;
        this.backendCommandText = `Requesting controller ${target}; awaiting a final acknowledgement.`;
        this.reconcileBackendSelector();
        const result = await this.daemonBridge.sendCommand('switch_backend', {backend: target}, queued => {
            cmd.commandId = queued.command_id;
            if (cmd !== this.backendCommand) return;
            this.backendCommandText = `Controller ${target} queued; not yet applied.`;
            this.reconcileBackendSelector();
        }, {clientCommandId});
        // A newer request was made meanwhile: this reply must not overwrite its status.
        if (cmd !== this.backendCommand) return false;
        return this.finishBackendCommand(result, target);
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
        if (this.daemonBridge?.replayMode) {
            const label = document.getElementById('arenaRunState');
            if (label) label.textContent = 'Replaying a recording: exit the replay to switch assays.';
            return;
        }
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
                if (activeTab === tabAssayTools) {
                    this.updateAssayTools();
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
        const sCpg = document.getElementById('sliderStimCpg');
        if (sCpg) sCpg.addEventListener('input', (e) => {
            if (!this.arena.isStandalonePreview()) return;
            const v = parseFloat(e.target.value);
            document.getElementById('valStimCpg').textContent = v.toFixed(1);
            this.arena.cpg.baseFreq = v;
        });
        const bGf = document.getElementById('btnFlareGf');
        if (bGf) bGf.addEventListener('click', () => {
            if (!this.arena.isStandalonePreview()) return;
            this.arena.dn.dnp01Gf += 1;
            this.arena.dn.escapeActive = true;
            this.arena.dn.escapeTimer = 0.40;
        });
        const bWind = document.getElementById('btnFlareWind');
        if (bWind) bWind.addEventListener('click', () => this.arena.startPreviewGust());
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
            watchEl.innerHTML = '<div><b>Engineered preview, not connectome results.</b></div><ul>' + guide.whatToWatch.map(pt => `<li>${pt}</li>`).join('') + '</ul>';
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
                        if (this.arena.isStandalonePreview()) p.apply(this.arena, val);
                    });
                });
            } else {
                container.innerHTML = '<div style="font-size:9.5px; color:#94a3b8; font-style:italic;">Local preview: no supported adjustable parameters.</div>';
            }
        }
    }

    setupLesionEvents() {
        const lesions = [
            { id: 'btnLesionWT', type: 'WT', label: 'WT CONTROL' },
            { id: 'btnLesionMB', type: 'DELTA_MB', label: 'ΔMB (KENYON)' },
            { id: 'btnLesionCX', type: 'DELTA_CX', label: 'ΔCX (COMPASS)' },
            { id: 'btnLesionGF', type: 'DELTA_GF', label: 'ΔGF (ESCAPE)' },
            { id: 'btnLesionJO', type: 'DELTA_JO', label: 'ΔJO (WIND)' }
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
            btnReset.addEventListener('click', () => {if(!this.daemonBridge?.replayMode)this.arena.resetTrial();});
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
                this.setSpeed(e.target.value);
            });
        }

        const customSpeed = document.getElementById('customSpeed');
        const applyCustomSpeed = () => this.setSpeed(customSpeed?.value ?? '');
        const btnApplyCustomSpeed = document.getElementById('btnApplyCustomSpeed');
        if (btnApplyCustomSpeed) btnApplyCustomSpeed.addEventListener('click', applyCustomSpeed);
        if (customSpeed) customSpeed.addEventListener('keydown', (e) => {
            if (e.key === 'Enter') applyCustomSpeed();
        });

        const selBackend = document.getElementById('selectBackend');
        if (selBackend) selBackend.addEventListener('change', e => this.setBackend(e.target.value));
        this.reconcileBackendSelector();

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

    reconcileToolCapabilities() {
        const caps = arenaToolCapabilities(this.arena, this.daemonBridge);
        if (this.arena.isStandalonePreview && !this.arena.isStandalonePreview()) this.arena.cancelPreviewGust?.();
        if (!caps.tools[this.arena.toolMode]?.enabled) this.arena.toolMode = 'select';
        for (const id of ARENA_TOOL_IDS) {
            const button = document.getElementById(id), mode = id.replace('tool', '').toLowerCase();
            if (!button) continue;
            const spec = caps.tools[mode];button.disabled = !spec.enabled;button.title = spec.title;
            if (button.textContent !== spec.label) button.textContent = spec.label;
            const selected = mode === this.arena.toolMode;
            button.classList.toggle('active', selected);button.setAttribute('aria-pressed', String(selected));
        }
        const help = document.getElementById('arenaToolHelp');
        if (help && help.textContent !== caps.help) help.textContent = caps.help;
        const assist = document.getElementById('previewWallAssist');
        if (assist) {
            assist.disabled = !caps.previewAssist;assist.checked = !!this.arena.previewWallAssist;
            assist.title = caps.previewAssist ? 'Standalone preview wall-reflex assist; does not control the daemon.' : 'Preview-only assist is unavailable during live, waiting, disconnected remote view or replay.';
        }
        const label = document.getElementById('previewWallAssistLabel');
        if (label) {
            label.classList.toggle('preview-control-unavailable', !caps.previewAssist);
            label.title = assist?.title || '';
        }
        const note = document.getElementById('previewWallAssistNote');
        if (note) note.textContent = caps.previewAssist ? 'Standalone preview only' : 'Unavailable in live/replay';
        return caps;
    }

    setupToolEvents() {
        ARENA_TOOL_IDS.forEach(id => {
            document.getElementById(id)?.addEventListener('click', () => {
                const mode = id.replace('tool', '').toLowerCase();
                if (!this.reconcileToolCapabilities().tools[mode]?.enabled) return;
                this.arena.toolMode = mode;this.reconcileToolCapabilities();
            });
        });
        document.getElementById('previewWallAssist')?.addEventListener('change', event => {
            if (arenaToolCapabilities(this.arena, this.daemonBridge).previewAssist)
                this.arena.previewWallAssist = event.target.checked;
            this.reconcileToolCapabilities();
        });
        this.reconcileToolCapabilities();
    }

    renderAssayTools(pid) {
        const panel = document.getElementById('assayToolsPanel');
        if (!panel) return;

        this.liveAssayMounted = !!(this.arena.remoteDriven && this.arena.remotePacket?.live_assay);
        this.liveAssayMountKey = this.liveAssayMounted ? liveAssayMountKey(this.arena.remotePacket) : null;
        if (this.liveAssayMounted) {
            this.liveAssayParadigm = this.arena.remotePacket.paradigm;
            window.mountLiveAssay(this, panel);
            return;
        }
        if (this.arena.remoteDriven || this.arena.awaitingDaemon) {
            panel.replaceChildren();
            const measurements = document.createElement('div');
            measurements.id = 'liveMetrics';
            panel.appendChild(measurements);
            this.activeAssayUpdater = null;
            this.renderObservationPanels();
            return;
        }
        const spec = ASSAY_CONFIGS[pid] || ASSAY_CONFIGS['open-arena'];
        panel.innerHTML = `
            <!-- Assay Header Card -->
            <div class="hud-card" style="margin-bottom:2px;">
                <div class="hud-card-header">
                    <span style="color:#38bdf8; font-weight:800;">${spec.title} · local preview</span>
                    <span class="badge badge-amber">${spec.badge}</span>
                </div>
                <div style="font-size:9px; color:#94a3b8; font-style:italic; line-height:1.3; margin-top:2px;">
                    ${spec.ref}
                </div>
            </div>

            <div class="slider-desc">Only implemented local preview controls are shown; unsupported interventions are unavailable. ${spec.controlNote || ''}</div>

            <!-- Assay Specific Parameters -->
            <div class="hud-card">
                <div class="hud-card-header">
                    <span>Local preview parameters</span>
                    <span style="color:#cbd5e1; font-size:8.5px;">PREVIEW ONLY</span>
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
                    `).join('') || '<div class="slider-desc">No supported adjustable local preview parameters.</div>'}
                </div>
            </div>

            <!-- Assay Levers & Interventions -->
            <div class="hud-card">
                <div class="hud-card-header">
                    <span>Local preview actions</span>
                    <span style="color:#fbbf24; font-size:8.5px;">PREVIEW ONLY</span>
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
                    <span>Local preview readouts</span>
                    <span style="color:#4ade80; font-size:8.5px;">PREVIEW STATE</span>
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
                    <span>${spec.illustrativeChart ? 'Illustrative static diagram · not measurements' : 'Local preview state chart'}</span>
                    <span style="color:#c084fc; font-size:8.5px;">${spec.illustrativeChart ? 'ILLUSTRATION' : 'PREVIEW ONLY'}</span>
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
                    if (!this.arena.isStandalonePreview()) return;
                    if (s.apply) s.apply(this.arena, val);
                });
            }
        });

        // Wire action button events
        (spec.actions || []).forEach((act, idx) => {
            const btnEl = panel.querySelector(`#assay_btn_${idx}`);
            if (btnEl) {
                btnEl.addEventListener('click', () => {
                    if (!this.arena.isStandalonePreview()) return;
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

    renderObservationPanels() {
        const measurements = document.getElementById('liveMetrics');
        if (!measurements) return;
        const display = this.arena.getObservationDisplay();
        if (this.renderedObservationDisplay === display && this.renderedObservationHost === measurements) return;
        this.renderedObservationDisplay = display;
        this.renderedObservationHost = measurements;
        window.NeuroFlyObservationRenderer.render(measurements, display);
        if (this.arena.activeParadigmId === 'multisensory-sandbox') {
            const proxy = document.createElement('p');
            const score = sandboxScoreView(this.arena);
            proxy.textContent = 'Heuristic body proxy · ' + (display.live.state === 'available' ? score.text.composite : 'Unavailable') + ' · ' + score.note;
            measurements.appendChild(proxy);
        }
    }

    updateAssayTools() {
        if ((this.arena.remoteDriven || this.arena.awaitingDaemon) && !document.getElementById('liveMetrics'))
            this.renderAssayTools(this.arena.activeParadigmId);
        if ((!this.liveAssayMounted || this.liveAssayMountKey !== liveAssayMountKey(this.arena.remotePacket)) && this.arena.remoteDriven && this.arena.remotePacket?.live_assay) this.renderAssayTools(this.arena.activeParadigmId);
        if (this.arena.remoteDriven || this.arena.awaitingDaemon) {
            const display = this.arena.getObservationDisplay();
            const updateKey = JSON.stringify([this.daemonBridge.connected, this.daemonBridge.readOnly,
                this.daemonBridge.switchPending, document.getElementById('assayToolsPanel')?.style.display]);
            if (this.lastAssayDisplay !== display || this.lastAssayUpdateKey !== updateKey) {
                this.lastAssayDisplay = display;
                this.lastAssayUpdateKey = updateKey;
                if (this.activeAssayUpdater) this.activeAssayUpdater();
                this.renderObservationPanels();
            }
        } else if (this.activeAssayUpdater) this.activeAssayUpdater();
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
                project: `Project NeuroFly — ${this.arena.remotePacket?.identity?.label || this.arena.remotePacket?.identity?.backend || 'compact modular model'}`,
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
            canonicalMetrics: this.arena.remoteDriven || this.arena.awaitingDaemon ? this.arena.getObservationDisplay() : this.arena.getParadigmMetrics(),
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
        if (this.arena.remoteDriven || this.arena.awaitingDaemon) { this.renderLiveOutcome(); return; }
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
        if(this.liveOutcomeSegment!==metric.contextKey){
            this.liveOutcomeSegment=metric.contextKey;this.liveOutcomeHistory=[];this.liveOutcomeStep=null;
        }
        if(t && this.liveOutcomeStep!==t.step){
            this.liveOutcomeHistory.push({time:t.sim_time_s,value:metric.rawValue});
            this.liveOutcomeHistory=this.liveOutcomeHistory.slice(-300);this.liveOutcomeStep=t.step;
        }
        const ctx=this.curveCtx,w=this.curveWidth||this.curveCanvas.clientWidth,h=this.curveHeight||this.curveCanvas.clientHeight;
        ctx.clearRect(0,0,w,h);
        const rows=this.liveOutcomeHistory, values=rows.map(r=>r.value).filter(Number.isFinite);
        ctx.fillStyle='#94a3b8';ctx.font='8px monospace';ctx.textAlign='center';
        ctx.fillText('Live provisional · '+metric.unit+' · sim s',w/2,10);
        if(!values.length){ctx.fillText(metric.value+' · '+metric.sub,w/2,h/2);return;}
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

    updateCardMetric(metric) {
        // Inactive cards have no current observation or mode-specific metric.
        document.querySelectorAll('[id^="cardMetric"]').forEach(el => {
            el.textContent = 'Not selected';
            const row = el.parentElement;
            const label = row?.querySelector('[data-card-metric-label]');
            if (label) label.textContent = 'Metric:';
            if (row) {
                row.title = 'No current metric · assay not selected';
                row.setAttribute('aria-label', row.title);
            }
        });
        const activeCardMetricEl = {
            'open-arena': 'cardMetricOpenArena',
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
        if (!activeCardMetricEl) return;
        const el = document.getElementById(activeCardMetricEl);
        if (!el) return;
        // Canonical value already carries its formatted unit/status; do not
        // reformat, append a second unit or infer a value from another mode.
        el.textContent = metric.value;
        const row = el.parentElement;
        const label = row?.querySelector('[data-card-metric-label]');
        if (label) label.textContent = 'Metric: ' + metric.label;
        if (row) {
            const unit = metric.unit ? ' · unit: ' + metric.unit.trim() : '';
            const raw = Number.isFinite(metric.rawValue)
                ? ' · Raw value: ' + (Object.is(metric.rawValue, -0) ? '-0' : String(metric.rawValue)) + unit : '';
            row.title = metric.label + ': ' + metric.value + unit + (metric.sub ? ' · ' + metric.sub : '') + raw;
            row.setAttribute('aria-label', row.title);
        }
    }

    update() {
        window.neuroflyTrainingOwnerReconcile?.();
        const reset=document.getElementById('btnResetTrial');
        if(reset){reset.disabled=!!this.daemonBridge?.replayMode;reset.title=reset.disabled?'Replay is read only. Exit replay to reset a live or preview trial.':'';}
        this.reconcileBackendSelector();
        this.reconcileToolCapabilities();
        if (this.arena.remoteDriven && this.arena.remotePacket?.live_assay) {
            const t=this.arena.remotePacket;
            const watch=document.getElementById('guideWhatToWatch');
            if(watch)watch.textContent=assayLimitationForController(t)
                +' Use Assay Tools & Levers for connected controls and measured motor responses.';
            document.querySelectorAll('#paramControlsBox input,#limbDeckPanel input,#limbDeckPanel button,[id^="btnLesion"]').forEach(e=>{e.disabled=true;e.title='Standalone preview control. Use the connected controls in Assay Tools & Levers.';});
        }
        const clocks = clockReadouts(this.arena, this.arena.remotePacket, !!this.arena.remoteDriven,
            !!this.daemonBridge?.replayMode);
        const setText = (id, text, title) => {
            const el = document.getElementById(id);
            if (!el) return;
            el.textContent = text;
            if (title !== undefined) el.title = title;
        };
        setText('statSimTimeLabel', clocks.timeLabel);
        setText('statStepLabel', clocks.stepLabel);
        for (const id of ['statSessionClock', 'statSessionStep']) {
            const box = document.getElementById(id);
            if (box) box.title = clocks.sessionTitle;
        }
        setText('statSimTime', clocks.simTime);
        setText('statStep', clocks.step);
        const graphBox = document.getElementById('statGraphTimeBox');
        if (graphBox) { graphBox.hidden = !clocks.graphVisible; graphBox.title = clocks.graphTitle; }
        setText('statGraphTime', clocks.graphTime);

        setText('paradigmTrial', clocks.trial, clocks.trialTitle);
        setText('guideTrialBadge', clocks.guideTrial, clocks.trialTitle);
        setText('paradigmElapsed', clocks.elapsed, clocks.elapsedTitle);
        const windowEl = document.getElementById('paradigmWindow');
        if (windowEl) { windowEl.hidden = !clocks.windowVisible; windowEl.textContent = clocks.window; windowEl.title = clocks.windowTitle; }

        const metric = this.arena.getCanonicalMetricInfo();
        const mLabelEl = document.getElementById('paradigmMetricLabel');
        const mValEl = document.getElementById('paradigmMetricValue');
        if (mLabelEl) mLabelEl.textContent = metric.label;
        if (mValEl) {
            mValEl.textContent = metric.value;
            mValEl.title = Number.isFinite(metric.rawValue)
                ? 'Raw value: ' + (Object.is(metric.rawValue, -0) ? '-0' : String(metric.rawValue)) + (metric.unit ? ' ' + metric.unit.trim() : '') : '';
            mValEl.setAttribute('aria-label', metric.value + (mValEl.title ? ' · ' + mValEl.title : ''));
        }

        this.updateCardMetric(metric);
        const observationDisplay = (this.arena.remoteDriven || this.arena.awaitingDaemon) ? this.arena.getObservationDisplay() : null;
        const watchTyped = document.getElementById('guideWhatToWatch');
        if (watchTyped && observationDisplay) watchTyped.textContent = assayLimitationForController(this.arena.remotePacket || {})
            + ' ' + metric.label + ': ' + metric.value + '. ' + metric.sub + '. See separate live and saved observations in Assay Tools & Levers.';

        const panelCaps = this.arena.remoteDriven
            ? graphPanelCapabilities(this.arena.remotePacket || {})
            : graphPanelCapabilities({identity:{backend:'modular', label:'Standalone modular preview'}});
        const mbTitle = document.getElementById('mushroomBodyPanelTitle');
        const mbReason = document.getElementById('mushroomBodyPanelReason');
        if (mbTitle) mbTitle.textContent = panelCaps.modularMemory
            ? '[1] Mushroom Body (120 KCs · modular controller)'
            : `[1] Mushroom Body · unavailable for ${panelCaps.label}`;
        if (mbReason) {
            mbReason.textContent = panelCaps.modularMemory
                ? 'Measured from the active modular controller.' : panelCaps.modularReason;
            mbReason.style.color = panelCaps.modularMemory ? '#94a3b8' : '#fbbf24';
        }
        const lesionDescription = document.getElementById('lesionCardDesc');
        if (lesionDescription) {
            lesionDescription.textContent = panelCaps.graph
                ? `${panelCaps.label} is the active controller. The genotype buttons are standalone modular-preview controls and cannot alter this graph run.`
                : (LESION_INFO[this.arena.lesion] || LESION_INFO.WT).mechanism;
        }
        const guideRef = document.getElementById('guideRef');
        if (guideRef) {
            const guide = EXPERIMENT_GUIDES[this.arena.activeParadigmId] || EXPERIMENT_GUIDES['open-arena'];
            const modelVersion = this.arena.remotePacket?.live_assay?.model_version || 'server assay implementation not declared';
            guideRef.textContent = panelCaps.graph
                ? `${panelCaps.label} · ${modelVersion} · modular lesion and memory descriptions do not apply to this controller.`
                : guide.ref;
        }

        // Dopamine readouts. Graph packets also contain the retained modular helper
        // brain, so these values are unavailable for graph-controller interpretation.
        const pamHz = this.arena.mb.pamRate;
        const ppl1Hz = this.arena.mb.ppl1Rate;
        const pamRateEl = document.getElementById('paradigmPamRate');
        const ppl1RateEl = document.getElementById('paradigmPpl1Rate');
        const pamFillEl = document.getElementById('paradigmPamFill');
        const ppl1FillEl = document.getElementById('paradigmPpl1Fill');

        if (pamRateEl) pamRateEl.textContent = panelCaps.modularMemory ? pamHz.toFixed(1) + ' Hz' : 'Unavailable';
        if (ppl1RateEl) ppl1RateEl.textContent = panelCaps.modularMemory ? ppl1Hz.toFixed(1) + ' Hz' : 'Unavailable';
        if (pamFillEl) pamFillEl.style.width = panelCaps.modularMemory ? `${Math.min(100, (pamHz / 40.0) * 100)}%` : '0%';
        if (ppl1FillEl) ppl1FillEl.style.width = panelCaps.modularMemory ? `${Math.min(100, (ppl1Hz / 40.0) * 100)}%` : '0%';

        const dStateEl = document.getElementById('valDopamineState');
        if (dStateEl) {
            dStateEl.title = panelCaps.modularReason;
            if (!panelCaps.modularMemory) {
                dStateEl.textContent = 'UNAVAILABLE · MODULAR MB NOT CONTROLLER';
                dStateEl.style.color = '#94a3b8';
            } else if (pamHz > 5.0) {
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
        if (kcActiveEl) {
            kcActiveEl.textContent = panelCaps.modularMemory
                ? `${activeKcs} / 120 (${((activeKcs / 120) * 100).toFixed(0)}%)` : 'Unavailable';
            kcActiveEl.title = panelCaps.modularReason;
        }

        const valence = this.arena.mb.netValence;
        const valNetEl = document.getElementById('valNetValence');
        const needleEl = document.getElementById('valNeedle');
        const curPiEl = document.getElementById('currentPiVal');
        if (valNetEl) {
            valNetEl.textContent = panelCaps.modularMemory ? (valence >= 0 ? '+' : '') + valence.toFixed(2) : 'Unavailable';
            valNetEl.title = panelCaps.modularReason;
        }
        if (needleEl) {
            // The gauge saturates at its endpoints; the numeric readout stays raw.
            needleEl.style.display = panelCaps.modularMemory ? 'block' : 'none';
            needleEl.style.left = `${((Math.max(-1, Math.min(1, valence)) + 1) / 2) * 100}%`;
            needleEl.title = panelCaps.modularMemory
                ? `Raw valence: ${valence.toFixed(4)} (gauge spans -1 to +1)` : panelCaps.modularReason;
        }
        if (curPiEl) curPiEl.textContent = (valence >= 0 ? '+' : '') + valence.toFixed(2);
        this.renderKcMatrix(panelCaps);
        this.renderLearningCurve();
        if (this.arena.remoteDriven && curPiEl) curPiEl.textContent = metric.value;

        // Compass & E-PG
        const compassHeadingEl = document.getElementById('valCompassHeading');
        const headingBumpEl = document.getElementById('valHeadingBump');
        const upwindAngleEl = document.getElementById('valUpwindAngle');
        const pfl3ErrorEl = document.getElementById('valPfl3Error');
        const antennaDeflectEl = document.getElementById('valAntennaDeflect');

        if (compassHeadingEl) compassHeadingEl.textContent = `${((this.arena.fly.heading * 180) / Math.PI).toFixed(1)}°`;
        if (headingBumpEl) {
            headingBumpEl.textContent = !panelCaps.graph || panelCaps.epgMeasured
                ? `${((this.arena.cx.headingBump * 180) / Math.PI).toFixed(0)}°` : 'Unavailable';
            headingBumpEl.title = panelCaps.graph && !panelCaps.epgMeasured
                ? 'No finite 16-wedge EPG measurement and bump phase were streamed for this graph step.' : '';
        }
        const windGlobal = Math.atan2(-this.arena.windVector[1], -this.arena.windVector[0]);
        if (upwindAngleEl) upwindAngleEl.textContent = `${((windGlobal * 180) / Math.PI).toFixed(0)}°`;
        if (pfl3ErrorEl) pfl3ErrorEl.textContent = this.arena.remoteDriven ? 'Not streamed' : (this.arena.cx.pfl3ErrorR - this.arena.cx.pfl3ErrorL).toFixed(2);
        if (antennaDeflectEl) antennaDeflectEl.textContent = this.arena.remoteDriven ? 'Not streamed' : `${(Math.hypot(this.arena.windVector[0], this.arena.windVector[1]) * 0.12).toFixed(1)} μN`;
        this.renderCompass(panelCaps);

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
        if (cpgFreqEl) {
            const cadence = this.arena.remotePacket?.biomechanics?.cadence_hz;
            const noStep = this.arena.remotePacket?.biomechanics?.cadence_unavailable;
            cpgFreqEl.textContent = panelCaps.graph
                ? (Number.isFinite(cadence) ? `${cadence.toFixed(1)} Hz · proxy` : (noStep ? 'No step yet' : 'Unavailable'))
                : this.arena.cpg.steppingFreq.toFixed(1) + ' Hz';
            cpgFreqEl.title = panelCaps.graph
                ? (Number.isFinite(cadence) || !noStep
                    ? 'Model-derived body cadence from the streamed biomechanics channel; not a measured connectome CPG readout.'
                    : `No measurement yet: ${noStep}.`) : '';
        }
        const gaitHeading = document.getElementById('gaitPanelTitle');
        if (gaitHeading) gaitHeading.textContent = `[3] ${panelCaps.gaitLabel}`;

        // Clear sandbox-only displays on every frame, including after assay switches.
        renderSandboxScorecard(this.arena, document);
        const p = this.arena.paradigmState;
        if (p && this.arena.activeParadigmId === 'multisensory-sandbox') {
            const jointBox = document.getElementById('jointAnglesBox');
            if (jointBox) {
                const ja = (p && p.jointAngles) ? p.jointAngles : (this.arena.remotePacket?.biomechanics?.joint_angles);
                const ls = this.arena.cpg.legStates || {};
                if (ja) {
                    const fmt = (v) => Number.isFinite(v) ? v.toFixed(0) : '--';
                    jointBox.innerHTML = Object.entries(ja).map(([leg, info]) => {
                        const ctr = info?.ctr !== undefined ? info.ctr : 0;
                        const fti = info?.fti !== undefined ? info.fti : 80;
                        const isStance = info?.phase ? (info.phase === 'STANCE') : !!ls[leg];
                        const sign = ctr >= 0 ? '+' : '';
                        return `<div>${leg}: CTr: ${sign}${fmt(ctr)}° FTi: ${fmt(fti)}° [${isStance ? 'STANCE' : 'SWING'}]</div>`;
                    }).join('');
                }
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

        // Update Premotor & Descending HUD (Phase 3 Step 3.3)
        this.updatePremotorHUD();
    }

    updatePremotorHUD() {
        const pkt = this.arena.remotePacket || {};
        const capabilities = graphPanelCapabilities(pkt);
        const graphDn = graphDnReadout(pkt, capabilities);
        // Graph readouts must come from the graph controller's own nested payload.
        // The top-level values can be pose-derived compatibility data and are not
        // evidence that a graph DN population was resolved.
        const dn = graphDn.rates || {
            dna02_l: Math.max(0, -this.arena.fly.yawRate * 8.0),
            dna02_r: Math.max(0, this.arena.fly.yawRate * 8.0),
            dnp09: Math.max(0, this.arena.fly.speed * 2.5),
            mdn: this.arena.dn?.mdn ? 25.0 : 0.0,
            gf: this.arena.dn?.escapeActive ? 50.0 : 0.0
        };
        const rawCtrl = pkt.controller_id || (pkt.identity?.backend?.startsWith('connectome') ? 'connectome-v3' : 'modular');
        const backendName = String(pkt.identity?.backend || rawCtrl || 'modular').toLowerCase();
        const isPlastic = backendName.includes('plastic');
        const isConn = backendName.includes('connectome');
        const displayLabel = capabilities.label || backendName;

        const badge = document.getElementById('controllerBadge');
        if (badge) {
            badge.textContent = displayLabel.toUpperCase();
            badge.className = isPlastic ? 'badge badge-purple' : isConn ? 'badge badge-green' : 'badge badge-cyan';
        }
        const v3dTag = document.getElementById('v3dControllerTag');
        if (v3dTag) {
            v3dTag.textContent = displayLabel.toUpperCase();
            v3dTag.style.color = isPlastic ? '#c084fc' : isConn ? '#22c55e' : '#38bdf8';
        }
        const compassSource = document.getElementById('valCompassSource');
        if (compassSource) {
            compassSource.textContent = isConn
                ? (capabilities.epgMeasured ? `${displayLabel} · EPG measured` : `${displayLabel} · EPG unavailable`)
                : displayLabel;
            compassSource.className = isPlastic ? 'badge badge-purple' : isConn ? 'badge badge-green' : 'badge badge-cyan';
        }

        const epgBumpEl = document.getElementById('valEpgBump');
        if (epgBumpEl) {
            if (capabilities.epgMeasured) {
                const deg = Math.round(this.arena.cx.headingBump * 180 / Math.PI);
                epgBumpEl.textContent = `${deg}°`;
                epgBumpEl.style.color = '#22c55e';
            } else {
                epgBumpEl.textContent = 'Unavailable';
                epgBumpEl.style.color = '#94a3b8';
                epgBumpEl.title = isConn
                    ? 'No finite 16-wedge EPG measurement and bump phase were streamed for this graph step.'
                    : 'EPG connectome readout applies only when a graph controller streams it.';
            }
        }

        const wp6Block = document.getElementById('wp6PlasticityBlock');
        if (wp6Block) {
            wp6Block.style.display = capabilities.wp6Measured ? 'block' : 'none';
        }
        if (capabilities.wp6Measured) {
            const wp6 = pkt.plasticity?.wp6 || pkt.connectome?.wp6 || {};
            const meanDelta = Number.isFinite(wp6.mean_delta) ? wp6.mean_delta : 0.0;
            const maxDelta = Number.isFinite(wp6.max_delta) ? wp6.max_delta : Math.abs(meanDelta);
            const pct = Math.min(100, Math.max(0, (Math.abs(meanDelta) / 0.5) * 100));

            const mDeltaEl = document.getElementById('hudWp6MeanDelta');
            const pctEl = document.getElementById('hudWp6Pct');
            const barEl = document.getElementById('barWp6Delta');
            const maxDeltaEl = document.getElementById('hudWp6MaxDelta');

            if (mDeltaEl) mDeltaEl.textContent = meanDelta.toFixed(4);
            if (pctEl) pctEl.textContent = pct.toFixed(1);
            if (barEl) barEl.style.width = pct + '%';
            if (maxDeltaEl) maxDeltaEl.textContent = maxDelta.toFixed(4);
        }

        const unavailable = graphDn.unavailable;
        const graphMissingReason = graphDn.missingReason;
        const setMeter = (valId, barId, val, maxVal, reason = null) => {
            const vEl = document.getElementById(valId);
            const bEl = document.getElementById(barId);
            const available = !reason && Number.isFinite(val);
            const num = available ? val : null;
            if (vEl) {
                vEl.textContent = available ? num.toFixed(1) : 'Unavailable';
                vEl.title = available ? 'Measured rate streamed by the active controller.'
                    : (reason || 'No finite rate was streamed by the active controller.');
            }
            if (bEl) {
                const pct = available ? Math.min(100, Math.max(0, (num / maxVal) * 100)) : 0;
                bEl.style.width = pct + '%';
            }
        };

        setMeter('hudDna02L', 'barDna02L', dn.dna02_l, 20.0, graphMissingReason || unavailable.dna02_l);
        setMeter('hudDna02R', 'barDna02R', dn.dna02_r, 20.0, graphMissingReason || unavailable.dna02_r);
        setMeter('hudDnp09', 'barDnp09', dn.dnp09, 40.0, graphMissingReason || unavailable.dnp09);
        setMeter('hudMdn', 'barMdn', dn.mdn, 30.0, graphMissingReason || unavailable.mdn);
        setMeter('hudGf', 'barGf', dn.gf, 60.0, graphMissingReason || unavailable.gf || unavailable.dnp01);

        const gaitName = document.getElementById('cpgGaitName');
        if (gaitName) {
            gaitName.textContent = capabilities.graph ? 'MODEL-DERIVED BODY PHASE' : 'KURAMOTO TRIPOD GAIT';
            gaitName.title = capabilities.graph
                ? 'Phase labels come from the streamed body model, not a connectome CPG measurement.' : '';
        }

        const legStates = this.arena.cpg.legStates || {};
        const legs = ['L1', 'L2', 'L3', 'R1', 'R2', 'R3'];
        legs.forEach(leg => {
            const dot = document.getElementById(`cpgDot${leg}`);
            if (dot) {
                const isStance = !!legStates[leg];
                dot.style.background = isStance ? '#22c55e' : '#1e293b';
                dot.style.color = isStance ? '#0f172a' : '#94a3b8';
                dot.style.fontWeight = isStance ? 'bold' : 'normal';
            }
        });
    }

    renderKcMatrix(capabilities = graphPanelCapabilities({identity:{backend:'modular'}})) {
        if (!this.kcCanvas || !this.kcCtx) return;
        const w = this.kcCanvas.width;
        const h = this.kcCanvas.height;
        this.kcCtx.clearRect(0, 0, w, h);

        if (!capabilities.modularMemory) {
            this.kcCtx.fillStyle = '#94a3b8';
            this.kcCtx.font = '11px monospace';
            this.kcCtx.textAlign = 'center';
            this.kcCtx.fillText('Unavailable — modular KC activity is not this controller', w / 2, h / 2);
            this.kcCtx.textAlign = 'start';
            return;
        }

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

    renderCompass(capabilities = graphPanelCapabilities({identity:{backend:'modular'}})) {
        if (!this.compassCanvas || !this.compassCtx) return;
        const w = this.compassCanvas.width;
        const h = this.compassCanvas.height;
        const cx = w / 2, cy = h / 2, r = w / 2 - 6;

        this.compassCtx.clearRect(0, 0, w, h);

        // 16 wedges of protocerebral bridge aligned with Cartesian needle
        for (let i = 0; i < 16; i++) {
            const a1 = -Math.PI + (i * 2 * Math.PI) / 16;
            const a2 = a1 + (2 * Math.PI) / 16;
            const act = capabilities.graph && !capabilities.epgMeasured ? 0 : this.arena.cx.bumpProfile[i];

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

        // Decoded EPG Bump needle (cyan, representing connectome bump phase)
        const isConn = capabilities.graph && capabilities.epgMeasured;
        if (isConn) {
            this.compassCtx.strokeStyle = '#22c55e';
            this.compassCtx.lineWidth = 2.0;
            this.compassCtx.shadowColor = 'rgba(34, 197, 94, 0.8)';
            this.compassCtx.shadowBlur = 5;
            this.compassCtx.beginPath();
            this.compassCtx.moveTo(cx, cy);
            this.compassCtx.lineTo(cx + Math.cos(this.arena.cx.headingBump) * (r - 2), cy - Math.sin(this.arena.cx.headingBump) * (r - 2));
            this.compassCtx.stroke();
            this.compassCtx.shadowBlur = 0;
        }

        // Ground-truth fly heading needle (white)
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
    } catch (e) {
        // Startup failed: say so with context instead of leaving a blank or half-built page.
        NeuroflyErrors.report('startup', e);
    }
});

// =========================================================================
// Phase 3 Step 3.2: Three.js 3D Articulated Viewport
// =========================================================================

function orbitFrameValues(cameraPosition, controlTarget, flyPosition, preferredOffset = null) {
    const finiteVector = (v) => v && [v.x, v.y, v.z].every(Number.isFinite);
    if (!finiteVector(cameraPosition) || !finiteVector(controlTarget) || !finiteVector(flyPosition)) return null;
    const offset = finiteVector(preferredOffset) ? preferredOffset : {
        x: cameraPosition.x - controlTarget.x,
        y: cameraPosition.y - controlTarget.y,
        z: cameraPosition.z - controlTarget.z
    };
    return {
        target: {x: flyPosition.x, y: flyPosition.y, z: flyPosition.z},
        camera: {
            x: flyPosition.x + offset.x,
            y: flyPosition.y + offset.y,
            z: flyPosition.z + offset.z
        },
        offset: {x: offset.x, y: offset.y, z: offset.z}
    };
}
window.neuroflyOrbitFrameValues = orbitFrameValues;

function arenaPointFor3D(x, y, bounds) {
    if (![x, y, bounds?.minX, bounds?.maxX, bounds?.minY, bounds?.maxY].every(Number.isFinite)
            || bounds.maxX <= bounds.minX || bounds.maxY <= bounds.minY) return null;
    return {
        x: x - (bounds.minX + bounds.maxX) / 2,
        y: y - (bounds.minY + bounds.maxY) / 2
    };
}

function assayGeometry3DDescriptor(arena) {
    const pid = arena?.activeParadigmId || '';
    const packet = arena?.remotePacket;
    const packetPid = (packet?.paradigm || '').toLowerCase().replace(/_/g, '-');
    const streamed = packet?.scene?.geometry;
    const streamCurrent = !!packet && packetPid === pid;
    const streamValid = streamCurrent && streamed?.schema === 'neurofly.assay-geometry.v1'
        && streamed.source === 'python-arena' && streamed.coordinate_frame === 'arena-mm';
    let bounds = null;
    let boundsSource = 'browser preview configuration';
    let offset = [0, 0];
    if (streamValid && Array.isArray(streamed.bounds) && streamed.bounds.length === 4
            && streamed.bounds.every(Number.isFinite)) {
        offset = daemonFrameOffset(packet);
        bounds = {
            minX: streamed.bounds[0] + offset[0], maxX: streamed.bounds[2] + offset[0],
            minY: streamed.bounds[1] + offset[1], maxY: streamed.bounds[3] + offset[1]
        };
        boundsSource = 'Python arena telemetry';
    } else if (!packet && !arena?.awaitingDaemon && !arena?.remoteDriven && arena?.worldBounds) {
        bounds = {...arena.worldBounds};
    }
    const validBounds = [bounds?.minX, bounds?.maxX, bounds?.minY, bounds?.maxY].every(Number.isFinite)
        && bounds.maxX > bounds.minX && bounds.maxY > bounds.minY;
    if (!validBounds) {
        return {key: JSON.stringify([pid, 'unavailable']), paradigm: pid, available: false,
            status: streamCurrent
                ? '3D assay geometry unavailable · current daemon supplied no valid geometry contract'
                : '3D assay geometry unavailable · awaiting current assay geometry telemetry'};
    }
    const walls = [];
    const rawWalls = streamValid ? streamed.walls : (arena.currentWalls || []);
    if (!Array.isArray(rawWalls)) {
        return {key: JSON.stringify([pid, 'unavailable-walls']), paradigm: pid, available: false,
            status: '3D assay geometry unavailable · wall collection is malformed'};
    }
    for (const wall of rawWalls) {
        const values = [wall?.p1?.[0] + offset[0], wall?.p1?.[1] + offset[1],
            wall?.p2?.[0] + offset[0], wall?.p2?.[1] + offset[1]];
        if (!values.every(Number.isFinite)) {
            return {key: JSON.stringify([pid, 'unavailable-wall-values']), paradigm: pid, available: false,
                status: '3D assay geometry unavailable · wall endpoint is nonfinite'};
        }
        const p1 = arenaPointFor3D(values[0], values[1], bounds);
        const p2 = arenaPointFor3D(values[2], values[3], bounds);
        if (!p1 || !p2 || Math.hypot(p2.x - p1.x, p2.y - p1.y) <= 1e-9) {
            return {key: JSON.stringify([pid, 'unavailable-wall-segment']), paradigm: pid, available: false,
                status: '3D assay geometry unavailable · wall segment is degenerate'};
        }
        walls.push({p1, p2});
    }
    const width = bounds.maxX - bounds.minX;
    const depth = bounds.maxY - bounds.minY;
    let rawSurfaces = streamValid ? streamed.surfaces : null;
    let surfaceNote = streamValid ? 'Python arena physical surfaces' : 'browser preview extent';
    if (!rawSurfaces && pid === 'gap-crossing' && Number.isFinite(arena.paradigmState?.gapWidthMm)) {
        const gapStart = 45.0;
        const gapEnd = gapStart + arena.paradigmState.gapWidthMm;
        rawSurfaces = [
            {shape:'rectangle', bounds:[0.0, 7.0, gapStart, 13.0]},
            {shape:'rectangle', bounds:[gapEnd, 7.0, 100.0, 13.0]}
        ];
        surfaceNote = `two browser preview track platforms with ${arena.paradigmState.gapWidthMm} mm gap`;
    } else if (!rawSurfaces && pid === 'circadian-dam') {
        rawSurfaces = Array.from({length: 16}, (_, i) => (
            {shape:'rectangle', bounds:[0.0, i * 10.0, 65.0, (i + 1) * 10.0]}));
        surfaceNote = '16 browser preview DAM tube extents';
    }
    if (!rawSurfaces) rawSurfaces = [{shape:'rectangle', bounds:[bounds.minX, bounds.minY, bounds.maxX, bounds.maxY]}];
    if (!Array.isArray(rawSurfaces)) {
        return {key: JSON.stringify([pid, 'unavailable-surfaces']), paradigm: pid, available: false,
            status: '3D assay geometry unavailable · surface collection is malformed'};
    }
    const surfaces = [];
    for (const surface of rawSurfaces) {
        if (surface?.shape === 'rectangle' && Array.isArray(surface.bounds) && surface.bounds.length === 4
                && surface.bounds.every(Number.isFinite)) {
            surfaces.push({shape:'rectangle', minX:surface.bounds[0] + offset[0],
                minY:surface.bounds[1] + offset[1], maxX:surface.bounds[2] + offset[0],
                maxY:surface.bounds[3] + offset[1], role:surface.role || null});
        } else if (surface?.shape === 'circle' && Array.isArray(surface.center)
                && surface.center.length === 2 && surface.center.every(Number.isFinite)
                && Number.isFinite(surface.radius) && surface.radius > 0) {
            surfaces.push({shape:'circle', center:{x:surface.center[0] + offset[0],
                y:surface.center[1] + offset[1]}, radius:surface.radius, role:surface.role || null});
        } else {
            return {key: JSON.stringify([pid, 'unavailable-surface-values']), paradigm: pid, available: false,
                status: '3D assay geometry unavailable · physical surface is malformed'};
        }
    }
    if (!surfaces.length) {
        return {key: JSON.stringify([pid, 'unavailable-empty-surfaces']), paradigm: pid, available: false,
            status: '3D assay geometry unavailable · no finite physical surfaces supplied'};
    }
    let outline = null;
    const containment = streamValid ? streamed.containment : null;
    if (containment?.kind === 'circle' && Array.isArray(containment.center)
            && containment.center.length === 2 && containment.center.every(Number.isFinite)
            && Number.isFinite(containment.radius) && containment.radius > 0) {
        outline = {shape:'circle', center:{x:containment.center[0] + offset[0],
            y:containment.center[1] + offset[1]}, radius:containment.radius};
    }
    const kind = streamValid ? (walls.length ? 'python-wall-segments' : 'python-containment')
        : (walls.length ? 'browser-preview-wall-segments' : 'browser-preview-bounds');
    const canonical = {pid, bounds, walls, surfaces, outline, kind};
    return {
        key: JSON.stringify(canonical),
        paradigm: pid, available: true, bounds, boundsSource, width, depth, walls, surfaces,
        surfaceNote, kind, outline,
        status: streamValid
            ? (walls.length
                ? `3D assay geometry · Python arena telemetry · ${walls.length} physical wall segments · ${surfaceNote} · illustrative 2 mm wall height`
                : outline?.shape === 'circle'
                    ? `3D assay geometry · Python arena telemetry · exact ${outline.radius} mm circular containment · ${surfaceNote}`
                    : `3D assay geometry · Python arena telemetry · exact bounds and physical surfaces · no wall segments`)
            : `3D assay geometry · browser preview configuration · ${walls.length} configured wall segments · ${surfaceNote}`
    };
}

function disposeThreeTree(root) {
    if (!root?.traverse) return;
    root.traverse((object) => {
        object.geometry?.dispose?.();
        const materials = Array.isArray(object.material) ? object.material : [object.material];
        for (const material of materials) material?.dispose?.();
    });
}

window.neuroflyArenaPointFor3D = arenaPointFor3D;
window.neuroflyAssayGeometry3DDescriptor = assayGeometry3DDescriptor;

class ArticulatedFly3DViewport {
    constructor(canvasId, containerId, arena) {
        this.canvas = document.getElementById(canvasId);
        this.container = document.getElementById(containerId);
        this.statusPanel = document.getElementById('viewport3DOverlay');
        this.arena = arena;
        this.visible = false;
        this.cameraMode = 'orbit'; // 'orbit' or 'chase'
        this.initialized = false;
        this.cacheIdentity = null;
        this.orbitFrameIdentity = null;
        this.lastPose = null;
        this.lastHeading = null;
        this.lastBodyZ = null;
        this.lastJointAngles = null;
        this.lastLegContacts = null;
        this.pendingOrbitFrame = true;
        this.pendingOrbitOffset = null;
        this.savedOrbitOffset = null;
        this.assayGeometryKey = null;
        this.assayGeometryDescriptor = null;
        this.assayGeometryGroup = null;
        this.init();
    }

    init() {
        if (typeof THREE === 'undefined' || !this.canvas || !this.container) return;

        const width = this.container.clientWidth || 800;
        const height = this.container.clientHeight || 600;

        // Scene
        this.scene = new THREE.Scene();
        this.scene.background = new THREE.Color(0x060913);
        this.scene.fog = new THREE.FogExp2(0x060913, 0.005);

        // Camera
        this.camera = new THREE.PerspectiveCamera(45, width / height, 0.1, 1000);
        this.camera.position.set(0, 25, 45);

        // Renderer
        this.renderer = new THREE.WebGLRenderer({ canvas: this.canvas, antialias: true, alpha: false });
        this.renderer.setSize(width, height);
        this.renderer.setPixelRatio(Math.min(window.devicePixelRatio || 1, 2));
        this.renderer.shadowMap.enabled = true;

        // OrbitControls
        if (THREE.OrbitControls) {
            this.controls = new THREE.OrbitControls(this.camera, this.renderer.domElement);
            this.controls.enableDamping = true;
            this.controls.dampingFactor = 0.08;
            this.controls.maxPolarAngle = Math.PI / 2 + 0.05;
            this.controls.target.set(0, 2, 0);
        }

        // Lighting
        const ambientLight = new THREE.AmbientLight(0xffffff, 0.65);
        this.scene.add(ambientLight);

        const dirLight = new THREE.DirectionalLight(0x38bdf8, 1.2);
        dirLight.position.set(20, 40, 20);
        dirLight.castShadow = true;
        this.scene.add(dirLight);

        const backLight = new THREE.DirectionalLight(0xf59e0b, 0.6);
        backLight.position.set(-20, 20, -20);
        this.scene.add(backLight);

        // Build anatomical fly mesh
        this.buildFlyMesh();

        // Keep missing telemetry visible without throwing away the last valid frame.
        this.poseStatus = document.createElement('div');
        this.poseStatus.style.cssText = 'display:none;margin-top:5px;padding:5px 9px;'
            + 'border:1px solid #f59e0b;border-radius:5px;background:rgba(15,23,42,.9);'
            + 'color:#fbbf24;font:10px monospace';
        (this.statusPanel || this.container).appendChild(this.poseStatus);

        this.geometryStatus = document.createElement('div');
        this.geometryStatus.style.cssText = 'margin-top:5px;padding:5px 9px;'
            + 'border:1px solid #334155;border-radius:5px;background:rgba(15,23,42,.9);'
            + 'color:#94a3b8;font:10px monospace';
        (this.statusPanel || this.container).appendChild(this.geometryStatus);
        this.updateAssayGeometry();
        this.updateAssayCues();

        // Resize handler
        window.addEventListener('resize', () => this.onResize());
        if (typeof ResizeObserver !== 'undefined') {
            this.resizeObserver = new ResizeObserver(() => this.onResize());
            this.resizeObserver.observe(this.container);
        }

        this.initialized = true;
    }

    buildFlyMesh() {
        this.flyGroup = new THREE.Group();
        this.scene.add(this.flyGroup);

        // Materials
        const thoraxMat = new THREE.MeshStandardMaterial({ color: 0x1f2937, roughness: 0.5, metalness: 0.2 });
        const headMat = new THREE.MeshStandardMaterial({ color: 0x374151, roughness: 0.4, metalness: 0.3 });
        const eyeMat = new THREE.MeshStandardMaterial({ color: 0xb91c1c, roughness: 0.2, metalness: 0.1 });
        const abdomenMat = new THREE.MeshStandardMaterial({ color: 0x111827, roughness: 0.6, metalness: 0.1 });
        const wingMat = new THREE.MeshStandardMaterial({ color: 0x7dd3fc, transparent: true, opacity: 0.45, roughness: 0.1, side: THREE.DoubleSide });
        const legMat = new THREE.MeshStandardMaterial({ color: 0x4b5563, roughness: 0.7 });

        // Thorax (center at y=2.2 mm)
        const thoraxGeo = new THREE.SphereGeometry(1.2, 16, 16);
        thoraxGeo.scale(1.0, 1.1, 1.5);
        this.thoraxMesh = new THREE.Mesh(thoraxGeo, thoraxMat);
        this.thoraxMesh.position.set(0, 2.2, 0);
        this.thoraxMesh.castShadow = true;
        this.flyGroup.add(this.thoraxMesh);

        // Head
        const headGeo = new THREE.SphereGeometry(0.8, 16, 16);
        headGeo.scale(1.2, 1.0, 0.9);
        const headMesh = new THREE.Mesh(headGeo, headMat);
        headMesh.position.set(0, 0.2, 1.6);
        this.thoraxMesh.add(headMesh);

        // Compound Eyes (left & right)
        const eyeGeo = new THREE.SphereGeometry(0.45, 12, 12);
        eyeGeo.scale(0.8, 1.2, 1.2);
        const leftEye = new THREE.Mesh(eyeGeo, eyeMat);
        leftEye.position.set(-0.65, 0.2, 0.1);
        headMesh.add(leftEye);

        const rightEye = new THREE.Mesh(eyeGeo, eyeMat);
        rightEye.position.set(0.65, 0.2, 0.1);
        headMesh.add(rightEye);

        // Abdomen (posterior)
        const abdGeo = new THREE.SphereGeometry(1.1, 16, 16);
        abdGeo.scale(0.9, 0.9, 2.0);
        const abdMesh = new THREE.Mesh(abdGeo, abdomenMat);
        abdMesh.position.set(0, -0.1, -2.4);
        this.thoraxMesh.add(abdMesh);

        // Wings (left & right)
        const wingGeo = new THREE.PlaneGeometry(1.4, 4.0);
        const leftWing = new THREE.Mesh(wingGeo, wingMat);
        leftWing.position.set(-0.8, 0.9, -1.8);
        leftWing.rotation.x = Math.PI / 2 - 0.2;
        leftWing.rotation.y = -0.25;
        this.thoraxMesh.add(leftWing);

        const rightWing = new THREE.Mesh(wingGeo, wingMat);
        rightWing.position.set(0.8, 0.9, -1.8);
        rightWing.rotation.x = Math.PI / 2 - 0.2;
        rightWing.rotation.y = 0.25;
        this.thoraxMesh.add(rightWing);

        // Build 6 articulated legs: L1, L2, L3, R1, R2, R3
        this.legs = [];
        this.contactSpheres = [];

        const legConfigs = [
            { name: 'L1', side: -1, zOffset: 0.8,  baseAngle: -Math.PI / 4 },
            { name: 'L2', side: -1, zOffset: 0.0,  baseAngle: -Math.PI / 2 },
            { name: 'L3', side: -1, zOffset: -0.8, baseAngle: -3 * Math.PI / 4 },
            { name: 'R1', side: 1,  zOffset: 0.8,  baseAngle: Math.PI / 4 },
            { name: 'R2', side: 1,  zOffset: 0.0,  baseAngle: Math.PI / 2 },
            { name: 'R3', side: 1,  zOffset: -0.8, baseAngle: 3 * Math.PI / 4 }
        ];

        legConfigs.forEach((cfg) => {
            // Coxa root group (Thorax joint)
            const coxaGroup = new THREE.Group();
            coxaGroup.position.set(cfg.side * 1.0, -0.2, cfg.zOffset);
            coxaGroup.rotation.y = cfg.baseAngle;
            this.thoraxMesh.add(coxaGroup);

            // Coxa segment mesh (length ~1.0mm)
            const coxaGeo = new THREE.CylinderGeometry(0.2, 0.18, 1.0, 8);
            coxaGeo.translate(0, -0.5, 0);
            const coxaMesh = new THREE.Mesh(coxaGeo, legMat);
            coxaGroup.add(coxaMesh);

            // Femur joint group (at tip of coxa)
            const femurGroup = new THREE.Group();
            femurGroup.position.set(0, -1.0, 0);
            coxaGroup.add(femurGroup);

            // Femur segment mesh (length ~2.2mm)
            const femurGeo = new THREE.CylinderGeometry(0.18, 0.15, 2.2, 8);
            femurGeo.translate(0, -1.1, 0);
            const femurMesh = new THREE.Mesh(femurGeo, legMat);
            femurGroup.add(femurMesh);

            // Tibia joint group (at tip of femur)
            const tibiaGroup = new THREE.Group();
            tibiaGroup.position.set(0, -2.2, 0);
            femurGroup.add(tibiaGroup);

            // Tibia segment mesh (length ~2.4mm)
            const tibiaGeo = new THREE.CylinderGeometry(0.15, 0.10, 2.4, 8);
            tibiaGeo.translate(0, -1.2, 0);
            const tibiaMesh = new THREE.Mesh(tibiaGeo, legMat);
            tibiaGroup.add(tibiaMesh);

            // Tarsus contact indicator sphere (at tip of tibia)
            const contactGeo = new THREE.SphereGeometry(0.25, 8, 8);
            const contactMat = new THREE.MeshStandardMaterial({
                color: 0x64748b,
                emissive: 0x334155,
                emissiveIntensity: 0.15,
                roughness: 0.3
            });
            const contactMesh = new THREE.Mesh(contactGeo, contactMat);
            contactMesh.position.set(0, -2.4, 0);
            tibiaGroup.add(contactMesh);

            this.legs.push({
                name: cfg.name,
                side: cfg.side,
                baseAngle: cfg.baseAngle,
                coxa: coxaGroup,
                femur: femurGroup,
                tibia: tibiaGroup,
                contact: contactMesh,
                contactMat: contactMat
            });
            this.contactSpheres.push(contactMesh);
        });
    }

    onResize() {
        if (!this.renderer || !this.container) return;
        const width = this.container.clientWidth;
        const height = this.container.clientHeight;
        if (width === 0 || height === 0) return;
        this.camera.aspect = width / height;
        this.camera.updateProjectionMatrix();
        this.renderer.setSize(width, height);
    }

    setVisible(visible) {
        const entering = visible && !this.visible;
        this.visible = visible;
        if (this.container) {
            this.container.style.display = visible ? 'block' : 'none';
        }
        if (this.statusPanel) {
            this.statusPanel.hidden = !visible;
        }
        const canvas2d = document.getElementById('arenaCanvas');
        if (canvas2d) {
            canvas2d.style.display = visible ? 'none' : 'block';
        }
        if (visible) {
            if (entering && this.cameraMode === 'orbit') this.requestOrbitFrame();
            this.onResize();
        }
    }

    setCameraMode(mode) {
        const previousMode = this.cameraMode;
        if (previousMode === 'orbit' && mode === 'chase') {
            this.savedOrbitOffset = this.currentOrbitOffset();
        }
        this.cameraMode = mode;
        const btn = document.getElementById('btnCameraMode');
        if (btn) {
            btn.textContent = mode === 'orbit' ? 'Cam: Orbit' : 'Cam: Chase';
        }
        if (this.controls) {
            this.controls.enabled = (mode === 'orbit');
        }
        if (previousMode === 'chase' && mode === 'orbit') {
            this.requestOrbitFrame(this.savedOrbitOffset);
        }
    }

    currentOrbitOffset() {
        const frame = this.camera && this.controls
            ? orbitFrameValues(this.camera.position, this.controls.target, this.controls.target)
            : null;
        return frame?.offset || null;
    }

    updateAssayCues() {
        const helper = window.NeuroFlyAssayCues3D;
        const geometry = this.assayGeometryDescriptor;
        if (!helper || !geometry) return;
        const bridge = window.hud?.daemonBridge;
        const descriptor = helper.describe(this.arena, geometry, daemonFrameOffset(this.arena.remotePacket || {}), {
            switchPending: !!bridge?.switchPending,
            ownerProblem: identityRejection(this.arena.remotePacket, bridge?.lastSwitchAck)
        });
        if (!this.cueStatus) {
            this.cueStatus = document.createElement('div');
            this.cueStatus.style.cssText = 'margin-top:5px;padding:5px 9px;color:#cbd5e1;font-size:11px';
            (this.statusPanel || this.container).appendChild(this.cueStatus);
        }
        this.cueStatus.textContent = descriptor.status;
        if (!geometry.available) { if (this.cuePlane) this.cuePlane.visible = false; return; }
        if (!this.cuePlane) {
            this.cueCanvas = document.createElement('canvas');
            this.cueCanvas.width = this.cueCanvas.height = 1024;
            this.cueTexture = new THREE.CanvasTexture(this.cueCanvas);
            const material = new THREE.MeshBasicMaterial({map:this.cueTexture,transparent:true,
                depthWrite:false,side:THREE.DoubleSide});
            this.cuePlane = new THREE.Mesh(new THREE.PlaneGeometry(1,1),material);
            this.cuePlane.rotation.x = -Math.PI/2;
            this.cuePlane.position.y = 0.04;
            this.scene.add(this.cuePlane);
        }
        this.cuePlane.visible = true;
        this.cuePlane.scale.set(geometry.width,geometry.depth,1);
        const key=JSON.stringify([this.packetIdentity(this.arena.remotePacket), geometry.bounds,descriptor]);
        if(key!==this.cueKey) {
            helper.paint(this.cueCanvas.getContext('2d'),descriptor,geometry);
            this.cueTexture.needsUpdate=true;this.cueKey=key;
        }
    }

    updateAssayGeometry() {
        const descriptor = assayGeometry3DDescriptor(this.arena);
        this.assayGeometryDescriptor = descriptor;
        if (!this.scene || typeof THREE === 'undefined') return false;
        if (descriptor.key === this.assayGeometryKey) {
            if (this.geometryStatus) this.geometryStatus.textContent = descriptor.status;
            return false;
        }
        if (this.assayGeometryGroup) {
            this.scene.remove(this.assayGeometryGroup);
            disposeThreeTree(this.assayGeometryGroup);
        }
        this.assayGeometryKey = descriptor.key;
        const group = new THREE.Group();
        group.name = 'assay-geometry-3d';
        this.assayGeometryGroup = group;
        this.scene.add(group);
        if (this.geometryStatus) this.geometryStatus.textContent = descriptor.status;
        if (!descriptor.available) return true;

        // Floors preserve the Python arena's rectangle/circle primitives. Wall height
        // and thickness remain illustrative viewing aids around its 2D segments.
        for (const surface of descriptor.surfaces) {
            const circle = surface.shape === 'circle';
            const surfaceCenter = circle
                ? arenaPointFor3D(surface.center.x, surface.center.y, descriptor.bounds)
                : arenaPointFor3D((surface.minX + surface.maxX) / 2,
                    (surface.minY + surface.maxY) / 2, descriptor.bounds);
            const floorGeometry = circle
                ? new THREE.CircleGeometry(surface.radius, 64)
                : new THREE.PlaneGeometry(surface.maxX - surface.minX, surface.maxY - surface.minY);
            const floorMaterial = new THREE.MeshStandardMaterial({
                color: 0x0b1220, roughness: 0.95, metalness: 0.0, transparent: true, opacity: 0.72,
                side: THREE.DoubleSide
            });
            const floor = new THREE.Mesh(floorGeometry, floorMaterial);
            floor.rotation.x = -Math.PI / 2;
            floor.position.set(surfaceCenter.x, -0.02, -surfaceCenter.y);
            group.add(floor);
        }

        if (descriptor.walls.length) {
            for (const wall of descriptor.walls) {
                const dx = wall.p2.x - wall.p1.x;
                const dy = wall.p2.y - wall.p1.y;
                const length = Math.hypot(dx, dy);
                const geometry = new THREE.BoxGeometry(length, 2.0, 0.45);
                const material = new THREE.MeshStandardMaterial({
                    color: 0x38bdf8, emissive: 0x0c4a6e, emissiveIntensity: 0.25,
                    roughness: 0.65, transparent: true, opacity: 0.72
                });
                const mesh = new THREE.Mesh(geometry, material);
                mesh.position.set((wall.p1.x + wall.p2.x) / 2, 1.0, -(wall.p1.y + wall.p2.y) / 2);
                mesh.rotation.y = Math.atan2(dy, dx);
                group.add(mesh);
            }
        } else if (descriptor.outline?.shape === 'circle') {
            const center = arenaPointFor3D(
                descriptor.outline.center.x, descriptor.outline.center.y, descriptor.bounds);
            const points = Array.from({length:64}, (_, i) => {
                const angle = i * 2 * Math.PI / 64;
                return new THREE.Vector3(center.x + descriptor.outline.radius * Math.cos(angle), 0.03,
                    -(center.y + descriptor.outline.radius * Math.sin(angle)));
            });
            const outlineGeometry = new THREE.BufferGeometry().setFromPoints(points);
            const outlineMaterial = new THREE.LineBasicMaterial({color: 0xf59e0b});
            group.add(new THREE.LineLoop(outlineGeometry, outlineMaterial));
        } else {
            const halfW = descriptor.width / 2;
            const halfD = descriptor.depth / 2;
            const points = [
                new THREE.Vector3(-halfW, 0.03, -halfD), new THREE.Vector3(halfW, 0.03, -halfD),
                new THREE.Vector3(halfW, 0.03, halfD), new THREE.Vector3(-halfW, 0.03, halfD)
            ];
            const outlineGeometry = new THREE.BufferGeometry().setFromPoints(points);
            const outlineMaterial = new THREE.LineBasicMaterial({color: 0xf59e0b});
            group.add(new THREE.LineLoop(outlineGeometry, outlineMaterial));
        }
        return true;
    }

    requestOrbitFrame(offset = null) {
        this.pendingOrbitFrame = true;
        this.pendingOrbitOffset = offset;
    }

    applyPendingOrbitFrame() {
        if (!this.pendingOrbitFrame || this.cameraMode !== 'orbit' || !this.controls
                || !this.lastPose || !this.flyGroup) return false;
        const frame = orbitFrameValues(
            this.camera.position, this.controls.target, this.flyGroup.position, this.pendingOrbitOffset);
        if (!frame) return false;
        this.controls.target.set(frame.target.x, frame.target.y, frame.target.z);
        this.camera.position.set(frame.camera.x, frame.camera.y, frame.camera.z);
        this.pendingOrbitFrame = false;
        this.pendingOrbitOffset = null;
        return true;
    }

    packetIdentity(pkt) {
        const id = pkt?.identity || {};
        const hasIdentity = pkt?.type === 'telemetry' && (pkt.segment_id || pkt.run_id
            || id.run_id || id.instance_id || Number.isInteger(id.activation));
        return hasIdentity
            ? JSON.stringify([id.run_id || pkt.run_id || '', id.instance_id || '',
                Number.isInteger(id.activation) ? id.activation : null, pkt.segment_id || ''])
            : `local:${this.arena.activeParadigmId || ''}`;
    }

    cameraIdentity(pkt) {
        const id = pkt?.identity || {};
        const hasIdentity = pkt?.type === 'telemetry' && (pkt.run_id
            || id.run_id || id.instance_id || Number.isInteger(id.activation));
        return hasIdentity
            ? JSON.stringify([id.run_id || pkt.run_id || '', id.instance_id || '',
                Number.isInteger(id.activation) ? id.activation : null])
            : `local:${this.arena.activeParadigmId || ''}`;
    }

    syncCameraIdentity(pkt) {
        const identity = this.cameraIdentity(pkt);
        if (identity === this.orbitFrameIdentity) return false;
        this.orbitFrameIdentity = identity;
        this.requestOrbitFrame();
        return true;
    }

    resetPoseCache(identity) {
        this.cacheIdentity = identity;
        this.lastPose = null;
        this.lastHeading = null;
        this.lastBodyZ = null;
        this.lastJointAngles = null;
        this.lastLegContacts = null;
        if (this.flyGroup) {
            this.flyGroup.position.set(0, 0, 0);
            this.flyGroup.rotation.set(0, 0, 0);
        }
        for (const leg of this.legs || []) {
            leg.coxa.rotation.y = leg.baseAngle;
            leg.femur.rotation.z = 0;
            leg.tibia.rotation.z = 0;
            leg.contactMat.color.setHex(0x64748b);
            leg.contactMat.emissive.setHex(0x334155);
            leg.contactMat.emissiveIntensity = 0.15;
        }
    }

    updatePose() {
        if (!this.initialized || !this.flyGroup) return;

        const pkt = this.arena.remotePacket || {};
        const fly = this.arena.fly || {};
        this.updateAssayGeometry();
        this.updateAssayCues();
        this.syncCameraIdentity(pkt);
        const identity = this.packetIdentity(pkt);
        if (identity !== this.cacheIdentity) this.resetPoseCache(identity);
        const remote = pkt.type === 'telemetry';
        const currentFrame = !this.arena.awaitingDaemon;
        const validPosition = currentFrame && Number.isFinite(fly.x) && Number.isFinite(fly.y);
        const validHeading = currentFrame && Number.isFinite(fly.heading)
            && (!remote || Number.isFinite(pkt.fly?.heading));
        const validBodyZ = currentFrame && Array.isArray(pkt.body_position_mm)
            && pkt.body_position_mm.length >= 3 && pkt.body_position_mm.slice(0, 3).every(Number.isFinite);
        const validJoints = currentFrame && Array.isArray(pkt.joint_angles_rad)
            && pkt.joint_angles_rad.length >= 18 && pkt.joint_angles_rad.slice(0, 18).every(Number.isFinite);
        const validContacts = currentFrame && Array.isArray(pkt.leg_contacts)
            && pkt.leg_contacts.length >= 6
            && pkt.leg_contacts.slice(0, 6).every(v => typeof v === 'boolean');
        if (validPosition) {
            const point = arenaPointFor3D(fly.x, fly.y, this.assayGeometryDescriptor?.bounds);
            if (point) this.lastPose = {xMm: point.x, yMm: point.y};
        }
        if (validHeading) this.lastHeading = fly.heading;
        if (validBodyZ) this.lastBodyZ = pkt.body_position_mm[2];
        if (validJoints) this.lastJointAngles = pkt.joint_angles_rad.slice(0, 18);
        if (validContacts) this.lastLegContacts = pkt.leg_contacts.slice(0, 6);

        const channels = [
            ['XY', validPosition, this.lastPose !== null],
            ['heading', validHeading, this.lastHeading !== null],
            ['body height', validBodyZ, this.lastBodyZ !== null],
            ['joints', validJoints, this.lastJointAngles !== null],
            ['contacts', validContacts, this.lastLegContacts !== null]
        ];
        const held = channels.filter(([, current, cached]) => !current && cached).map(([name]) => name);
        const unavailable = channels.filter(([, current, cached]) => !current && !cached).map(([name]) => name);
        if (this.poseStatus) {
            this.poseStatus.style.display = held.length || unavailable.length ? 'block' : 'none';
            this.poseStatus.textContent = [
                held.length ? `HELD: ${held.join(', ')}` : '',
                unavailable.length ? `UNAVAILABLE: ${unavailable.join(', ')}` : ''
            ].filter(Boolean).join(' · ');
        }
        const dataStatus = document.getElementById('v3dDataStatus');
        if (dataStatus) {
            const xyLabel = validPosition ? 'arena XY stream current'
                : this.lastPose ? 'arena XY stream held' : 'arena XY unavailable';
            const poseLabel = remote
                ? `${xyLabel} + ${validBodyZ ? 'body-height model stream current' : this.lastBodyZ !== null ? 'body-height model stream held' : 'illustrative lift'}`
                : '2D preview pose + illustrative lift';
            const limbLabel = validJoints ? '18-DOF model stream current'
                : this.lastJointAngles ? '18-DOF model stream held' : 'joint stream unavailable';
            const contactLabel = validContacts ? 'contact model stream current'
                : this.lastLegContacts ? 'contact model stream held' : 'contact stream unavailable (gray)';
            dataStatus.textContent = `Illustrative 3D view · ${poseLabel} · ${limbLabel} · ${contactLabel} · Drag to Orbit`;
        }
        // A missing frame must not replace the last measured pose with made-up
        // coordinates. If no valid pose has ever arrived, leave the mesh at its
        // initialized position until the arena supplies one.
        if (!this.lastPose) return;
        const {xMm, yMm} = this.lastPose;
        const zMm = this.lastBodyZ ?? 0.5; // Illustrative lift when no 3D body height was measured.
        this.flyGroup.position.set(xMm, zMm + 1.2, -yMm);
        if (this.lastHeading !== null) {
            this.flyGroup.rotation.set(0, -this.lastHeading + Math.PI / 2, 0);
        }
        if (validPosition) this.applyPendingOrbitFrame();
        const anglesRad = this.lastJointAngles;
        const contacts = this.lastLegContacts;

        for (let i = 0; i < 6; i++) {
            const leg = this.legs[i];
            if (!leg) continue;

            if (anglesRad) {
                const coxaRad = anglesRad[i * 3 + 0];
                const femurRad = anglesRad[i * 3 + 1];
                const tibiaRad = anglesRad[i * 3 + 2];
                leg.coxa.rotation.y = leg.baseAngle + coxaRad * 0.8;
                leg.femur.rotation.z = leg.side * (0.35 + femurRad * 0.5);
                leg.tibia.rotation.z = -leg.side * (0.6 + (tibiaRad - 1.4) * 0.6);
            }
            if (contacts) {
                const isStance = contacts[i];
                const targetColor = isStance ? 0x22c55e : 0x38bdf8;
                leg.contactMat.color.setHex(targetColor);
                leg.contactMat.emissive.setHex(targetColor);
                leg.contactMat.emissiveIntensity = isStance ? 0.9 : 0.4;
            }
        }

        if (this.cameraMode === 'chase' && this.lastHeading !== null) {
            const chaseDist = 20.0;
            const chaseHeight = 10.0;
            const camX = this.flyGroup.position.x - Math.cos(-this.lastHeading + Math.PI / 2) * chaseDist;
            const camZ = this.flyGroup.position.z + Math.sin(-this.lastHeading + Math.PI / 2) * chaseDist;
            const camY = this.flyGroup.position.y + chaseHeight;

            this.camera.position.lerp(new THREE.Vector3(camX, camY, camZ), 0.1);
            this.camera.lookAt(this.flyGroup.position);
        } else if (this.controls && this.controls.enabled) {
            this.controls.update();
        }
    }

    render() {
        if (!this.initialized || !this.visible) return;
        this.updatePose();
        this.renderer.render(this.scene, this.camera);
    }
}

function startNeuroflyApp() {
    const arena = new ScientificBioArena('arenaCanvas');
    const hud = new ScientificHUD(arena);
    window.arena = arena;
    window.hud = hud;

    // Phase 3 Step 3.2: Initialize 3D Articulated Viewport
    let viewport3D = null;
    try {
        viewport3D = new ArticulatedFly3DViewport('viewport3DCanvas', 'viewport3DContainer', arena);
        window.viewport3D = viewport3D;
    } catch (e) {
        console.warn('[3D Viewport] Initialization failed:', e);
    }

    const btnToggle3D = document.getElementById('btnToggle3D');
    const btnCameraMode = document.getElementById('btnCameraMode');
    if (btnToggle3D) {
        btnToggle3D.addEventListener('click', () => {
            if (!viewport3D) return;
            const nextVisible = !viewport3D.visible;
            viewport3D.setVisible(nextVisible);
            btnToggle3D.textContent = nextVisible ? 'View: 3D Viewport' : 'View: 2D Arena';
            btnToggle3D.style.borderColor = nextVisible ? '#22c55e' : '#38bdf8';
            btnToggle3D.style.color = nextVisible ? '#22c55e' : '#38bdf8';
            if (btnCameraMode) {
                btnCameraMode.style.display = nextVisible ? 'inline-block' : 'none';
            }
        });
    }

    if (btnCameraMode) {
        btnCameraMode.addEventListener('click', () => {
            if (!viewport3D) return;
            const nextMode = viewport3D.cameraMode === 'orbit' ? 'chase' : 'orbit';
            viewport3D.setCameraMode(nextMode);
        });
    }

    let isPaused = false;
    const btnPause = document.getElementById('btnPauseToggle');
    if (btnPause) {
        btnPause.addEventListener('click', () => {
            if (hud.daemonBridge?.replayMode) {
                window.neuroflyReplay?.toggle();
                return;
            }
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
        viewport3D,
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

                const stepDt = 0.02;
                const maxStepsPerFrame = Math.max(160, Math.ceil(speed * 1.8));
                let stepsExecuted = 0;

                while (accumulator >= stepDt && stepsExecuted < maxStepsPerFrame) {
                    arena.step(stepDt);
                    accumulator -= stepDt;
                    stepsExecuted++;
                }
                if (stepsExecuted >= maxStepsPerFrame) {
                    accumulator = 0;
                }
                hud.update();
            });
        }
        runPhase('render', () => {
            if (renderFaultPending && arena.remotePacket) {
                renderFaultPending = false;
                throw new Error('Injected renderer fault (test build)');
            }
            if (viewport3D && viewport3D.visible) {
                viewport3D.render();
            } else {
                arena.render();
            }
        });
        requestAnimationFrame(loop);
    }
    requestAnimationFrame(loop);
}
