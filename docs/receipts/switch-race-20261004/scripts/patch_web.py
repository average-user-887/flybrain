from pathlib import Path
p = Path('web/app.js'); s = p.read_text()


def rep(old, new):
    global s
    assert s.count(old) == 1, old
    s = s.replace(old, new)


rep("""        this.lastHeartbeat = null;          // {step_in_progress_s, last_step_wall_s, at}
""", """        this.lastHeartbeat = null;          // {step_in_progress_s, last_step_wall_s, at}
        this.daemonHalt = null;             // {error, detail} while a step error halts the daemon
""")
rep("""        this.lastPacketTime = performance.now();
        this.lastOrderedPacket = pkt;
        renderIdentity(pkt);""", """        this.lastPacketTime = performance.now();
        this.lastOrderedPacket = pkt;
        // A step error halts the daemon (nothing advances) until a switch rebuilds it.
        // Frames still arrive, so without this the pill would read LIVE over a frozen run.
        const halt = (pkt.halted || pkt.error) ? {error: pkt.error || 'unknown error', detail: pkt.error_detail || null} : null;
        const haltChanged = (halt?.error || null) !== (this.daemonHalt?.error || null);
        this.daemonHalt = halt;
        renderIdentity(pkt);""")
rep("""        this.arena.remoteDriven = poseMatch;

        if (poseMatch) {""", """        this.arena.remoteDriven = poseMatch;
        if (haltChanged) this.updateFreshness();

        if (poseMatch) {""")
rep("""        const slowStep = stale ? this.slowStepSeconds() : null;
        const state = slowStep !== null ? 'slow' : stale ? 'stale' : 'live';
        const ro = this.readOnly ? ' (READ-ONLY)' : '';
        if (state === 'slow') {""", """        const slowStep = stale ? this.slowStepSeconds() : null;
        const halt = this.daemonHalt;
        const state = halt ? 'error' : slowStep !== null ? 'slow' : stale ? 'stale' : 'live';
        const ro = this.readOnly ? ' (READ-ONLY)' : '';
        if (state === 'error') {
            // Connected and fresh, but the simulation does not advance: never show LIVE.
            this.statusPill.textContent = `● SIMULATION HALTED${ro} · ERROR`;
            this.statusPill.title = `The daemon is connected but the simulation is halted and not advancing: `
                + `${halt.error}` + (halt.detail?.paradigm ? ` (assay ${halt.detail.paradigm}, step ${halt.detail.step}). ` : '. ')
                + (halt.detail?.recover || 'Select an assay to rebuild the controller and resume.');
        }
        if (state === 'slow') {""")
rep("""        if (state === this.freshnessState && state !== 'slow') return;
        this.freshnessState = state;
        this.showingStale = state === 'stale';
        if (state !== 'slow') {""", """        if (state === this.freshnessState && state !== 'slow' && state !== 'error') return;
        this.freshnessState = state;
        this.showingStale = state === 'stale';
        if (state === 'live' || state === 'stale') {""")
rep("""        const color = {live: ['#4ade80', '#22c55e'], stale: ['#fbbf24', '#f59e0b'], slow: ['#38bdf8', '#0ea5e9']}[state];""",
    """        const color = {live: ['#4ade80', '#22c55e'], stale: ['#fbbf24', '#f59e0b'], slow: ['#38bdf8', '#0ea5e9'],
                       error: ['#f87171', '#ef4444']}[state];""")
p.write_text(s)
print("patched")
