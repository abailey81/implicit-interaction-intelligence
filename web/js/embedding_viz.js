/**
 * Canvas-based 2D embedding visualisation.
 *
 * Draws a scatter plot of the user's state trajectory on a 2D canvas,
 * with cluster labels, point trails, and a glowing current-state marker.
 */

class EmbeddingViz {
    /**
     * @param {HTMLCanvasElement} canvas  The target canvas element.
     */
    constructor(canvas) {
        this.canvas = canvas;
        this.ctx = canvas.getContext('2d');
        this.points = [];         // { x, y, timestamp, age }
        this.current = null;      // { x, y }
        this.maxPoints = 60;
        this.animFrame = null;

        // Approximate cluster label positions (in normalised [-1, 1] space)
        this.stateLabels = {
            'Energetic': { x: 0.6, y: -0.5 },
            'Tired':     { x: -0.6, y: 0.5 },
            'Focused':   { x: 0.3, y: -0.7 },
            'Relaxed':   { x: -0.2, y: -0.2 },
            'Stressed':  { x: 0.5, y: 0.6 },
            'Difficulty': { x: -0.7, y: 0.7 },
        };

        this._setupCanvas();
        this.draw();
    }

    /**
     * Set up the canvas for high-DPI rendering.
     * @private
     */
    _setupCanvas() {
        const resize = () => {
            const rect = this.canvas.parentElement.getBoundingClientRect();
            const dpr = window.devicePixelRatio || 1;
            this.canvas.width = rect.width * dpr;
            this.canvas.height = 200 * dpr;
            this.canvas.style.width = rect.width + 'px';
            this.canvas.style.height = '200px';
            this.ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
            this.draw();
        };

        resize();
        window.addEventListener('resize', resize);
    }

    /**
     * Convert normalised coordinates [-1, 1] to canvas pixel coordinates.
     * @param {number} nx  Normalised x.
     * @param {number} ny  Normalised y.
     * @returns {{ px: number, py: number }}
     * @private
     */
    _toPixel(nx, ny) {
        const w = this.canvas.clientWidth;
        const h = this.canvas.clientHeight;
        const margin = 24;
        const px = margin + ((nx + 1) / 2) * (w - 2 * margin);
        const py = margin + ((ny + 1) / 2) * (h - 2 * margin);
        return { px, py };
    }

    /**
     * Add a new 2D point from the server state update.
     * @param {Array|{x: number, y: number}} point2d  The 2D embedding.
     */
    update(point2d) {
        if (!point2d) return;

        let x, y;
        if (Array.isArray(point2d)) {
            [x, y] = point2d;
        } else if (typeof point2d === 'object') {
            x = point2d.x !== undefined ? point2d.x : point2d[0];
            y = point2d.y !== undefined ? point2d.y : point2d[1];
        } else {
            return;
        }

        // SEC: Coerce to Number and reject NaN/Infinity before clamping.
        // Math.min/max(NaN) returns NaN, which would corrupt canvas state.
        x = Number(x);
        y = Number(y);
        if (!Number.isFinite(x) || !Number.isFinite(y)) return;

        // Clamp to [-1, 1]
        x = Math.max(-1, Math.min(1, x));
        y = Math.max(-1, Math.min(1, y));

        this.points.push({ x, y, timestamp: Date.now() });

        // SEC: Hard cap on points array (defence-in-depth alongside maxPoints)
        // to ensure no memory leak from a misbehaving server.
        while (this.points.length > this.maxPoints) {
            this.points.shift();
        }

        this.current = { x, y };
        this.draw();
    }

    /**
     * Render the full visualisation.
     */
    draw() {
        const ctx = this.ctx;
        const w = this.canvas.clientWidth;
        const h = this.canvas.clientHeight;

        // Clear with dark background
        ctx.fillStyle = '#0d1b30';
        ctx.fillRect(0, 0, w, h);

        // Draw grid
        this._drawGrid(ctx, w, h);

        // Draw state labels
        this._drawLabels(ctx);

        // Draw trail and points
        this._drawTrail(ctx);
        this._drawPoints(ctx);

        // Draw current point with glow
        this._drawCurrent(ctx);
    }

    /** @private */
    _drawGrid(ctx, w, h) {
        ctx.strokeStyle = 'rgba(255, 255, 255, 0.04)';
        ctx.lineWidth = 0.5;

        const steps = 8;
        for (let i = 0; i <= steps; i++) {
            const frac = i / steps;

            // Vertical
            const vx = 24 + frac * (w - 48);
            ctx.beginPath();
            ctx.moveTo(vx, 24);
            ctx.lineTo(vx, h - 24);
            ctx.stroke();

            // Horizontal
            const hy = 24 + frac * (h - 48);
            ctx.beginPath();
            ctx.moveTo(24, hy);
            ctx.lineTo(w - 24, hy);
            ctx.stroke();
        }

        // Axis lines (slightly brighter)
        ctx.strokeStyle = 'rgba(255, 255, 255, 0.08)';
        ctx.lineWidth = 1;

        // Horizontal center
        const cy = h / 2;
        ctx.beginPath();
        ctx.moveTo(24, cy);
        ctx.lineTo(w - 24, cy);
        ctx.stroke();

        // Vertical center
        const cx = w / 2;
        ctx.beginPath();
        ctx.moveTo(cx, 24);
        ctx.lineTo(cx, h - 24);
        ctx.stroke();
    }

    /** @private */
    _drawLabels(ctx) {
        ctx.font = '9px system-ui, sans-serif';
        ctx.textAlign = 'center';
        ctx.textBaseline = 'middle';

        for (const [label, pos] of Object.entries(this.stateLabels)) {
            const { px, py } = this._toPixel(pos.x, pos.y);
            ctx.fillStyle = 'rgba(160, 160, 176, 0.35)';
            ctx.fillText(label, px, py);
        }
    }

    /** @private */
    _drawTrail(ctx) {
        if (this.points.length < 2) return;

        ctx.lineWidth = 1;
        ctx.lineCap = 'round';
        ctx.lineJoin = 'round';

        for (let i = 1; i < this.points.length; i++) {
            const p0 = this.points[i - 1];
            const p1 = this.points[i];
            const age = 1 - (i / this.points.length);

            const from = this._toPixel(p0.x, p0.y);
            const to = this._toPixel(p1.x, p1.y);

            ctx.strokeStyle = `rgba(58, 123, 213, ${0.1 + age * 0.25})`;
            ctx.beginPath();
            ctx.moveTo(from.px, from.py);
            ctx.lineTo(to.px, to.py);
            ctx.stroke();
        }
    }

    /** @private */
    _drawPoints(ctx) {
        const len = this.points.length;
        for (let i = 0; i < len; i++) {
            const p = this.points[i];
            const { px, py } = this._toPixel(p.x, p.y);
            const alpha = 0.1 + (i / len) * 0.5;
            const radius = 1.5 + (i / len) * 1.5;

            ctx.beginPath();
            ctx.arc(px, py, radius, 0, Math.PI * 2);
            ctx.fillStyle = `rgba(58, 123, 213, ${alpha})`;
            ctx.fill();
        }
    }

    /** @private */
    _drawCurrent(ctx) {
        if (!this.current) return;

        const { px, py } = this._toPixel(this.current.x, this.current.y);

        // Outer glow
        const gradient = ctx.createRadialGradient(px, py, 0, px, py, 14);
        gradient.addColorStop(0, 'rgba(233, 69, 96, 0.5)');
        gradient.addColorStop(0.5, 'rgba(233, 69, 96, 0.15)');
        gradient.addColorStop(1, 'rgba(233, 69, 96, 0)');
        ctx.beginPath();
        ctx.arc(px, py, 14, 0, Math.PI * 2);
        ctx.fillStyle = gradient;
        ctx.fill();

        // Core dot
        ctx.beginPath();
        ctx.arc(px, py, 4, 0, Math.PI * 2);
        ctx.fillStyle = '#e94560';
        ctx.fill();

        // White center
        ctx.beginPath();
        ctx.arc(px, py, 1.5, 0, Math.PI * 2);
        ctx.fillStyle = '#ffffff';
        ctx.fill();
    }
}
