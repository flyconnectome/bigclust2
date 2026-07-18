/* The living embedding.
 *
 * ~9,000 GPU points that morph between six different 2D layouts of the *same*
 * observations - clusters condensing, dissolving and re-forming. One draw call,
 * no dependencies.
 *
 * The subject is the product: this is what pressing Space in BigClust does, and
 * why the app animates layout transitions instead of cutting between them. A
 * cluster that survives the morph is real; one that only exists in a single
 * layout is an artefact. That is the whole argument of the docs, running in the
 * header.
 *
 * Loaded on every page (so instant navigation can never swap it out) and gated on
 * the presence of #bc-clusters, which only the landing page renders. A
 * MutationObserver re-initialises when instant navigation swaps the container in
 * or out.
 */
(function () {
  "use strict";

  var N_CLUSTERS = 14;
  var N_LAYOUTS = 6;

  // Seconds held on each layout, and seconds spent in transit to the next. The
  // hold is deliberately short - the morph is the interesting part, and a long
  // pause on a static cloud just reads as a stalled animation.
  var HOLD_DUR = 2.0;
  var TRANS_DUR = 2.9;
  var STEP_DUR = HOLD_DUR + TRANS_DUR;

  var VERT = [
    "precision highp float;",

    // The same point in all six layouts. Interleaved into one buffer; which two
    // are live is a pair of one-hot weight arrays, which keeps the shader
    // branch-free - no indexing into attributes, no conditionals per vertex.
    "attribute vec2 aP0;",
    "attribute vec2 aP1;",
    "attribute vec2 aP2;",
    "attribute vec2 aP3;",
    "attribute vec2 aP4;",
    "attribute vec2 aP5;",
    // x: cluster index, y: per-point seed 0..1, z: size, w: depth 0..1
    "attribute vec4 aMeta;",

    "uniform vec2 uRes;",
    "uniform float uTime;",
    "uniform vec2 uMouse;",
    "uniform float uWFrom[6];",
    "uniform float uWTo[6];",
    "uniform float uPhase;",
    "uniform vec2 uOrigin;",
    "uniform float uScale;",
    "uniform float uPointScale;",

    "varying vec3 vCol;",
    "varying float vAlpha;",

    // Cosine gradient (Inigo Quilez) - cheaper than a lookup table and it makes
    // the whole 14-colour set from four constants. The amplitude is pushed close
    // to the offset so each channel swings nearly the full 0..1 range, which is
    // what gives the clusters their saturation on a near-black field.
    "vec3 palette(float t) {",
    "  vec3 a = vec3(0.52, 0.46, 0.45);",
    "  vec3 b = vec3(0.48, 0.48, 0.50);",
    "  vec3 c = vec3(1.00, 1.00, 1.00);",
    "  vec3 d = vec3(0.02, 0.30, 0.63);",
    "  return a + b * cos(6.28318 * (c * t + d));",
    "}",

    "void main() {",
    "  vec2 pFrom = aP0 * uWFrom[0] + aP1 * uWFrom[1] + aP2 * uWFrom[2]",
    "             + aP3 * uWFrom[3] + aP4 * uWFrom[4] + aP5 * uWFrom[5];",
    "  vec2 pTo   = aP0 * uWTo[0]   + aP1 * uWTo[1]   + aP2 * uWTo[2]",
    "             + aP3 * uWTo[3]   + aP4 * uWTo[4]   + aP5 * uWTo[5];",

    // Per-point stagger. Without it every point starts and stops together and the
    // cloud slides like a single rigid object; with it the clusters visibly come
    // apart and knit back together, which is the thing worth watching.
    "  float S = 0.55;",
    "  float t = clamp((uPhase - aMeta.y * S) / (1.0 - S), 0.0, 1.0);",
    "  t = t * t * (3.0 - 2.0 * t);",
    "  t = t * t * (3.0 - 2.0 * t);",

    "  vec2 p = mix(pFrom, pTo, t);",

    // Bow each path sideways so points sweep rather than slide on rails. The
    // magnitude scales with travel distance, so points that barely move stay put.
    "  vec2 dir = pTo - pFrom;",
    "  float travel = length(dir);",
    "  if (travel > 0.0001) {",
    "    vec2 nrm = vec2(-dir.y, dir.x) / travel;",
    "    float bend = (aMeta.y - 0.5) * 2.0;",
    "    p += nrm * sin(3.14159 * t) * bend * min(travel, 1.2) * 0.16;",
    "  }",

    // A slow idle drift, so a held layout is never completely static.
    "  float ph = aMeta.y * 6.28318 + uTime * 0.22;",
    "  p += vec2(cos(ph), sin(ph * 1.31)) * 0.008;",

    // Parallax: near points shift more than far ones.
    "  p += uMouse * 0.045 * (0.35 + aMeta.w);",

    "  vec2 pos = uOrigin + p * uScale;",
    "  float aspect = uRes.x / uRes.y;",
    "  gl_Position = vec4(pos.x / aspect, pos.y, 0.0, 1.0);",

    "  float inTransit = sin(3.14159 * t);",
    "  gl_PointSize = uPointScale * aMeta.z * (0.55 + aMeta.w) * (1.0 + 0.22 * inTransit);",

    // Golden-ratio hue spacing. Hashing the cluster index is the obvious move and
    // is worse: being random, it clumps, and with 14 clusters it dropped eight of
    // them into the same magenta wedge. This low-discrepancy sequence bounds the
    // largest hue gap at ~44 degrees, which is what lets the clusters be told
    // apart at a glance. Cluster 0 lands on the site's accent orange.
    "  vec3 col = clamp(palette(fract(aMeta.x * 0.6180339887)), 0.0, 1.0);",
    // Gamma lift: brightens the mid-tones without pulling the hues back toward
    // white the way a straight multiply would.
    "  vCol = pow(col, vec3(0.88));",

    "  vAlpha = 0.70 + 0.26 * aMeta.w + 0.16 * inTransit;",
    "}"
  ].join("\n");

  var FRAG = [
    "precision mediump float;",
    "varying vec3 vCol;",
    "varying float vAlpha;",
    "void main() {",
    "  vec2 d = gl_PointCoord - 0.5;",
    "  float r2 = dot(d, d) * 4.0;",
    "  if (r2 > 1.0) discard;",
    // Gaussian rather than a hard disc: at 2-3px a hard-edged circle aliases
    // badly, and the soft falloff is what makes dense cluster cores glow when
    // hundreds of these land on top of each other.
    "  float a = exp(-r2 * 3.4) * vAlpha;",
    "  gl_FragColor = vec4(vCol * a, a);",
    "}"
  ].join("\n");

  var state = null;

  // ---------------------------------------------------------------- layouts

  // Deterministic RNG - the same cloud every visit, so the hero is a fixed piece
  // of artwork rather than a lottery that is occasionally ugly.
  function mulberry32(seed) {
    return function () {
      seed |= 0;
      seed = (seed + 0x6d2b79f5) | 0;
      var t = Math.imul(seed ^ (seed >>> 15), 1 | seed);
      t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
      return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
    };
  }

  function gauss(rnd) {
    var u = 1 - rnd();
    var v = rnd();
    return Math.sqrt(-2 * Math.log(u)) * Math.cos(6.28318 * v);
  }

  /* Six layouts of the same points, each one a shape you actually meet in real
     embeddings. Coordinates are roughly in [-1, 1]; the shader scales them.

     The cycle order matters as much as the shapes: it runs separated ->
     evenly-separated -> grouped -> radial -> arc -> dissolved, so consecutive
     transitions never look like the same move twice. */
  function buildLayouts(n, rnd) {
    var L = [];
    for (var k = 0; k < N_LAYOUTS; k++) L.push(new Float32Array(n * 2));

    // Per-cluster parameters, shared across layouts so a cluster keeps its
    // identity as it moves.
    var cx = [], cy = [], sig = [], elong = [], rot = [];
    var i, c;
    for (c = 0; c < N_CLUSTERS; c++) {
      var ang = (c / N_CLUSTERS) * 6.28318 + rnd() * 0.5;
      var rad = 0.34 + rnd() * 0.56;
      cx.push(Math.cos(ang) * rad);
      cy.push(Math.sin(ang) * rad * 0.86);
      sig.push(0.045 + rnd() * 0.075);
      elong.push(1.0 + rnd() * 2.4);
      rot.push(rnd() * 3.14159);
    }

    var HALF = N_CLUSTERS / 2;

    for (i = 0; i < n; i++) {
      c = i % N_CLUSTERS;
      var s = sig[c];
      var e = elong[c];
      var ca = Math.cos(rot[c]);
      var sa = Math.sin(rot[c]);

      // 0 - "UMAP": anisotropic blobs, well separated, a few stragglers.
      var gx = gauss(rnd) * s * e;
      var gy = gauss(rnd) * s;
      var stray = rnd() < 0.03 ? 3.2 : 1.0;
      L[0][i * 2] = cx[c] + (gx * ca - gy * sa) * stray;
      L[0][i * 2 + 1] = cy[c] + (gx * sa + gy * ca) * stray;

      // 1 - "t-SNE": rounder and tighter, spaced by golden angle on a sunflower
      // so the clusters fill the disc evenly instead of leaving a donut hole -
      // which is both what t-SNE output actually looks like and the reason its
      // between-cluster distances mean so much less than UMAP's.
      var a1 = c * 2.39996;
      var r1 = 0.16 + 0.78 * Math.sqrt((c + 0.5) / N_CLUSTERS);
      L[1][i * 2] = Math.cos(a1) * r1 + gauss(rnd) * s * 0.58;
      L[1][i * 2 + 1] = Math.sin(a1) * r1 * 0.92 + gauss(rnd) * s * 0.58;

      // 2 - "two lobes": clusters gathered into two superclusters, the shape you
      // get whenever one split dominates everything else - hemispheres, or
      // excitatory versus inhibitory.
      var lobe = c < HALF ? -1 : 1;
      var wi = c % HALF;
      var a2 = (wi / HALF) * 6.28318 + (lobe > 0 ? 0.42 : 0.0);
      L[2][i * 2] = lobe * 0.50 + Math.cos(a2) * 0.30 + gauss(rnd) * s * 0.85;
      L[2][i * 2 + 1] = Math.sin(a2) * 0.30 * 1.05 + gauss(rnd) * s * 0.85;

      // 3 - "filaments": each cluster smeared along an arc of a spiral, the way
      // morphology embeddings tend to look.
      var u = rnd();
      var a3 = (c / N_CLUSTERS) * 6.28318 * 1.15 + u * 0.62;
      var r3 = 0.20 + u * 0.78 + (c % 4) * 0.045;
      L[3][i * 2] = Math.cos(a3) * r3 + gauss(rnd) * 0.024;
      L[3][i * 2 + 1] = Math.sin(a3) * r3 * 0.92 + gauss(rnd) * 0.024;

      // 4 - "horseshoe": clusters strung along a C. The arch effect is a real and
      // very common ordination artefact - one dominant gradient bent into a
      // crescent - so it belongs in an honest tour of embedding shapes.
      var hu = (c + 0.5) / N_CLUSTERS;
      var ha = -2.44 + hu * 4.88 + (rnd() - 0.5) * 0.20;
      var hr = 0.70 + 0.14 * Math.sin(hu * 3.14159);
      L[4][i * 2] = Math.cos(ha) * hr + gauss(rnd) * 0.032;
      L[4][i * 2 + 1] = Math.sin(ha) * hr * 0.95 + gauss(rnd) * 0.032;

      // 5 - "continuum": one swirl, clusters bleeding into each other. What an
      // embedding looks like when there is no clean structure to find.
      var v = (c + rnd()) / N_CLUSTERS;
      var a5 = v * 6.28318 * 1.9;
      var r5 = 0.16 + v * 0.82;
      L[5][i * 2] = Math.cos(a5) * r5 + gauss(rnd) * 0.055;
      L[5][i * 2 + 1] = Math.sin(a5) * r5 * 0.88 + gauss(rnd) * 0.055;
    }
    return L;
  }

  // ------------------------------------------------------------------- gl

  function compile(gl, type, src) {
    var s = gl.createShader(type);
    gl.shaderSource(s, src);
    gl.compileShader(s);
    if (!gl.getShaderParameter(s, gl.COMPILE_STATUS)) {
      console.error("[bigclust] shader:", gl.getShaderInfoLog(s));
      gl.deleteShader(s);
      return null;
    }
    return s;
  }

  function teardown() {
    if (!state) return;
    if (state.raf) cancelAnimationFrame(state.raf);
    window.removeEventListener("resize", state.onResize);
    window.removeEventListener("pointermove", state.onMove);
    if (state.io) state.io.disconnect();
    state = null;
  }

  function init() {
    var canvas = document.getElementById("bc-clusters");
    if (!canvas) {
      teardown();
      return;
    }
    if (state && state.canvas === canvas) return;
    teardown();

    var hero = canvas.closest(".bc-hero");
    var gl =
      canvas.getContext("webgl", { alpha: true, antialias: false, premultipliedAlpha: true }) ||
      canvas.getContext("experimental-webgl", { alpha: true, antialias: false });

    // No WebGL: the CSS fallback (the blurred screenshot) is what is already
    // showing, so leave it rather than covering it with a dead black rectangle.
    if (!gl) return;

    // Six vec2 layouts plus the meta vec4 is seven attributes. WebGL guarantees
    // at least eight, so this is at the edge of what is portable - another layout
    // would have to pack into a texture instead.
    var vs = compile(gl, gl.VERTEX_SHADER, VERT);
    var fs = compile(gl, gl.FRAGMENT_SHADER, FRAG);
    if (!vs || !fs) return;

    var prog = gl.createProgram();
    gl.attachShader(prog, vs);
    gl.attachShader(prog, fs);
    gl.linkProgram(prog);
    if (!gl.getProgramParameter(prog, gl.LINK_STATUS)) {
      console.error("[bigclust] link:", gl.getProgramInfoLog(prog));
      return;
    }
    gl.useProgram(prog);

    // Scale the cloud to the viewport: a phone does not need nine thousand
    // points, and it is the one place where fill rate actually bites.
    var wide = window.innerWidth >= 900;
    var n = wide ? 9000 : 4200;

    var rnd = mulberry32(0x8c1a57);
    var L = buildLayouts(n, rnd);

    // Interleaved: p0.xy .. p5.xy meta.xyzw = 16 floats per point.
    var STRIDE = 16;
    var data = new Float32Array(n * STRIDE);
    for (var i = 0; i < n; i++) {
      var o = i * STRIDE;
      for (var k = 0; k < N_LAYOUTS; k++) {
        data[o + k * 2] = L[k][i * 2];
        data[o + k * 2 + 1] = L[k][i * 2 + 1];
      }
      data[o + 12] = i % N_CLUSTERS;
      data[o + 13] = rnd();
      data[o + 14] = 0.75 + rnd() * 0.7;
      data[o + 15] = rnd();
    }

    var buf = gl.createBuffer();
    gl.bindBuffer(gl.ARRAY_BUFFER, buf);
    gl.bufferData(gl.ARRAY_BUFFER, data, gl.STATIC_DRAW);

    var BPF = 4;
    ["aP0", "aP1", "aP2", "aP3", "aP4", "aP5"].forEach(function (name, idx) {
      var loc = gl.getAttribLocation(prog, name);
      gl.enableVertexAttribArray(loc);
      gl.vertexAttribPointer(loc, 2, gl.FLOAT, false, STRIDE * BPF, idx * 2 * BPF);
    });
    var mLoc = gl.getAttribLocation(prog, "aMeta");
    gl.enableVertexAttribArray(mLoc);
    gl.vertexAttribPointer(mLoc, 4, gl.FLOAT, false, STRIDE * BPF, 12 * BPF);

    var U = {};
    ["uRes", "uTime", "uMouse", "uPhase", "uOrigin", "uScale", "uPointScale"]
      .forEach(function (name) {
        U[name] = gl.getUniformLocation(prog, name);
      });
    // Uniform arrays are addressed by their first element; querying the bare name
    // works on most drivers but not all, so ask for the form that is specified.
    U.uWFrom = gl.getUniformLocation(prog, "uWFrom[0]");
    U.uWTo = gl.getUniformLocation(prog, "uWTo[0]");

    // Premultiplied alpha, so overlapping points accumulate light instead of
    // flattening each other. Dense cluster cores glowing is the point.
    gl.disable(gl.DEPTH_TEST);
    gl.enable(gl.BLEND);
    gl.blendFunc(gl.ONE, gl.ONE_MINUS_SRC_ALPHA);

    var reduced = window.matchMedia("(prefers-reduced-motion: reduce)").matches;

    state = {
      canvas: canvas,
      gl: gl,
      raf: 0,
      visible: true,
      mouse: [0, 0],
      target: [0, 0],
      t0: performance.now(),
      io: null
    };

    // The screenshot underneath is the no-WebGL fallback; now that there is
    // something better to look at, fade it out.
    if (hero) hero.classList.add("bc-hero--webgl");

    var wFrom = new Float32Array(N_LAYOUTS);
    var wTo = new Float32Array(N_LAYOUTS);

    function resize() {
      // Cap the device pixel ratio: nine thousand additive sprites at 3x on a 4K
      // panel is a lot of fragments for no visible gain.
      var dpr = Math.min(window.devicePixelRatio || 1, 2);
      var w = Math.max(1, Math.round(canvas.clientWidth * dpr));
      var h = Math.max(1, Math.round(canvas.clientHeight * dpr));
      if (canvas.width !== w || canvas.height !== h) {
        canvas.width = w;
        canvas.height = h;
      }
      gl.viewport(0, 0, w, h);
      gl.uniform2f(U.uRes, w, h);

      // Coordinates are normalised by height, so x spans +/- aspect. Sit the
      // cloud to the right of the copy on wide screens. Narrow screens have no
      // room beside the copy, so it rides above it as a band rather than sitting
      // behind the text where it would either be invisible or in the way.
      var aspect = w / h;
      var isWide = window.innerWidth >= 900;
      gl.uniform2f(U.uOrigin, isWide ? aspect * 0.40 : 0.0, isWide ? 0.0 : 0.46);
      gl.uniform1f(U.uScale, isWide ? 0.80 : 0.52);
      gl.uniform1f(U.uPointScale, dpr * (isWide ? 2.3 : 1.9));
    }

    function frame(now) {
      state.raf = requestAnimationFrame(frame);
      if (!state.visible) return;

      var t = reduced ? 2.0 : (now - state.t0) / 1000;

      state.mouse[0] += (state.target[0] - state.mouse[0]) * 0.045;
      state.mouse[1] += (state.target[1] - state.mouse[1]) * 0.045;

      // Which two layouts are live, and how far between them.
      var step = Math.floor(t / STEP_DUR);
      var local = t - step * STEP_DUR;
      var phase = local < HOLD_DUR ? 0 : (local - HOLD_DUR) / TRANS_DUR;

      var from = ((step % N_LAYOUTS) + N_LAYOUTS) % N_LAYOUTS;
      var to = (from + 1) % N_LAYOUTS;
      for (var j = 0; j < N_LAYOUTS; j++) {
        wFrom[j] = 0;
        wTo[j] = 0;
      }
      wFrom[from] = 1;
      wTo[to] = 1;

      gl.uniform1f(U.uTime, t);
      gl.uniform2f(U.uMouse, state.mouse[0], state.mouse[1]);
      gl.uniform1fv(U.uWFrom, wFrom);
      gl.uniform1fv(U.uWTo, wTo);
      gl.uniform1f(U.uPhase, phase);

      gl.clearColor(0, 0, 0, 0);
      gl.clear(gl.COLOR_BUFFER_BIT);
      gl.drawArrays(gl.POINTS, 0, n);

      // Reduced motion: draw one frame, then stop entirely.
      if (reduced) {
        cancelAnimationFrame(state.raf);
        state.raf = 0;
      }
    }

    state.onResize = function () {
      resize();
      if (reduced) requestAnimationFrame(frame);
    };
    state.onMove = function (e) {
      state.target[0] = (e.clientX / window.innerWidth) * 2 - 1;
      state.target[1] = 1 - (e.clientY / window.innerHeight) * 2;
    };

    window.addEventListener("resize", state.onResize);
    window.addEventListener("pointermove", state.onMove, { passive: true });

    // The hero scrolls away with the page rather than being pinned, so an
    // IntersectionObserver is an honest answer to "can anyone still see this?"
    // and stops the loop dead once it is off screen.
    if ("IntersectionObserver" in window && hero) {
      state.io = new IntersectionObserver(function (entries) {
        state.visible = entries[0].isIntersecting;
      });
      state.io.observe(hero);
    }

    resize();
    state.raf = requestAnimationFrame(frame);
  }

  function boot() {
    init();
    var container = document.querySelector("[data-md-component=container]");
    if (container && "MutationObserver" in window) {
      // Instant navigation swaps the container wholesale, taking the hero with
      // it. Rather than hook into the theme's internals, just watch the DOM.
      new MutationObserver(function () {
        init();
      }).observe(container.parentNode || document.body, { childList: true, subtree: true });
    }
  }

  if (document.readyState === "loading") {
    document.addEventListener("DOMContentLoaded", boot);
  } else {
    boot();
  }
})();
