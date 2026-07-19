/**
 * Simulates app.js tokenizer + seed matching against graph.json.
 * Run: node test_seed_match.js
 */
"use strict";

const fs = require("fs");
const path = require("path");

const EN_STOP = new Set(
  `a about above after again against all am an and any are as at be because been
  before being below between both but by can could did do does doing don down during
  each few for from further had has have having he her here hers herself him himself
  his how i if in into is it its itself just ll me more most my myself no nor not
  now of off on once only or other our ours ourselves out over own re s same she
  should so some such t than that the their theirs them themselves then there these
  they this those through to too under until up ve very was we were what when where
  which while who whom why will with would you your yours yourself yourselves
  also get got like make made use used using one two three new way may well much
  many still even back go going come came take took see saw say said says tell
  told think thought know knew want need let put find give given keep look looking
  something anything everything nothing someone anyone everyone into onto upon`.split(
    /\s+/
  )
);

function tokenize(text) {
  const raw = text.toLowerCase().match(/[a-z][a-z0-9_-]{2,}/g) || [];
  return raw.filter((t) => t.length >= 3 && !EN_STOP.has(t));
}

function tokenVariants(tok) {
  const variants = new Set([tok]);
  if (tok.endsWith("s") && tok.length > 4) variants.add(tok.slice(0, -1));
  if (!tok.endsWith("s")) variants.add(`${tok}s`);
  return [...variants];
}

function matchScore(queryTok, nodeId, nodeLabel) {
  const id = nodeId.toLowerCase();
  const label = (nodeLabel || nodeId).toLowerCase();
  const variants = tokenVariants(queryTok);

  for (const v of variants) {
    if (id === v || label === v) return 1.0;
  }
  for (const v of variants) {
    if (id.startsWith(v) || label.startsWith(v) || v.startsWith(id) || v.startsWith(label)) {
      return 0.75;
    }
  }
  for (const v of variants) {
    if (id.includes(v) || label.includes(v) || v.includes(id) || v.includes(label)) {
      return 0.6;
    }
  }
  return 0;
}

function seedActivation(queryTokens, nodes) {
  const act = new Map();
  for (const qt of queryTokens) {
    for (const n of nodes) {
      const score = matchScore(qt, n.id, n.label);
      if (score > 0) {
        act.set(n.id, Math.max(act.get(n.id) || 0, score));
      }
    }
  }
  return act;
}

function buildAdjacency(edges) {
  const adj = new Map();
  for (const e of edges) {
    for (const [a, b] of [
      [e.source, e.target],
      [e.target, e.source],
    ]) {
      if (!adj.has(a)) adj.set(a, []);
      adj.get(a).push({ nb: b, wRaw: e.w_raw, lastTick: e.last_tick });
    }
  }
  return adj;
}

const EPS = 0.05;
const TAU = 50;

function effWeight(wRaw, lastTick, curTick) {
  const dt = Math.max(0, curTick - lastTick);
  return EPS + (wRaw - EPS) * Math.exp(-dt / TAU);
}

function spreadActivation(act, adj, maxHops, tick) {
  let current = new Map(act);
  for (let hop = 0; hop < maxHops; hop++) {
    const nxt = new Map(current);
    for (const [src, srcAct] of current) {
      if (srcAct < 0.01) continue;
      for (const { nb, wRaw, lastTick } of adj.get(src) || []) {
        const ew = effWeight(wRaw, lastTick, tick);
        nxt.set(nb, (nxt.get(nb) || 0) + srcAct * ew * 0.5);
      }
    }
    current = nxt;
  }
  return current;
}

const data = JSON.parse(fs.readFileSync(path.join(__dirname, "graph.json"), "utf8"));
const query = data.demo_query || data.meta.question;
const qtoks = tokenize(query);
const seeds = seedActivation(qtoks, data.nodes);
const adj = buildAdjacency(data.edges);
const tick = data.meta.current_tick;
const hop2 = spreadActivation(seeds, adj, 2, tick);

const pointers = new Map(Object.entries(data.pointers || {}));
const answerSet = new Set(data.answer_line_idxs || []);
const lineScores = new Map();

for (const [concept, score] of [...seeds.entries()].sort((a, b) => b[1] - a[1])) {
  for (const idx of pointers.get(concept) || []) {
    lineScores.set(idx, Math.max(lineScores.get(idx) || 0, score));
  }
}
for (const [concept, score] of hop2) {
  for (const idx of pointers.get(concept) || []) {
    lineScores.set(idx, Math.max(lineScores.get(idx) || 0, score));
  }
}

const answerHits = [...lineScores.keys()].filter((idx) => answerSet.has(idx));

console.log("=== Engram seed-match simulation ===");
console.log("query:", query);
console.log("tokens:", qtoks.join(", "));
console.log(
  "seeds:",
  [...seeds.entries()]
    .sort((a, b) => b[1] - a[1])
    .map(([id, s]) => `${id}(${s.toFixed(2)})`)
    .join(", ")
);
console.log("seed count:", seeds.size);
console.log("lines reached:", lineScores.size);
console.log("answer line hits:", answerHits.length ? answerHits.join(", ") : "(none)");
console.log("graph:", data.nodes.length, "nodes,", data.edges.length, "edges");

if (seeds.size < 2) {
  console.error("FAIL: need >= 2 seeds");
  process.exit(1);
}
if (answerHits.length < 1) {
  console.error("FAIL: need >= 1 answer line");
  process.exit(1);
}
console.log("PASS");
