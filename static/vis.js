// ------------------------------------------------------
// vis.js — 情绪仪表盘版（大圆 + 条形图, 使用 segments 汇总）
// ------------------------------------------------------

let versions = null;

/**
 * 对一个版本的 segments 做汇总（简单平均）
 * segments: [{tempo, energy, valence, brightness, roughness, loudness, ...}, ...]
 */
function summarizeSegments(segments) {
  if (!segments || segments.length === 0) {
    return {
      tempo: 80,
      loudness: -22,
      energy: 0.5,
      valence: 0.5,
      brightness: 2400,
      roughness: 0.3,
    };
  }

  let tempo = 0;
  let energy = 0;
  let valence = 0;
  let brightness = 0;
  let roughness = 0;
  let loudness = 0;

  for (const s of segments) {
    tempo      += s.tempo      ?? 80;
    energy     += s.energy     ?? 0.5;
    valence    += s.valence    ?? 0.5;
    brightness += s.brightness ?? 2400;
    roughness  += s.roughness  ?? 0.3;
    loudness   += s.loudness   ?? -22;
  }

  const n = segments.length;
  tempo      /= n;
  energy     /= n;
  valence    /= n;
  brightness /= n;
  roughness  /= n;
  loudness   /= n;

  return {
    tempo,
    loudness,
    energy,
    valence,
    brightness,
    roughness,
  };
}

/**
 * 由 main.js 调用，传入完整的 data（自动 or 自定义）
 * data: { versions: [ {name, segments:[...]} ] }
 */
function setVersionsForVis(data) {
  if (!data || !data.versions) {
    versions = null;
    return;
  }

  versions = data.versions.map((v) => {
    if (v.segments && Array.isArray(v.segments)) {
      const stats = summarizeSegments(v.segments);
      return {
        name: v.name || v.slot || "version",
        ...stats,
        segments: v.segments,
        aiColor: v.aiColor || null,
        aiComment: v.aiComment || "",
      };
    } else {
      return v;
    }
  });

  console.log("vis.js summarized versions:", versions);
}

// 暴露给 main.js
window.setVersionsForVis = setVersionsForVis;


/**
 * 默认情绪颜色映射（如果没有 AI 颜色时使用）：
 * - valence 控制色相：低 → 冷蓝，高 → 暖橙
 * - energy (arousal 代理) 控制明度/饱和度：低 → 暗淡，高 → 明亮
 */
function baseEmotionColor(valence, energy) {
  valence = constrain(valence ?? 0.5, 0, 1);
  energy  = constrain(energy  ?? 0.5, 0, 1);

  const cool = [80, 110, 255];   // 低 valence：冷
  const warm = [255, 170, 60];   // 高 valence：暖

  let r = cool[0] + (warm[0] - cool[0]) * valence;
  let g = cool[1] + (warm[1] - cool[1]) * valence;
  let b = cool[2] + (warm[2] - cool[2]) * valence;

  const brightnessScale = map(energy, 0, 1, 0.45, 1.0);
  r *= brightnessScale;
  g *= brightnessScale;
  b *= brightnessScale;

  return [r, g, b];
}

function setup() {
  const parent = document.getElementById("p5-container");
  const canvas = createCanvas(parent.clientWidth, parent.clientHeight);
  canvas.parent("p5-container");
  textFont("system-ui");
}

function windowResized() {
  const parent = document.getElementById("p5-container");
  resizeCanvas(parent.clientWidth, parent.clientHeight);
}

function draw() {
  background(8, 10, 25);

  if (!versions) {
    fill(220);
    textSize(14);
    textAlign(LEFT, TOP);
    text("Waiting for analysis data ...", 16, 16);
    return;
  }

  const n = versions.length;
  const bandW = width / n;

  const centerY   = height * 0.25; // 大圆中心
  const infoTextY = height * 0.52; // 数值信息起始 y
  const barBaseY  = height * 0.88; // 条形图底部 y

  for (let i = 0; i < n; i++) {
    const v = versions[i];
    const cx = (i + 0.5) * bandW;
    const cy = centerY;

    const tempo      = v.tempo      ?? 80;
    const loudness   = v.loudness   ?? -22;
    const energy     = v.energy     ?? 0.5;
    const valence    = v.valence    ?? 0.5;
    const brightness = v.brightness ?? 2400;
    const roughness  = v.roughness  ?? 0.3;

    // 背景竖条
    noStroke();
    fill(20, 25, 60, 160);
    rect(i * bandW, 0, bandW, height);

    // 圆大小：energy 控制
    let baseRadius = map(energy, 0, 1, 45, 85);
    baseRadius = constrain(baseRadius, 40, 90);

    const beatSpeed  = map(tempo, 60, 180, 0.8, 2.2);
    const beatPhase  = frameCount * 0.03 * beatSpeed;
    const beatAmount = 1 + energy * 0.6;
    const pulse      = sin(beatPhase) * baseRadius * 0.2 * beatAmount;
    const radius     = baseRadius + pulse;

    // 颜色：优先 AI 颜色，否则根据 valence/energy 计算
    let colorArr;
    if (Array.isArray(v.aiColor) && v.aiColor.length === 3) {
      colorArr = v.aiColor;
    } else {
      colorArr = baseEmotionColor(valence, energy);
    }
    const [r, g, b] = colorArr;

    const ringSteps = 80;
    const roughAmp  = map(roughness, 0, 1, 0, 10);

    push();
    translate(cx, cy);

    // 外光晕
    noStroke();
    fill(r, g, b, 40);
    ellipse(0, 0, radius * 2.2, radius * 2.2);

    // 外圈毛刺（roughness）
    noFill();
    stroke(r, g, b, 180);
    strokeWeight(2);
    beginShape();
    for (let t = 0; t < TWO_PI; t += TWO_PI / ringSteps) {
      const noiseVal = sin(t * 8 + frameCount * 0.05);
      const extra    = noiseVal * roughAmp;
      const rr       = radius + 6 + extra;
      const x        = rr * cos(t);
      const y        = rr * sin(t);
      vertex(x, y);
    }
    endShape(CLOSE);

    // 填充圆
    noStroke();
    fill(r, g, b, 210);
    ellipse(0, 0, radius * 2, radius * 2);

    pop();

    // 标题（版本名）
    textAlign(CENTER, TOP);
    fill(230);
    textSize(22);
    text(v.name, cx, 16);

    // 数值信息（只保留这几行，不再画 “AI: …” 文字）
    textSize(13);
    fill(190);
    textAlign(CENTER, TOP);
    text(`tempo: ${tempo.toFixed(0)} BPM`, cx, infoTextY);
    text(`loudness: ${loudness.toFixed(1)} dB`, cx, infoTextY + 18);
    text(
      `energy: ${energy.toFixed(2)}, valence: ${valence.toFixed(2)}`,
      cx,
      infoTextY + 36
    );
    text(`roughness: ${roughness.toFixed(2)}`, cx, infoTextY + 54);

    // ===== 条形图 =====
    const tempoNorm  = constrain(map(tempo, 60, 180, 0, 1), 0, 1);
    const energyNorm = constrain(energy, 0, 1);
    const roughNorm  = constrain(roughness, 0, 1);

    const barW = bandW * 0.18;
    const gap  = bandW * 0.07;

    drawFeatureBar(
      cx - barW - gap,
      barBaseY,
      barW,
      map(tempoNorm, 0, 1, 20, 90),
      [250, 200, 120],
      "tempo"
    );

    drawFeatureBar(
      cx,
      barBaseY,
      barW,
      map(energyNorm, 0, 1, 20, 90),
      [120, 230, 180],
      "energy"
    );

    drawFeatureBar(
      cx + barW + gap,
      barBaseY,
      barW,
      map(roughNorm, 0, 1, 20, 90),
      [150, 170, 255],
      "rough"
    );
  }
}

function drawFeatureBar(cx, baseY, w, h, colorArr, label) {
  const [r, g, b] = colorArr;
  const x = cx - w / 2;
  const y = baseY - h;

  // 背景卡片
  noStroke();
  fill(40, 45, 80, 220);
  rect(x, baseY - 95, w, 100, 4);

  // 条
  fill(r, g, b, 220);
  rect(x, y, w, h, 4);

  // 标签
  fill(220);
  textAlign(CENTER, TOP);
  textSize(11);
  text(label, cx, baseY + 8);
}
