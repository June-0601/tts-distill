/**
 * Pure ESM (browser Web Worker & Node compatible, zero external dependencies) runtime helpers
 * for zh_CN-kokoro_zh_multi-medium.onnx:
 *  - ZH_PIPER_INFERENCE_DEFAULTS: tightened length_scale & noise_w to curb random phoneme stalling
 *  - normalizeZhPunctuation: preserves line breaks, folds repeated punctuation, lightens dunhao
 *  - splitZhClauses: splits long Chinese paragraphs cleanly along sentences / balanced clauses
 *  - compressInternalSilence: smooth RMS silence-cap with raised-cosine crossfade (eliminates 1s stalls)
 */

export const ZH_PIPER_INFERENCE_DEFAULTS = Object.freeze({
  noiseScale: 0.45,
  lengthScale: 0.96,
  noiseW: 0.25,
});

const LINE_END_PUNCT_RE = /[。？！，、：；.?!,;:][”’"')）】》]*$/;

/**
 * 中文标点与换行规整，减少 Piper 蒸馏模型的错误停顿与叠乘静音：
 * 1. 无标点结尾的换行补逗号，防止标题/段落直接黏连；
 * 2. 折叠连续破折号/省略号/重复逗号（——、……、...），防止连发多个 72/69 长停顿；
 * 3. 在当前蒸馏模型中顿号「、」映射到 72 会造成 ~0.9s 卡顿，短列举处默认去掉（两边均为 >=6 字长分句时保留为停顿）；
 * 4. 合并相邻的停顿标点（如「：，」「，、」）。
 */
export function normalizeZhPunctuation(rawText, { keepDunhao = false } = {}) {
  let s = String(rawText || '').replace(/\r\n?/g, '\n');
  const lines = s.split('\n').map(l => l.trim()).filter(Boolean);
  if (lines.length > 1) {
    s = lines.map((line, idx) => {
      if (idx === lines.length - 1) return line;
      return LINE_END_PUNCT_RE.test(line) ? line : `${line}，`;
    }).join('');
  } else {
    s = lines[0] || '';
  }

  s = s.replace(/…{2,}|\.{3,}/g, '，');
  s = s.replace(/[—\-]{2,}/g, '，');
  s = s.replace(/[，,]{2,}/g, '，');

  if (!keepDunhao) {
    s = s.replace(/、/g, (_match, offset, full) => {
      const left = full.slice(0, offset).match(/[一-鿿]+$/)?.[0] || '';
      const right = full.slice(offset + 1).match(/^[一-鿿]+/)?.[0] || '';
      return (left.length >= 6 && right.length >= 6) ? '，' : '';
    });
  }

  s = s.replace(/([：:；;，,])\s*([：:；;，,]+)/g, '$1');
  s = s.replace(/([：:；;，,])\s*([。？！.?!])/g, '$2');
  return s.trim();
}

/**
 * 将长段落切分为适合 Piper VITS 朗读的短句/分句列表，避免单次过长推理导致韵律衰减。
 */
export function splitZhClauses(rawText, maxClauseChars = 48) {
  const normalized = normalizeZhPunctuation(rawText);
  if (!normalized) return [];

  const sents = [];
  // 匹配中文句末标点 [。？！?!] 或作为句末使用的英文句点（后跟空白、引号、汉字或结尾，避开 3.14 / U.S.A.）
  const boundaryRe = /[。？！?!]+|(?<![A-Z0-9])\.(?=\s|$|[”’"')）】》一-鿿])/g;
  let lastIdx = 0;
  let m;
  while ((m = boundaryRe.exec(normalized)) !== null) {
    let end = m.index + m[0].length;
    const tail = normalized.slice(end).match(/^[”’"')）】》]+/);
    if (tail) end += tail[0].length;
    const piece = normalized.slice(lastIdx, end).trim();
    if (piece) sents.push(piece);
    lastIdx = end;
    boundaryRe.lastIndex = end;
  }
  const rest = normalized.slice(lastIdx).trim();
  if (rest) sents.push(rest);

  const clauses = [];
  for (const sent of sents) {
    if (sent.length <= maxClauseChars) {
      clauses.push(sent);
      continue;
    }
    const parts = sent.split(/([；;：:，,])/).filter(Boolean);
    let cur = '';
    for (let i = 0; i < parts.length; i++) {
      const p = parts[i];
      if (/^[；;：:，,]$/.test(p)) {
        cur += p;
        if (cur.length >= Math.floor(maxClauseChars * 0.55)) {
          clauses.push(cur.trim());
          cur = '';
        }
      } else if (cur && cur.length + p.length > maxClauseChars && cur.length >= 8) {
        clauses.push(cur.trim());
        cur = p;
      } else {
        cur += p;
      }
    }
    if (cur.trim()) clauses.push(cur.trim());
  }
  return clauses;
}

/**
 * 统计 PCM 波形中的静音停顿段（>= minPauseMs），便于自检停顿时长。
 */
export function analyzePauses(pcm, sampleRate = 22050, { threshold = 0.01, minPauseMs = 60, frameMs = 10 } = {}) {
  const hop = Math.max(1, Math.round((sampleRate * frameMs) / 1000));
  const totalSamples = pcm.length;
  let nzStart = 0;
  while (nzStart < totalSamples && Math.abs(pcm[nzStart]) <= 0.005) nzStart++;
  let nzEnd = totalSamples - 1;
  while (nzEnd > nzStart && Math.abs(pcm[nzEnd]) <= 0.005) nzEnd--;
  const nFrames = Math.floor((nzEnd - nzStart + 1) / hop);
  const gaps = [];
  let run = 0;
  let startFrame = 0;
  const minFrames = Math.max(1, Math.round(minPauseMs / frameMs));
  for (let f = 0; f < nFrames; f++) {
    const base = nzStart + f * hop;
    let sq = 0;
    for (let k = 0; k < hop; k++) {
      const v = pcm[base + k];
      sq += v * v;
    }
    const rms = Math.sqrt(sq / hop);
    if (rms < threshold) {
      if (run === 0) startFrame = f;
      run++;
    } else {
      if (run >= minFrames) {
        gaps.push({ atSec: +(startFrame * frameMs / 1000).toFixed(2), durSec: +(run * frameMs / 1000).toFixed(2) });
      }
      run = 0;
    }
  }
  return { durSec: +((nzEnd - nzStart + 1) / sampleRate).toFixed(2), gaps };
}

/**
 * 压缩合成 PCM 中的过长首尾及句内死静音（支持带淡入淡出的平滑拼接，消除 1s 停顿与句中卡顿）。
 */
export function compressInternalSilence(pcm, sampleRate = 22050, options = {}) {
  const {
    frameMs = 10,
    threshold = 0.008,
    leadMs = 35,
    tailMs = 80,
    minSilenceMs = 130,
    maxPauseMs = 200,
    targetPauseMs = 180,
    fadeMs = 8,
  } = options;

  if (!pcm || pcm.length === 0) return pcm;
  const hop = Math.max(1, Math.round((sampleRate * frameMs) / 1000));
  const nFrames = Math.floor(pcm.length / hop);
  if (nFrames < 6) return pcm;

  const isSil = new Uint8Array(nFrames);
  for (let f = 0; f < nFrames; f++) {
    const base = f * hop;
    let sq = 0;
    for (let k = 0; k < hop; k++) {
      const v = pcm[base + k];
      sq += v * v;
    }
    isSil[f] = Math.sqrt(sq / hop) < threshold ? 1 : 0;
  }

  const leadFrames = Math.max(0, Math.round(leadMs / frameMs));
  const tailFrames = Math.max(0, Math.round(tailMs / frameMs));
  const minSilFrames = Math.max(1, Math.round(minSilenceMs / frameMs));
  const maxPauseFrames = Math.max(1, Math.round(maxPauseMs / frameMs));
  const targetFrames = Math.max(1, Math.min(maxPauseFrames, Math.round(targetPauseMs / frameMs)));
  const fadeSamples = Math.min(Math.round((sampleRate * fadeMs) / 1000), Math.floor((targetFrames * hop) / 2));

  const chunks = [];
  let totalOut = 0;
  let i = 0;

  while (i < nFrames) {
    const sil = isSil[i];
    let j = i + 1;
    while (j < nFrames && isSil[j] === sil) j++;
    const run = j - i;

    if (!sil) {
      const seg = pcm.subarray(i * hop, j * hop);
      chunks.push(seg);
      totalOut += seg.length;
    } else if (i === 0) {
      const keep = Math.min(run, leadFrames);
      if (keep > 0) {
        const seg = pcm.subarray((j - keep) * hop, j * hop);
        chunks.push(seg);
        totalOut += seg.length;
      }
    } else if (j === nFrames) {
      const keep = Math.min(run, tailFrames);
      if (keep > 0) {
        const seg = pcm.subarray(i * hop, (i + keep) * hop);
        chunks.push(seg);
        totalOut += seg.length;
      }
    } else if (run > maxPauseFrames && run >= minSilFrames) {
      const leftF = Math.floor(targetFrames / 2);
      const rightF = targetFrames - leftF;
      const leftSeg = pcm.subarray(i * hop, (i + leftF) * hop);
      const rightSeg = pcm.subarray((j - rightF) * hop, j * hop);
      if (fadeSamples > 0 && leftSeg.length >= fadeSamples && rightSeg.length >= fadeSamples) {
        const merged = new Float32Array(leftSeg.length + rightSeg.length);
        merged.set(leftSeg, 0);
        merged.set(rightSeg, leftSeg.length);
        const mid = leftSeg.length;
        for (let k = 0; k < fadeSamples; k++) {
          const wOut = 0.5 * (1 + Math.cos((Math.PI * (k + 1)) / fadeSamples));
          const wIn = 1 - wOut;
          merged[mid - fadeSamples + k] *= wOut;
          merged[mid + k] *= wIn;
        }
        chunks.push(merged);
        totalOut += merged.length;
      } else {
        chunks.push(leftSeg, rightSeg);
        totalOut += leftSeg.length + rightSeg.length;
      }
    } else {
      const seg = pcm.subarray(i * hop, j * hop);
      chunks.push(seg);
      totalOut += seg.length;
    }
    i = j;
  }

  const out = new Float32Array(totalOut);
  let off = 0;
  for (const c of chunks) {
    out.set(c, off);
    off += c.length;
  }
  return out;
}
