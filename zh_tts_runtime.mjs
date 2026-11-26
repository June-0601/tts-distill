/**
 * Pure ESM (browser Web Worker & Node compatible, zero external dependencies) runtime helpers
 * for zh_CN-kokoro_zh_multi-medium.onnx and zh_CN-kokoro_zh_v2-medium.onnx (see zhV2Settings):
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

/**
 * zh_CN-kokoro_zh_v2-medium（v2：CosyVoice3 / Qwen3-TTS 老师 + 电子书句块语料 + 中文词边界）。
 * 用 zhV2Settings(config) 按模型配置取前端参数：
 *  - wordBoundary：传给 pinyinTokens(text, { wordBoundary: true })，在词与词之间插 '#'（ID 116，
 *    分词用浏览器/Node 自带的 Intl.Segmenter('zh', {granularity: 'word'})）；不传则模型会读得发紧、断句乱；
 *  - maxClauseChars：splitZhClauses 的分句上限。v2 用 15–110 字的整句块训练，切太碎反而丢掉它学到的句内语气；
 *  - silence：compressInternalSilence 的参数。v2 句内停顿是老师的自然换气（中位约 0.3s），
 *    只压真正的死停顿（>450ms → 380ms），不要用旧模型 200→180ms 的激进压缩。
 */
export const ZH_V2_RUNTIME = Object.freeze({
  maxClauseChars: 100,
  silence: Object.freeze({ maxPauseMs: 450, targetPauseMs: 380, minSilenceMs: 200, tailMs: 120 }),
});

export function zhV2Settings(modelConfig) {
  const fe = modelConfig?.zh_frontend;
  if (!fe?.word_boundary) {
    return { wordBoundary: false, maxClauseChars: 48, silence: {} };  // 旧 zh_multi：保持原行为
  }
  return { wordBoundary: true, maxClauseChars: ZH_V2_RUNTIME.maxClauseChars, silence: { ...ZH_V2_RUNTIME.silence } };
}

export const EN_PIPER_INFERENCE_DEFAULTS = Object.freeze({
  noiseScale: 0.5,
  lengthScale: 1.0,
  noiseW: 0.6,
});

const EN_AFFIX_PAREN_RE = /(?<=[A-Za-z])\(\s*(s|es|ies|ren|ed|ing)\s*\)/gi;

/**
 * 英文标点与括号停顿规整（供 Piper 英文/混排 espeak-ng 前端及 Kokoro 复用）：
 * espeak-ng 默认会直接吞掉 '(' ')' '[' ']' 而不生成任何停顿音素；
 * 将括号映射为 ',' 停顿，并在 '1996).' / 'Wait: (' / '(Note)' 处自动去重与收尾，
 * 同时保留单词词尾变体如 author(s) 不被误拆成 "author, s"。
 */
export function normalizeEnPunctuation(rawText, { keepRoundBrackets = false } = {}) {
  let s = String(rawText || '').replace(/\r\n?/g, '\n');
  s = s.replace(/([A-Za-z0-9,;:\-"'])\n+(?=[A-Za-z0-9("'“])/g, '$1 ');
  s = s.replace(/([.?!])\n+/g, '$1 ');
  s = s.replace(/\n+/g, '. ');

  const affixes = [];
  s = s.replace(EN_AFFIX_PAREN_RE, m => {
    const token = `\uE010${affixes.length}\uE011`;
    affixes.push(m);
    return token;
  });

  const openClass = keepRoundBrackets ? '\\[\\uFF08\\u3010' : '(\\[\\uFF08\\u3010';
  const closeClass = keepRoundBrackets ? '\\]\\uFF09\\u3011' : ')\\]\\uFF09\\u3011';
  const pauseSep = keepRoundBrackets ? ', ' : '; ';
  s = s.replace(new RegExp(`\\s*([${openClass}])\\s*`, 'g'), (_, _br, offset, full) => {
    const prev = full.slice(0, offset);
    return (!prev.trim() || /[.?!,;:\-—(["'“‘]\s*$/.test(prev)) ? ' ' : pauseSep;
  });
  s = s.replace(new RegExp(`\\s*([${closeClass}])\\s*`, 'g'), (match, _br, offset, full) => {
    const after = full.slice(offset + match.length);
    return (!after.trim() || /^[.?!,;:)\]"'”’]/.test(after)) ? '' : pauseSep;
  });

  s = s.replace(/\s+([.?!,;:])/g, '$1');
  s = s.replace(/[ \t]{2,}/g, ' ');
  s = s.replace(/^[;,\s]+|[;,\s]+$/g, '');

  if (affixes.length) {
    s = s.replace(/\uE010(\d+)\uE011/g, (_, idx) => affixes[+idx]);
  }
  return s;
}

const ZH_DIGITS = ['零', '一', '二', '三', '四', '五', '六', '七', '八', '九'];

export function intToZh(n) {
  if (n === 0) return '零';
  if (n < 0) return '负' + intToZh(-n);
  if (n >= 100000000) {
    const q = Math.floor(n / 100000000), r = n % 100000000;
    return intToZh(q) + '亿' + (r ? (r < 10000000 ? '零' : '') + intToZh(r) : '');
  }
  if (n >= 10000) {
    const q = Math.floor(n / 10000), r = n % 10000;
    return intToZh(q) + '万' + (r ? (r < 1000 ? '零' : '') + intToZh(r) : '');
  }
  const units = ['', '十', '百', '千'];
  const s = String(n);
  let out = '';
  let zero = false;
  for (let i = 0; i < s.length; i++) {
    const d = +s[i];
    const pos = s.length - 1 - i;
    if (d === 0) {
      zero = true;
    } else {
      if (zero) {
        out += '零';
        zero = false;
      }
      if (!(d === 1 && pos === 1 && out === '')) out += ZH_DIGITS[d];
      out += units[pos];
    }
  }
  return out;
}

export function zhNumStr(numStr) {
  if (numStr.includes('.')) {
    const [a, b] = numStr.split('.');
    return intToZh(+a) + '点' + b.split('').map(d => ZH_DIGITS[+d]).join('');
  }
  return numStr.length > 8 ? numStr.split('').map(d => ZH_DIGITS[+d]).join('') : intToZh(+numStr);
}

/** 将中文上下文中的日期、温度、百分数、分数、年份、负数与阿拉伯数字规范化为中文读法。 */
export function normalizeZhNumbers(text) {
  const s = String(text || '');
  let res = s.replace(/\b(\d{4})[-/](\d{1,2})[-/](\d{1,2})\b/g, (_, y, mo, d) => {
    const yZh = y.split('').map(c => ZH_DIGITS[+c]).join('');
    return `${yZh}年${intToZh(+mo)}月${intToZh(+d)}日`;
  });
  res = res.replace(/([-−])?(\d+(?:\.\d+)?)\s*(?:°\s*([CcFf])|([℃℉]))/g, (_, sign, num, degUnit, symUnit) => {
    const unit = ((degUnit || '').toUpperCase() === 'F' || symUnit === '℉') ? '华氏度' : '摄氏度';
    return (sign ? '零下' : '') + zhNumStr(num) + unit;
  });
  res = res.replace(/([-−])?(\d+(?:\.\d+)?)\s*[%％]/g, (_, sign, num) => {
    return (sign ? '负' : '') + '百分之' + zhNumStr(num);
  });
  res = res.replace(/(?<![A-Za-z0-9./\-])(\d+)\s*\/\s*(\d+)(?![A-Za-z0-9./\-])/g, (_, num, den) => {
    return `${zhNumStr(den)}分之${zhNumStr(num)}`;
  });
  res = res.replace(/\b(\d{4})(?=\s*年)/g, (_, y) => y.split('').map(d => ZH_DIGITS[+d]).join(''));
  res = res.replace(/(?<![A-Za-z0-9.])([-−])(\d+(?:\.\d+)?)/g, (_, _sign, num) => '负' + zhNumStr(num));
  res = res.replace(/\d+(\.\d+)?/g, m => zhNumStr(m));
  return res;
}

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
  // 括号（如「中文翻译（English Name）」）转成明确停顿边界，随后统一做邻近/首尾停顿归并
  s = s.replace(/[（(【\[]/g, '，').replace(/[）)】\]]/g, '，');
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
  return s.replace(/^[，,\s]+|[，,\s]+$/g, '').trim();
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
