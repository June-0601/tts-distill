// 中英双语文本 → Piper 拼音 + 英文 ARPAbet 音素 → 音素 ID。
//
// 训练标签和网页朗读必须用同一份逻辑：网页接入自训的双语中文女声模型时，
// 把本文件的分词/注音与 tokensToIds 搬进 web_app/src/js/03-tts-shelf-reader/02-tts-piper.js 的 Worker，
// pinyin-pro 版本与 PINYIN_OPTIONS 保持一致。
//
// 音素表设计（共 116 个符号，ID 0..115，小于 chaowen-medium 底模的 num_symbols=256）：
//   1. ID 0..72：与 piper1-gpl 的 phonemize_chinese.py 完全一致（Chaowen / Xiao Ya 底模原样继承）：
//      y、w 算声母；零声母记 Ø；j/q/x 后的 u 记 v；声调 1..5（轻声记 5）；声调和标点后补 PAD(_=0)。
//   2. ID 73..115：英文专用音素（大写 ARPAbet 15 元音 + 24 辅音 + 3 重音 '0/'1/'2 + 词间边界 @sp）：
//      与中文拼音符号完全不相交，避免中英同字母不同音互相串味；
//      英文每个音节重音后自动补 PAD(_=0)，词间用 @sp(+PAD) 分隔（不触发 72 号逗号长停顿），
//      词尾若以辅音结束且后面是标点或句尾，也补一个 PAD(_=0)。
//
// 命令行：node pinyin_ids.mjs <输入.tsv> <输出.jsonl>
//   输入每行 id<TAB>kind<TAB>text；输出每行一个 JSON：
//   {id, tokens, ids, bad, plain}，plain 是不做变调的逐字拼音（给 build_corpus.py 和 Kokoro 的注音对比）。

import { existsSync, readFileSync, writeFileSync } from 'node:fs';
import { dirname, resolve } from 'node:path';
import { fileURLToPath } from 'node:url';
import { pinyin } from 'pinyin-pro';

// 与网页 predictXiao 现用的选项相同（toneSandhi 处理「一」「不」变调）。
export const PINYIN_OPTIONS = { type: 'all', toneType: 'none', v: true, toneSandhi: true };

const INITIALS = ['zh', 'ch', 'sh', 'b', 'p', 'm', 'f', 'd', 't', 'n', 'l', 'g', 'k', 'h',
  'j', 'q', 'x', 'r', 'z', 'c', 's', 'y', 'w'];
const FINALS = new Set(['a', 'o', 'e', 'ai', 'ei', 'ao', 'ou', 'an', 'en', 'ang', 'eng', 'ong',
  'i', 'ia', 'ie', 'iao', 'iu', 'ian', 'in', 'iang', 'ing', 'iong',
  'u', 'ua', 'uo', 'uai', 'ui', 'uan', 'un', 'uang', 'ueng',
  'v', 've', 'van', 'vn', 'er', 'ue']);

export const EN_VOWELS = new Set([
  'AA', 'AE', 'AH', 'AO', 'AW', 'AY', 'EH', 'ER', 'EY', 'IH', 'IY', 'OW', 'OY', 'UH', 'UW',
]);
export const EN_CONSONANTS = new Set([
  'B', 'CH', 'D', 'DH', 'F', 'G', 'HH', 'JH', 'K', 'L', 'M', 'N', 'NG',
  'P', 'R', 'S', 'SH', 'T', 'TH', 'V', 'W', 'Y', 'Z', 'ZH',
]);

// ID 0..72 同 piper1-gpl src/piper/phonemize_chinese.py；ID 73..115 为扩展的英文音素与词边界。
export const PHONEME_TO_ID = {
  '_': [0], '^': [1], '$': [2],
  'Ø': [3], 'b': [4], 'p': [5], 'm': [6], 'f': [7], 'd': [8], 't': [9], 'n': [10], 'l': [11],
  'g': [12], 'k': [13], 'h': [14], 'j': [15], 'q': [16], 'x': [17], 'zh': [18], 'ch': [19],
  'sh': [20], 'r': [21], 'z': [22], 'c': [23], 's': [24], 'y': [25], 'w': [26],
  'a': [27], 'o': [28], 'e': [29], 'ai': [30], 'ei': [31], 'ao': [32], 'ou': [33], 'an': [34],
  'en': [35], 'ang': [36], 'eng': [37], 'ong': [38],
  'i': [39], 'ia': [40], 'ie': [41], 'iao': [42], 'iu': [43], 'ian': [44], 'in': [45],
  'iang': [46], 'ing': [47], 'iong': [48],
  'u': [49], 'ua': [50], 'uo': [51], 'uai': [52], 'ui': [53], 'uan': [54], 'un': [55],
  'uang': [56], 'ueng': [57],
  'v': [58], 've': [59], 'van': [60], 'vn': [61], 'er': [62], 'ue': [63],
  '1': [64], '2': [65], '3': [66], '4': [67], '5': [68],
  '。': [69], '.': [69], '？': [70], '?': [70], '！': [71], '!': [71],
  '—': [72], '…': [72], '、': [72], '，': [72], ',': [72], '：': [72], ':': [72],
  '；': [72], ';': [72], ' ': [72],
  // 英文 ARPAbet 元音 (73..87)
  'AA': [73], 'AE': [74], 'AH': [75], 'AO': [76], 'AW': [77], 'AY': [78], 'EH': [79], 'ER': [80],
  'EY': [81], 'IH': [82], 'IY': [83], 'OW': [84], 'OY': [85], 'UH': [86], 'UW': [87],
  // 英文 ARPAbet 辅音 (88..111)
  'B': [88], 'CH': [89], 'D': [90], 'DH': [91], 'F': [92], 'G': [93], 'HH': [94], 'JH': [95],
  'K': [96], 'L': [97], 'M': [98], 'N': [99], 'NG': [100], 'P': [101], 'R': [102], 'S': [103],
  'SH': [104], 'T': [105], 'TH': [106], 'V': [107], 'W': [108], 'Y': [109], 'Z': [110], 'ZH': [111],
  // 英文重音 (112..114) 与词间连读边界 (115)
  "'0": [112], "'1": [113], "'2": [114],
  '@sp': [115],
};

const PUNCT = new Set(['。', '.', '？', '?', '！', '!', '—', '…', '、', '，', ',', '：', ':', '；', ';']);
const GROUP_END = new Set(['1', '2', '3', '4', '5', "'0", "'1", "'2", '@sp', ...PUNCT, ' ']);

// 常见科技词、缩写及 listen.py 自检词内置表（即使脱离 cmudict.json 单独运行 listen.py 也能工作）
const CUSTOM_EN_DICT = {
  A: ['AH0'], AN: ['AE1', 'N'], THE: ['DH', 'AH0'], OF: ['AH1', 'V'], TO: ['T', 'UW1'],
  IN: ['IH1', 'N'], ON: ['AA1', 'N'], AT: ['AE1', 'T'], BY: ['B', 'AY1'], FOR: ['F', 'AO1', 'R'],
  WITH: ['W', 'IH1', 'DH'], FROM: ['F', 'R', 'AH1', 'M'], OR: ['AO1', 'R'], AND: ['AE1', 'N', 'D'],
  IS: ['IH1', 'Z'], ARE: ['AA1', 'R'], WAS: ['W', 'AA1', 'Z'], WERE: ['W', 'ER1'],
  IPHONE: ['AY1', 'F', 'OW2', 'N'], IPAD: ['AY1', 'P', 'AE2', 'D'], IOS: ['AY1', 'OW2', 'EH1', 'S'],
  MACOS: ['M', 'AE1', 'K', 'OW2', 'EH1', 'S'], WIFI: ['W', 'AY1', 'F', 'AY2'],
  APP: ['AE1', 'P'], APPS: ['AE1', 'P', 'S'], API: ['EY1', 'P', 'IY1', 'AY1'],
  PDF: ['P', 'IY1', 'D', 'IY1', 'EH1', 'F'], CPU: ['S', 'IY1', 'P', 'IY1', 'Y', 'UW1'],
  GPU: ['JH', 'IY1', 'P', 'IY1', 'Y', 'UW1'], WEBGPU: ['W', 'EH1', 'B', 'JH', 'IY1', 'P', 'IY1', 'Y', 'UW1'],
  WASM: ['W', 'AA1', 'Z', 'AH0', 'M'], ONNX: ['AA1', 'N', 'IH0', 'K', 'S'],
  AI: ['EY1', 'AY1'], UI: ['Y', 'UW1', 'AY1'], URL: ['Y', 'UW1', 'AA1', 'R', 'EH1', 'L'],
  VIP: ['V', 'IY1', 'AY1', 'P', 'IY1'], PPT: ['P', 'IY1', 'P', 'IY1', 'T', 'IY1'],
  OK: ['OW2', 'K', 'EY1'], OKAY: ['OW2', 'K', 'EY1'], EMAIL: ['IY1', 'M', 'EY2', 'L'],
  PODCAST: ['P', 'AA1', 'D', 'K', 'AE2', 'S', 'T'], OFFLINE: ['AO1', 'F', 'L', 'AY1', 'N'],
  ONLINE: ['AA1', 'N', 'L', 'AY1', 'N'], DEADLINE: ['D', 'EH1', 'D', 'L', 'AY2', 'N'],
  FEEDBACK: ['F', 'IY1', 'D', 'B', 'AE2', 'K'], WORKFLOW: ['W', 'ER1', 'K', 'F', 'L', 'OW2'],
  UBIQUITOUS: ['Y', 'UW0', 'B', 'IH1', 'K', 'W', 'IH0', 'T', 'AH0', 'S'],
  ALTHOUGH: ['AO2', 'L', 'DH', 'OW1'], THOROUGHLY: ['TH', 'ER1', 'OW0', 'L', 'IY0'],
  VOCABULARY: ['V', 'OW0', 'K', 'AE1', 'B', 'Y', 'AH0', 'L', 'EH2', 'R', 'IY0'],
  PHOTOSYNTHESIS: ['F', 'OW2', 'T', 'OW0', 'S', 'IH1', 'N', 'TH', 'AH0', 'S', 'IH0', 'S'],
  ABANDON: ['AH0', 'B', 'AE1', 'N', 'D', 'AH0', 'N'], COFFEE: ['K', 'AA1', 'F', 'IY0'],
  READING: ['R', 'IY1', 'D', 'IH0', 'NG'], EVERY: ['EH1', 'V', 'ER0', 'IY0'],
  DAY: ['D', 'EY1'], BUILDS: ['B', 'IH1', 'L', 'D', 'Z'], FASTER: ['F', 'AE1', 'S', 'T', 'ER0'],
  THAN: ['DH', 'AE1', 'N'], MEMORIZING: ['M', 'EH1', 'M', 'ER0', 'AY2', 'Z', 'IH0', 'NG'],
  LISTS: ['L', 'IH1', 'S', 'T', 'S'],
};

const LETTER_PHONES = {
  A: ['EY1'], B: ['B', 'IY1'], C: ['S', 'IY1'], D: ['D', 'IY1'], E: ['IY1'],
  F: ['EH1', 'F'], G: ['JH', 'IY1'], H: ['EY1', 'CH'], I: ['AY1'], J: ['JH', 'EY1'],
  K: ['K', 'EY1'], L: ['EH1', 'L'], M: ['EH1', 'M'], N: ['EH1', 'N'], O: ['OW1'],
  P: ['P', 'IY1'], Q: ['K', 'Y', 'UW1'], R: ['AA1', 'R'], S: ['EH1', 'S'], T: ['T', 'IY1'],
  U: ['Y', 'UW1'], V: ['V', 'IY1'], W: ['D', 'AH1', 'B', 'AH0', 'L', 'Y', 'UW0'],
  X: ['EH1', 'K', 'S'], Y: ['W', 'AY1'], Z: ['Z', 'IY1'],
};

let _cmuDict = null;
let _g2pModel = null;

function b64ToFloat32(b64) {
  const buf = Buffer.from(b64, 'base64');
  return new Float32Array(buf.buffer, buf.byteOffset, buf.byteLength / 4);
}

function reshape2D(flat, rows, cols) {
  const out = [];
  for (let i = 0; i < rows; i++) out.push(flat.subarray(i * cols, (i + 1) * cols));
  return out;
}

function ensureEnResources() {
  if (_cmuDict !== null) return;
  _cmuDict = {};
  try {
    const here = dirname(fileURLToPath(import.meta.url));
    const cmuPath = [
      resolve(here, './assets/cmudict.json'),
      resolve(here, '../../cloudflare/offline-tts-runtime/offline-tts/assets/cmudict.json'),
    ].find(p => existsSync(p));
    if (cmuPath) {
      _cmuDict = JSON.parse(readFileSync(cmuPath, 'utf8'));
    }
    const g2pPath = [
      resolve(here, './assets/g2p_model.json'),
      resolve(here, '../../cloudflare/offline-tts-runtime/offline-tts/assets/g2p_model.json'),
    ].find(p => existsSync(p));
    if (g2pPath) {
      const raw = JSON.parse(readFileSync(g2pPath, 'utf8'));
      const m = {};
      for (const name of ['enc_emb', 'enc_w_ih', 'enc_w_hh', 'enc_b_ih', 'enc_b_hh',
        'dec_emb', 'dec_w_ih', 'dec_w_hh', 'dec_b_ih', 'dec_b_hh', 'fc_w', 'fc_b']) {
        const { shape, data } = raw[name];
        const flat = b64ToFloat32(data);
        if (shape.length === 2) {
          m[name] = reshape2D(flat, shape[0], shape[1]);
          m[name + '_shape'] = shape;
        } else {
          m[name] = flat;
        }
      }
      m.g2idx = Object.fromEntries(raw.graphemes.map((g, i) => [g, i]));
      m.idx2p = Object.fromEntries(raw.phonemes.map((p, i) => [i, p]));
      _g2pModel = m;
    }
  } catch (_) {}
}

function gruCell(x, h, w_ih, w_hh, b_ih, b_hh, hiddenDim) {
  const dim2 = hiddenDim * 2, dim3 = hiddenDim * 3;
  const rzn_ih = new Float32Array(dim3);
  const rzn_hh = new Float32Array(dim3);
  for (let i = 0; i < dim3; i++) {
    let s1 = b_ih[i], s2 = b_hh[i];
    const r1 = w_ih[i], r2 = w_hh[i];
    for (let j = 0; j < x.length; j++) s1 += x[j] * r1[j];
    for (let j = 0; j < h.length; j++) s2 += h[j] * r2[j];
    rzn_ih[i] = s1;
    rzn_hh[i] = s2;
  }
  const newH = new Float32Array(hiddenDim);
  for (let i = 0; i < hiddenDim; i++) {
    const r = 1 / (1 + Math.exp(-(rzn_ih[i] + rzn_hh[i])));
    const z = 1 / (1 + Math.exp(-(rzn_ih[hiddenDim + i] + rzn_hh[hiddenDim + i])));
    const n = Math.tanh(rzn_ih[dim2 + i] + r * rzn_hh[dim2 + i]);
    newH[i] = (1 - z) * n + z * h[i];
  }
  return newH;
}

function g2pPredict(word) {
  ensureEnResources();
  const m = _g2pModel;
  if (!m) return null;
  const hiddenDim = m.enc_w_hh_shape[1];
  const chars = word.toLowerCase().replace(/[^a-z]/g, '').split('').concat(['</s>']);
  if (chars.length <= 1) return null;
  let h = new Float32Array(hiddenDim);
  for (const ch of chars) {
    const idx = m.g2idx[ch] ?? m.g2idx['<unk>'];
    h = gruCell(m.enc_emb[idx], h, m.enc_w_ih, m.enc_w_hh, m.enc_b_ih, m.enc_b_hh, hiddenDim);
  }
  let dec = m.dec_emb[2];
  const preds = [];
  for (let step = 0; step < 20; step++) {
    h = gruCell(dec, h, m.dec_w_ih, m.dec_w_hh, m.dec_b_ih, m.dec_b_hh, hiddenDim);
    let maxVal = -Infinity, maxIdx = 0;
    for (let j = 0; j < m.fc_w.length; j++) {
      let logit = m.fc_b[j];
      const row = m.fc_w[j];
      for (let k = 0; k < hiddenDim; k++) logit += h[k] * row[k];
      if (logit > maxVal) { maxVal = logit; maxIdx = j; }
    }
    if (maxIdx === 3) break;
    const p = m.idx2p[maxIdx];
    if (p && p !== '<unk>' && p !== '<pad>') preds.push(p);
    dec = m.dec_emb[maxIdx];
  }
  return preds.length ? preds : null;
}

/** 查找单个英文词的 ARPAbet 音素数组（如 ['K', 'AA1', 'F', 'IY0']）。 */
export function lookupEnglishPhones(rawWord, customDict = null) {
  ensureEnResources();
  const clean = rawWord.replace(/^'+|'+$/g, '');
  if (!clean) return null;
  const upper = clean.toUpperCase();
  const dict = customDict || _cmuDict;
  if (CUSTOM_EN_DICT[upper]) return CUSTOM_EN_DICT[upper];
  if (dict && dict[upper]) return dict[upper];
  // 2~5 位全大写缩写（如 HTML、USB）且不在词典中：逐字母拼读
  if (/^[A-Z]{2,5}$/.test(clean) && !/[aeiou]/i.test(clean)) {
    return clean.split('').flatMap(ch => LETTER_PHONES[ch] || []);
  }
  // 常见所有格或复数变形：word's -> word + Z/S
  if (upper.endsWith("'S") && upper.length > 2) {
    const base = lookupEnglishPhones(clean.slice(0, -2), dict);
    if (base && base.length) {
      const last = base[base.length - 1].replace(/[012]$/, '');
      const sfx = ['S', 'Z', 'SH', 'ZH', 'CH', 'JH'].includes(last) ? ['IH0', 'Z']
        : ['P', 'T', 'K', 'F', 'TH'].includes(last) ? ['S'] : ['Z'];
      return [...base, ...sfx];
    }
  }
  return g2pPredict(clean);
}

/** 将 ARPAbet 音素（如 ['K', 'AA1', 'F', 'IY0']）展开为模型音素序列（['K', 'AA', "'1", 'F', 'IY', "'0"]）。 */
export function expandArpabetPhones(arpabetList) {
  const out = [];
  for (const raw of arpabetList) {
    const m = raw.match(/^([A-Z]+)([012])?$/);
    if (!m) continue;
    const [, base, stress] = m;
    if (EN_VOWELS.has(base)) {
      out.push(base, `'${stress || '0'}`);
    } else if (EN_CONSONANTS.has(base)) {
      out.push(base);
    }
  }
  return out;
}

const ZH_DIGITS = ['零', '一', '二', '三', '四', '五', '六', '七', '八', '九'];
function intToZh(n) {
  if (n === 0) return '零';
  if (n < 0) return '负' + intToZh(-n);
  if (n >= 100000000) return intToZh(Math.floor(n / 100000000)) + '亿' + (n % 100000000 ? (n % 100000000 < 10000000 ? '零' : '') + intToZh(n % 100000000) : '');
  if (n >= 10000) return intToZh(Math.floor(n / 10000)) + '万' + (n % 10000 ? (n % 10000 < 1000 ? '零' : '') + intToZh(n % 10000) : '');
  const units = ['', '十', '百', '千'];
  const s = String(n);
  let out = '', zero = false;
  for (let i = 0; i < s.length; i++) {
    const d = +s[i], pos = s.length - 1 - i;
    if (d === 0) { zero = true; }
    else {
      if (zero) { out += '零'; zero = false; }
      if (!(d === 1 && pos === 1 && out === '')) out += ZH_DIGITS[d];
      out += units[pos];
    }
  }
  return out;
}

const EN_ONES = ['zero', 'one', 'two', 'three', 'four', 'five', 'six', 'seven', 'eight', 'nine',
  'ten', 'eleven', 'twelve', 'thirteen', 'fourteen', 'fifteen', 'sixteen', 'seventeen', 'eighteen', 'nineteen'];
const EN_TENS = ['', '', 'twenty', 'thirty', 'forty', 'fifty', 'sixty', 'seventy', 'eighty', 'ninety'];
function intToEn(n) {
  if (n < 20) return EN_ONES[n];
  if (n < 100) return EN_TENS[Math.floor(n / 10)] + (n % 10 ? ' ' + EN_ONES[n % 10] : '');
  if (n < 1000) return EN_ONES[Math.floor(n / 100)] + ' hundred' + (n % 100 ? ' ' + intToEn(n % 100) : '');
  if (n >= 1900 && n <= 2099 && n % 100 !== 0 && n < 2000) {
    return intToEn(Math.floor(n / 100)) + ' ' + (n % 100 < 10 ? 'oh ' + EN_ONES[n % 100] : intToEn(n % 100));
  }
  if (n < 1000000) return intToEn(Math.floor(n / 1000)) + ' thousand' + (n % 1000 ? ' ' + intToEn(n % 1000) : '');
  return String(n).split('').map(d => EN_ONES[+d]).join(' ');
}

/** 将文本中的阿拉伯数字按语境转为中文数字或英文单词。 */
export function normalizeNumbers(text) {
  const hasZh = /[一-鿿]/.test(text);
  if (hasZh) {
    // 年份如 1998年 -> 一九九八年
    let res = text.replace(/\b(\d{4})(?=\s*年)/g, (_, y) => y.split('').map(d => ZH_DIGITS[+d]).join(''));
    res = res.replace(/\d+(\.\d+)?/g, m => {
      if (m.includes('.')) {
        const [a, b] = m.split('.');
        return intToZh(+a) + '点' + b.split('').map(d => ZH_DIGITS[+d]).join('');
      }
      return m.length > 8 ? m.split('').map(d => ZH_DIGITS[+d]).join('') : intToZh(+m);
    });
    return res;
  }
  return text.replace(/\d+(\.\d+)?/g, m => {
    if (m.includes('.')) {
      const [a, b] = m.split('.');
      return intToEn(+a) + ' point ' + b.split('').map(d => EN_ONES[+d]).join(' ');
    }
    return intToEn(+m);
  });
}

/** 不带声调、ü 写作 v 的拼音音节 → [声母或 Ø, 韵母]；不是正常音节（嗯 ng、呣 m）返回 null。 */
export function splitSyllable(syl) {
  let ini = '';
  for (const c of INITIALS) if (syl.startsWith(c)) { ini = c; break; }
  let fin = ini ? syl.slice(ini.length) : syl;
  if ((ini === 'j' || ini === 'q' || ini === 'x') && fin.startsWith('u')) fin = 'v' + fin.slice(1);
  if (!FINALS.has(fin)) return null;
  return [ini || 'Ø', fin];
}

/** 文本 → 音素列表；支持纯中文、中英混排、纯英文。bad 收集无法拆分的汉字或无法识别的英文词。 */
export function pinyinTokens(rawText, pinyinFn = pinyin, options = PINYIN_OPTIONS, customEnDict = null) {
  const text = normalizeNumbers(String(rawText || ''));
  const tokens = [];
  const bad = [];
  let prevKind = null; // 'zh' | 'en' | 'punct'
  let nonZhBuf = '';

  const flushPunct = slice => {
    for (const ch of slice) {
      if (PUNCT.has(ch)) {
        tokens.push(ch);
        prevKind = 'punct';
      }
    }
  };

  const flushNonZh = () => {
    if (!nonZhBuf) return;
    const seg = nonZhBuf;
    nonZhBuf = '';
    const re = /[A-Za-z]+(?:'[A-Za-z]+)*/g;
    let lastIdx = 0, m;
    while ((m = re.exec(seg)) !== null) {
      flushPunct(seg.slice(lastIdx, m.index));
      const word = m[0];
      const arpabet = lookupEnglishPhones(word, customEnDict);
      const expanded = arpabet ? expandArpabetPhones(arpabet) : [];
      if (!expanded.length) {
        bad.push(`en:${word}`);
      } else {
        if (prevKind === 'en' || prevKind === 'zh') tokens.push('@sp');
        tokens.push(...expanded);
        prevKind = 'en';
      }
      lastIdx = re.lastIndex;
    }
    flushPunct(seg.slice(lastIdx));
  };

  for (const part of pinyinFn(text, options)) {
    if (part.isZh) {
      flushNonZh();
      const split = splitSyllable(part.pinyin);
      if (!split) { bad.push(`${part.origin}:${part.pinyin}`); continue; }
      if (prevKind === 'en') tokens.push('@sp');
      tokens.push(split[0], split[1], String(part.num || 5));
      prevKind = 'zh';
    } else {
      nonZhBuf += part.origin || '';
    }
  }
  flushNonZh();
  return { tokens, bad };
}

/** 音素 → ID：开头 BOS，声调、英文重音、词间 @sp 与标点后补 PAD，结尾 EOS。
 *  若英文单词以辅音结尾且紧跟标点、@sp 或句尾 $，也在该尾辅音后补一个 PAD，保证辅音收尾清晰。 */
export function tokensToIds(tokens, map = PHONEME_TO_ID) {
  const ids = [...map['^']];
  for (let i = 0; i < tokens.length; i++) {
    const t = tokens[i];
    const found = map[t];
    if (!found) continue;
    ids.push(...found);
    const next = tokens[i + 1];
    const isWordFinalConsonant = EN_CONSONANTS.has(t) && (!next || next === '@sp' || PUNCT.has(next));
    if (GROUP_END.has(t) || isWordFinalConsonant) {
      ids.push(...map['_']);
    }
  }
  ids.push(...map['$']);
  return ids;
}

/** 不做变调的逐字拼音（带调号，轻声 5），用来和 Kokoro 的中文注音逐字比对。 */
export function plainSyllables(rawText, pinyinFn = pinyin) {
  const text = normalizeNumbers(String(rawText || ''));
  return pinyinFn(text, { ...PINYIN_OPTIONS, toneSandhi: false })
    .filter(p => p.isZh)
    .map(p => `${p.pinyin}${p.num || 5}`);
}

if (process.argv[1] && import.meta.url.endsWith(process.argv[1].split('/').pop())) {
  const [input, output] = process.argv.slice(2);
  if (!input || !output) {
    console.error('用法：node pinyin_ids.mjs <输入.tsv> <输出.jsonl>');
    process.exit(2);
  }
  const lines = readFileSync(input, 'utf8').split('\n').filter(Boolean);
  const out = lines.map(line => {
    const [id, , text] = line.split('\t');
    const { tokens, bad } = pinyinTokens(text);
    return JSON.stringify({ id, tokens, ids: tokensToIds(tokens), bad, plain: plainSyllables(text) });
  });
  writeFileSync(output, out.join('\n') + '\n');
  console.error(`pinyin_ids: ${out.length} 行 → ${output}`);
}
