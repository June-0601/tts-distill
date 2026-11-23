/**
 * Browser Web Worker / kokoro-js & Node compatible text normalizers and audio post-processor
 * for the original Kokoro models (Kokoro-82M English + Kokoro-82M-v1.1-zh Chinese).
 *
 * Zero external npm/node dependencies. Shares punctuation/pause/compression primitives
 * with ../zh_tts_runtime.mjs so browser workers can import directly.
 */

import {
  analyzePauses,
  compressInternalSilence,
  normalizeEnPunctuation,
  normalizeZhNumbers,
  normalizeZhPunctuation,
  splitZhClauses,
} from '../zh_tts_runtime.mjs';

export { analyzePauses, normalizeEnPunctuation, normalizeZhPunctuation };

/**
 * 规整英文中的大写 'A'：
 * 原 Kokoro（尤其网页 kokoro-js 无 spaCy 词性标注时）会把句首或标题中大写的量词/冠词
 * 「A tree」「A woman」命中词典中的 'A' -> /ˈeɪ/ 误读为字母 A。
 * 本函数保留明确的序号/字母/等级/缩写语境中的大写 'A'，其余冠词位置的 'A' 规范化为 'a'，
 * 确保被稳定读作不定冠词 /ə/（ɐ）。
 */
const LETTER_A_BEFORE = new RegExp(
  '^(?:option|choice|part|section|group|answer|class|grade|team|level|step|case|type|plan|'
  + 'vitamin|category|item|appendix|figure|fig\\.?|table|column|row|zone|phase|stage|tier|version|'
  + 'model|series|track|exit|gate|terminal|block|building|wing|room|unit|seat|line|route|'
  + 'channel|point|node|side|factor|method|approach|sample|set|list|letter|exhibit|form|'
  + 'schedule|an|letter|hepatitis|marked|rated|scored|grade|from|between)$',
  'i'
);

export function normalizeKokoroEnText(rawText) {
  const s = String(rawText || '').replace(/\r\n?/g, '\n');
  const trimmed = s.trim();
  if (/^["'(]*A["').!?\s]*$/.test(trimmed)) return s;

  // 在折叠换行前做 'A' 消歧，避免上一行结尾词如 "First item\nA bird" 误触发 LETTER_A_BEFORE
  const disambiguated = s.replace(/(?<![A-Za-z0-9'.\-])A(?![A-Za-z0-9'\-])/g, (match, offset, full) => {
    const before = full.slice(0, offset);
    const after = full.slice(offset + 1);

    // 紧跟连字符/加号/括号/句点/冒号（序号或缩写，如 A-rated / A+ / A. First / A) / A:）
    if (/^\s*$/.test(after) && /^[\s(\["']*$/.test(before)) return match;
    if (/^[)+:/]/.test(after) || /^\.(?:\s|\d|&|$)/.test(after)) return match;
    if (/[(\[]$/.test(before) && /^[ \t]*[)\]]/.test(after)) return match;

    // 紧跟 "or B" / "and B" / "& B" / "vs B" / "to B" / ", B"
    if (/^[ \t]*(?:&\s*[A-Z]|(?:or|and|vs\.?|versus|to|through|,)[ \t]+[B-Z])\b/.test(after)) return match;
    // 作为字母主语（如 "A stands for ..." / "A is the first letter"）
    if (/^[ \t]+(?:stands[ \t]+for|represents|denotes|equals|comes[ \t]+before|is[ \t]+the[ \t]+(?:first|letter))\b/i.test(after)) {
      return match;
    }

    // 同行前面紧邻明确编号/字母提示词（Option A / Plan A / an A）
    const prevWord = before.match(/([A-Za-z]+\.?)[ \t]+$/)?.[1];
    if (prevWord && LETTER_A_BEFORE.test(prevWord)) return match;
    if (/(?:\bbetween[ \t]+[B-Z][ \t]+and|\bfrom[ \t]+[B-Z][ \t]+to|(?:grade|score|blood[ \t]+(?:type|group)|option|choice)[ \t]+(?:is|was))[ \t]+$/i.test(before)) {
      return match;
    }

    // 句尾孤立（非前面冠词修饰名词）不随意改动
    if (/^[ \t]*(?:[.!?'"”)]*[ \t]*(?:\n|$))/.test(after)) return match;

    return 'a';
  });

  // Kokoro-82M 英文词表原生支持 '(' ')'（~0.25s 自然停顿），同时折叠句内换行并把 '[' ']' '（' '）' '【' '】' 转为逗号停顿
  return normalizeEnPunctuation(disambiguated, { keepRoundBrackets: true });
}

/**
 * 规整送入原 Kokoro 中文的文本：
 * - 修复 -5°C / 85% / 1/3 / YYYY-MM-DD，避免 Kokoro 转成 ❓
 * - 把中文/英文括号、无标点换行、连续破折号/省略号规整成自然停顿
 */
export function normalizeKokoroZhText(rawText, options = {}) {
  const s = normalizeZhPunctuation(rawText, options);
  if (!s) return '';
  return /[一-鿿°℃％%]/.test(s) ? normalizeZhNumbers(s) : s;
}

/**
 * 针对 Kokoro KPipeline 的中文分句切片（修复 kokoro 只按英文 .!? 切分、忽略 。？！ 的 bug）。
 */
export function splitKokoroZhText(rawText, maxChars = 65) {
  const norm = normalizeKokoroZhText(rawText);
  return splitZhClauses(norm, maxChars);
}

/**
 * 针对原 Kokoro（24kHz PCM）的静音上限压缩与平滑拼接。
 */
export function compressKokoroSilence(pcm, sampleRate = 24000, options = {}) {
  return compressInternalSilence(pcm, sampleRate, {
    maxPauseMs: 200,
    targetPauseMs: 180,
    leadMs: 35,
    tailMs: 80,
    ...options,
  });
}
