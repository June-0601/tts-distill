// 供 en_zh_g2p.py 批量调用的中文/混排拼音分段器，直接复用 pinyin_ids.mjs 的词表选项与数字规整。
import { readFileSync } from 'node:fs';
import { pinyin } from 'pinyin-pro';
import { PINYIN_OPTIONS, normalizeNumbers, splitSyllable } from './pinyin_ids.mjs';

const inputPath = process.argv[2] || '/dev/stdin';
const lines = readFileSync(inputPath, 'utf8').split(/\r?\n/).filter(Boolean);

for (const line of lines) {
  const tabIdx = line.indexOf('\t');
  const id = tabIdx >= 0 ? line.slice(0, tabIdx) : '';
  const rawText = tabIdx >= 0 ? line.slice(tabIdx + 1) : line;
  const text = normalizeNumbers(rawText);
  const segs = [];
  const bad = [];

  for (const part of pinyin(text, PINYIN_OPTIONS)) {
    if (part.isZh) {
      const split = splitSyllable(part.pinyin);
      if (!split) {
        bad.push(`${part.origin}:${part.pinyin}`);
        continue;
      }
      segs.push(['z', split[0], split[1], String(part.num || 5)]);
    } else {
      const chunk = part.origin || '';
      if (!chunk) continue;
      const last = segs[segs.length - 1];
      if (last && last[0] === 't') last[1] += chunk;
      else segs.push(['t', chunk]);
    }
  }

  process.stdout.write(`${JSON.stringify({ id, segs, bad })}\n`);
}
