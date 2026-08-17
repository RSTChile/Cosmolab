import fs from 'node:fs';
import { correrBarrido } from './correr_barrido.mjs';

const [, , inputPath, outputPath] = process.argv;
const jobs = JSON.parse(fs.readFileSync(inputPath, 'utf8'));
const allRows = [];
for (const job of jobs) {
  allRows.push(...correrBarrido(job));
}
fs.writeFileSync(outputPath, JSON.stringify(allRows));
